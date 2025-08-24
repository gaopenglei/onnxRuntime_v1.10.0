# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import struct
from pathlib import Path
import numpy as np
import logging

import onnx
import onnx.numpy_helper
from onnx import onnx_pb as onnx_proto
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode, QuantizedValueType, QuantizedInitializer, QuantizedValue
from .quant_utils import find_by_name, get_elem_index, get_mul_node, generate_identified_filename, attribute_to_kwarg, type_to_name
from .quant_utils import quantize_nparray, quantize_data, compute_scale_zp, get_qrange_for_qType, get_qmin_qmax_for_qType
from .quant_utils import QuantType, onnx_domain, __producer__, __version__

from .registry import CreateOpQuantizer, CreateDefaultOpQuantizer

from .onnx_model import ONNXModel

#图量化的核心思路是在计算图中插入量化 / 反量化节点，并对权重进行预量化，最终生成一个可直接用于低精度推理的量化模型。

class ONNXQuantizer:
    def __init__(self, model, per_channel, reduce_range, mode, static, weight_qType, input_qType, tensors_range,
                 nodes_to_quantize, nodes_to_exclude, op_types_to_quantize, extra_options={}):

        # run shape inference on the model (enabled by default)
        self.extra_options = extra_options if extra_options is not None else {}
        if not ('DisableShapeInference' in self.extra_options and self.extra_options['DisableShapeInference']):
            model = onnx.shape_inference.infer_shapes(model)  #ONNX 中用于推断模型中所有张量形状信息
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})

        self.model = ONNXModel(model)
        self.per_channel = per_channel  # weight-pack per channel  #是否按通道对权重进行量化（按通道量化通常精度更高，适用于卷积等层）
        self.reduce_range = reduce_range  #是否缩小量化范围（如将权重从 0-255 范围缩小到 1-254，避免极端值影响精度）
        self.mode = mode  # QuantizationMode.Value
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = False
        self.enable_subgraph_quantization = 'EnableSubgraph' in self.extra_options and self.extra_options['EnableSubgraph']
        self.force_quantize_no_input_check = 'ForceQuantizeNoInputCheck' in self.extra_options and self.extra_options['ForceQuantizeNoInputCheck']
        self.q_matmul_const_b_only = 'MatMulConstBOnly' in self.extra_options and self.extra_options['MatMulConstBOnly'] # 若extra_options中存在MatMulConstBOnly且值为True，则仅对MatMul中作为常量的 B 权重进行量化。
        is_weight_int8 = weight_qType == QuantType.QInt8
        self.is_weight_symmetric = is_weight_int8 if 'WeightSymmetric' not in self.extra_options else self.extra_options['WeightSymmetric']
        self.is_activation_symmetric = False if 'ActivationSymmetric' not in self.extra_options else self.extra_options['ActivationSymmetric']  #配置激活值是否使用对称量化：默认False（非对称），

        self.input_qType = onnx_proto.TensorProto.INT8 if input_qType == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
        self.weight_qType = onnx_proto.TensorProto.INT8 if weight_qType == QuantType.QInt8 else onnx_proto.TensorProto.UINT8
        '''
            Dictionary specifying the min and max values for tensors. It has following format:
                {
                    "param_name": [min, max]
                }
            example:
                {
                    'Conv_3:0': [np.float32(0), np.float32(0.5)],
                    'Conv_4:0': [np.float32(1), np.float32(3.5)]
                }

        注释说明tensors_range的格式：一个字典，键为张量名称，值为该张量的 [最小值，最大值]，用于静态量化中计算量化参数（缩放因子、零点）。
        '''
        self.tensors_range = tensors_range
        self.nodes_to_quantize = nodes_to_quantize  # specific nodes to quantize #存储nodes_to_quantize参数：指定需要量化的节点列表（仅这些节点会被量化）。
        self.nodes_to_exclude = nodes_to_exclude  # specific nodes to exclude  #存储nodes_to_exclude参数：指定需要排除的节点列表（这些节点不参与量化）。
        self.op_types_to_quantize = op_types_to_quantize #存储op_types_to_quantize参数：指定需要量化的算子类型（如Conv、MatMul等）。
        self.new_nodes = []  #用于存储量化过程中生成的新节点（如量化 / 反量化节点）
        self.parent = None
        self.graph_scope = "/" # for human readable debug information
        self.tensor_names = { } # in case the shape inference not totally working
        self.tensor_names.update({ot.name: 1 for ot in model.graph.output})
        self.tensor_names.update({it.name: 1 for it in model.graph.input})
        for node in self.model.model.graph.node:
            self.tensor_names.update({output_name: 1 for output_name in node.output})

        self.opset_version = self.check_opset_version() #获取模型的 OPSET 版本：ONNX 的 OPSET 版本决定了支持的算子类型，量化需适配对应版本的算子。

        if not self.mode in QuantizationMode:
            raise ValueError('unsupported quantization mode {}'.format(self.mode))

        self.quantization_params = self.calculate_quantization_params() #计算量化参数（如缩放因子、零点等），并存储在quantization_params中。

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"  #存储值为 0 的张量名称，用于 UInt8 量化中计算零点（零点 = 0 - 最小值）。
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"  #存储值为 0 的张量名称，用于 Int8 量化（对称量化中零点固定为 0）。

        # Map of all original value names to quantized value names
        self.quantized_value_map = {}
        # some output from nodes will be quantized, yet itself should be treat as existing so
        # no dequantized will be applied when needed later
        self.generated_value_names = self.model.get_non_initializer_inputs()

    # routines for subgraph support
    def quantize_subgraph(self, subgraph, graph_key):
        '''
            generate submodel for the subgraph, so that we re-utilize current quantization implementation.
            quantize the submodel
            update subgraph and set it back to node
        '''
        warped_model = onnx.helper.make_model(subgraph, producer_name='onnx-quantizer',
                                              opset_imports=self.model.model.opset_import)
        sub_quanitzer = ONNXQuantizer(warped_model,
                                      self.per_channel,
                                      self.reduce_range,
                                      self.mode,
                                      self.static,
                                      self.weight_qType,
                                      self.input_qType,
                                      self.tensors_range,
                                      self.nodes_to_quantize,
                                      self.nodes_to_exclude,
                                      self.op_types_to_quantize,
                                      self.extra_options)
        sub_quanitzer.parent = self
        sub_quanitzer.graph_scope = "{}{}/".format(self.graph_scope, graph_key)
        sub_quanitzer.quantize_model()
        return sub_quanitzer.model.model.graph

#用于量化包含子图的节点的方法，主要功能是检查节点是否包含子图，若包含，则量化子图并替换原始子图。实现对嵌套子图结构的模型量化支持。
    def quantize_node_with_sub_graph(self, node):                 #node : 待处理的onnx节点  
        '''
        Check subgraph, if any, quantize it and replace it.
        return new_nodes added for quantizing subgraph
        '''
         #从节点的attribute中筛选出类型为GRAPH（单个子图）或GRAPHS（多个子图）的属性，存储在graph_attrs中。
        graph_attrs = [attr for attr in node.attribute if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS]
        if len(graph_attrs) == 0:
            return node
        node_name = node.name if node.name != "" else "{}_node_count_{}".format(node.op_type, len(self.new_nodes))
        kwargs = {}  #初始化空字典kwargs，用于存储节点属性的键值对，供后续创建新节点使用。
        for attr in node.attribute:
            """若属性是单个子图（GRAPH类型）：调用self.quantize_subgraph方法量化该子图（attr.g），子图的作用域名称由节点名称和属性名拼接而成（如 "node_name:attr_name"），
            并将量化后的子图以{属性名: 量化后子图}的形式存入kv。"""
            if attr.type == onnx.AttributeProto.GRAPH:
                kv = {attr.name: self.quantize_subgraph(attr.g, "{}:{}".format(node_name, attr.name))}
            """若属性是子图列表（GRAPHS类型）：初始化空列表value，遍历每个子图，调用self.quantize_subgraph量化子图，子图作用域名称由节点名称、属性名和当前列表长度拼接（确保唯一性，如 "node_name:attr_name:0"），
            将量化后的子图添加到value，最后将子图列表以{属性名: 量化后子图列表}的形式存入kv。"""
            elif attr.type == onnx.AttributeProto.GRAPHS:
                value = []
                for subgraph in attr.graphs:
                    value.extend([self.quantize_subgraph(subgraph, "{}:{}:{}".format(node_name, attr.name, len(value)))])
                kv = {attr.name: value}
            else:
                kv = attribute_to_kwarg(attr)  #保持原始属性不变
            kwargs.update(kv)
        """使用处理后的属性（kwargs）创建新节点：保持原始节点的算子类型（node.op_type）、输入（node.input）、输出（node.output）和名称（node.name）不变，
        但替换其中的子图属性为量化后的子图，返回新创建的节点。"""
        return onnx.helper.make_node(node.op_type, node.input, node.output, name=node.name, **kwargs)

    def check_opset_version(self):
        ai_onnx_domain = [
            opset for opset in self.model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        if 1 != len(ai_onnx_domain):
            raise ValueError('Failed to find proper ai.onnx domain')
        opset_version = ai_onnx_domain[0].version

        if opset_version == 10:
            logging.warning(
                "The original model opset version is {}, which does not support node fusions. Please update the model to opset >= 11 for better performance."
                .format(opset_version))
            return 10

        if opset_version < 10:
            logging.warning(
                "The original model opset version is {}, which does not support quantization. Please update the model to opset >= 11. Updating the model automatically to opset 11. Please verify the quantized model."
                .format(opset_version))
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
            opset_version = 11

        self.fuse_dynamic_quant = True
        return opset_version

"""
这段代码是 ONNX Runtime 中用于移除 “假量化节点对”（QuantizeLinear 和 DequantizeLinear 节点对）的方法，主要用于量化感知训练（Quantization-Aware Training）后的模型处理。
假量化节点在训练中模拟量化效果但实际仍用浮点数计算，推理时需要移除这些节点并恢复真实连接.

其核心逻辑是：识别量化感知训练中插入的QuantizeLinear/DequantizeLinear假量化节点对，通过重新连接计算图（跳过这对节点）移除它们，同时保留量化参数（scale 和 zero_point）供后续真实量化使用。
这一步是将 “训练时模拟量化” 的模型转换为 “推理时真实量化” 模型的关键步骤，确保计算图简洁且量化参数可复用。
"""
    def remove_fake_quantized_nodes(self):
        '''
            Detect and remove the quantize/dequantizelinear node pairs(fake quantized nodes in Quantization-Aware training)
            and reconnect and update the nodes.
        '''
        nodes_to_remove = []  #用于存储待移除的假量化节点
        initializers_to_remove = [] #用于存储这些节点关联的、不再使用的初始化器（如缩放因子、零点）。
 
        for curr_node in self.model.nodes():  #遍历模型中的所有节点，寻找假量化节点对的第一个节点（QuantizeLinear）(原因: QuantizeLinear节点是Q/DQ对的第一个节点)
            if curr_node.op_type == 'QuantizeLinear':
                next_node, prev_node, succ_node = None, None, None  #next_node用于存储对应的DequantizeLinear节点；prev_node存储QuantizeLinear的父节点；succ_node存储DequantizeLinear的后续节点.
                for child_node in self.model.get_children(curr_node):
                    if child_node.op_type == 'DequantizeLinear':   #寻找QuantizeLinear的子节点中类型为DequantizeLinear的节点，这两个节点构成 “假量化节点对”。
                        next_node = child_node
                if next_node is None:
                    raise ValueError(
                        "Remove fake-quantized node pair Error: DequantizeLinear node is not found for {}.".format(
                            curr_node.name))

                prev_node = self.model.get_parent(curr_node, 0)  #获取QuantizeLinear节点的父节点（即QuantizeLinear的输入来源节点，参数0表示取第一个输入的父节点）
                if prev_node is None:
                    raise ValueError("Remove fake-quantized node pair Error: Parent node is not found for {}.".format(
                        curr_node.name))

                succ_nodes = self.model.get_children(next_node)  #获取DequantizeLinear节点的所有子节点（即依赖DequantizeLinear输出的后续节点）
                if len(succ_nodes) == 0:
                    raise ValueError("Remove fake-quantized node pair Error: No successive nodes found for {}.".format(
                        next_node.name))

                # TODO: convert it to the specified input_type
                scale_tensor_name = curr_node.input[1] ##获取QuantizeLinear节点输入中缩放因子（scale）和零点（zero point）的张量名称：QuantizeLinear的输入格式为[输入张量, scale, zero_point]，
                zp_tensor_name = curr_node.input[2]   #因此索引 1 是 scale，索引 2 是 zero_point
                initializer_scale = find_by_name(scale_tensor_name, self.model.initializer())
                initializer_zp = find_by_name(zp_tensor_name, self.model.initializer()) #通过名称从模型的初始化器（initializer）中找到对应的 scale 和 zero_point 张量（初始化器是模型中存储常量的地方，如权重、量化参数）
                zp_and_scale = [
                    onnx.numpy_helper.to_array(initializer_zp),
                    onnx.numpy_helper.to_array(initializer_scale)
                ]

                # connect the previous and successive node input and output
                for succ_node in succ_nodes:  #遍历DequantizeLinear的所有后续节点，目的是重新连接计算图（跳过假量化节点对)
                    succ_idx = get_elem_index(next_node.output[0], succ_node.input) #找到DequantizeLinear的输出（next_node.output[0]）在后续节点（succ_node）输入列表中的索引（succ_idx）
                    if succ_idx != -1:
                        succ_node.input[succ_idx] = curr_node.input[0]  #若找到索引，则将后续节点的输入从DequantizeLinear的输出，改为QuantizeLinear的输入（即原始浮点数张量），实现 “跳过” 假量化节点对的连接。
                    else:
                        raise ValueError(
                            "Remove fake-quantized node pair Error: Connection failed. No matched successive node input found for {}."
                            .format(next_node.name))

                param_name = curr_node.input[0] #跳过假量化节点对后，获取原始浮点数张量的名称。
                if self.quantization_params is None:
                    self.quantization_params = {}
                self.quantization_params[param_name] = zp_and_scale #将原始张量名称与对应的量化参数（zero_point 和 scale）存入self.quantization_params，供后续真实量化使用

                # remove fake-quantized nodes
                nodes_to_remove.extend([curr_node])
                nodes_to_remove.extend([next_node])

                # remove unused initializers in graph  #将这对节点使用的 scale 和 zero_point 初始化器加入待移除列表（因为假量化节点已移除，这些初始化器不再使用）。
                initializers_to_remove.extend([initializer_scale])
                initializers_to_remove.extend([initializer_zp])

        self.model.remove_nodes(nodes_to_remove)
        self.model.remove_initializers(initializers_to_remove)

        return self.model.model

    def find_initializer_in_path(self, initializer_name):
        if find_by_name(initializer_name, self.model.initializer()) is not None:
            return True
        if self.parent is not None:
            return self.parent.find_initializer_in_path(initializer_name)
        return False

    def should_quantize(self, node):
        if self.nodes_to_quantize is not None and len(
                self.nodes_to_quantize) != 0 and node.name not in self.nodes_to_quantize:
            return False

        if (node.op_type not in self.op_types_to_quantize):
            return False

        if self.nodes_to_exclude is not None and node.name in self.nodes_to_exclude:
            return False

        # do not quantize non-constant B matrices for matmul
        if self.q_matmul_const_b_only:
            if node.op_type == "MatMul" and (not self.find_initializer_in_path(node.input[1])):
                print("Ignore MatMul due to non constant B: {}[{}]".format(self.graph_scope, node.name))
                return False

        return True

    def add_new_nodes(self, nodes):
        self.new_nodes.extend(nodes)
        for node in nodes:
            for output_name in node.output:
                self.generated_value_names.add(output_name)


#该接口整合了假量化节点移除、子图量化、算子量化、图结构更新等关键步骤，最终生成量化后的 ONNX 模型
    def quantize_model(self):
        self.remove_fake_quantized_nodes() #移除量化感知训练中插入的QuantizeLinear/DequantizeLinear假量化节点对，清理计算图并保留量化参数（scale 和 zero_point）

        for node in self.model.nodes():
            # quantize subgraphes if have
            if self.enable_subgraph_quantization:
                node = self.quantize_node_with_sub_graph(node) #处理节点中包含的子图，返回量化后的节点（子图已被量化）

            number_of_existing_new_nodes = len(self.new_nodes) #记录当前self.new_nodes（存储量化过程中生成的新节点）的长度，用于后续跟踪当前节点量化后新增的节点。
            if self.should_quantize(node):
                op_quantizer = CreateOpQuantizer(self, node) #创建针对该算子的专用量化器（CreateOpQuantizer）
            else:
                op_quantizer = CreateDefaultOpQuantizer(self, node) #否则，创建默认量化器（CreateDefaultOpQuantizer），通常不改变节点（或仅做兼容性处理）.

            op_quantizer.quantize() #执行具体的算子量化逻辑:
                                        #对于需要量化的算子（如Conv、MatMul），会生成量化 / 反量化节点（QuantizeLinear/DequantizeLinear），并将原算子替换为低精度版本（如QLinearConv），新节点会被添加到self.new_nodes。
                                        #对于不需要量化的算子，不生成新节点或仅复制原节点到self.new_nodes。
            for i in range(number_of_existing_new_nodes, len(self.new_nodes)):  #遍历从number_of_existing_new_nodes（量化前的长度）到当前self.new_nodes长度的区间，即当前节点量化过程中生成的新节点。
                for output_name in self.new_nodes[i].output:
                    self.generated_value_names.add(output_name) #将这些新节点的输出名称添加到self.generated_value_names（标记为量化过程生成的张量），后续可避免对这些张量重复执行反量化。

        self._dequantize_outputs()

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph().ClearField('node') #清除原始模型计算图中的节点列表（node字段），为替换为量化后的新节点做准备（protobuf 字段需要显式清除后再添加新内容）。
        self.model.graph().node.extend(self.new_nodes) # 将量化过程中生成的所有新节点（self.new_nodes）添加到模型计算图中，完成计算图的更新（用量化节点替换原始节点）。

        # Remove ununsed initializers from graph, starting from the top level graph.
        if self.parent is None:  #清理未使用的初始化器（仅对顶层图执行，self.parent is None表示顶层）
            _, initializers_not_found = ONNXQuantizer.CleanGraphInitializers(self.model.graph(), self.model.model)
            if len(initializers_not_found) > 0:
                raise RuntimeError("Invalid model with unknown initializers/tensors." + str(initializers_not_found))

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    @staticmethod
    def tensor_proto_to_array(initializer):
        if initializer.data_type == onnx_proto.TensorProto.FLOAT:
            weights = onnx.numpy_helper.to_array(initializer)
        else:
            raise ValueError('Only float type quantization is supported. Weights {} is {}. '.format(
                initializer.name, type_to_name[initializer.data_type]))
        return weights

    def is_input_a_weight(self, input_name):
        initializer = find_by_name(input_name, self.model.initializer())
        return initializer is not None

    def is_per_channel(self):
        return self.per_channel

    def is_valid_quantize_weight(self, weight_name):
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is not None:
            return weight.data_type == onnx_proto.TensorProto.FLOAT
        if (not self.enable_subgraph_quantization) or (self.parent is None):
            return False
        return self.parent.is_valid_quantize_weight(weight_name)

    def _get_dynamic_input_quantization_params(self, input_name, nodes_list, qType):
        '''
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if qType == onnx_proto.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name, nodes_list)

        return self._get_dynamic_input_quantization_params_uint8(input_name, nodes_list)


"""是 ONNX Runtime 动态量化中用于生成输入张量的 INT8 动态量化参数（缩放因子 scale 和零点 zero_point）节点的核心方法，通过添加ReduceMin、ReduceMax、Abs等算子计算量化参数，
最终返回参数名称和形状。"""
    def _get_dynamic_input_quantization_params_int8(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input to int8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        #生成ReduceMin节点（计算输入张量的全局最小值）
        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name], [reduce_min_name + ":0"],
                                                reduce_min_name,
                                                keepdims=0)  # 算子类型：ReduceMin（ONNX 内置算子，用于沿所有维度求最小值）。输入：[input_name]（待量化的原始输入张量）。输出：[reduce_min_name + ":0"]
        nodes_list.append(reduce_min_node) #将生成的节点追加到nodes_list

        #生成ReduceMax节点（计算输入张量的全局最大值）
        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name], [reduce_max_name + ":0"],
                                                reduce_max_name,
                                                keepdims=0)
        nodes_list.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin) #生成Abs节点,求最小值的绝对值（abs (rmin)）
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node("Abs", [reduce_min_node.output[0]], [reduce_min_abs_name + ":0"],  #输入：reduce_min_node.output[0]（之前ReduceMin节点的输出，即输入张量的最小值）。
                                                    reduce_min_abs_name)
        nodes_list.append(reduce_min_abs_node)
        #   Find abs(rmax) #生成Abs节点,计算最大值的绝对值，abs (rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node("Abs", [reduce_max_node.output[0]], [reduce_max_abs_name + ":0"],  #输入改为reduce_max_node.output[0]（ReduceMax节点的输出，即输入张量的最大值）。
                                                    reduce_max_abs_name)
        nodes_list.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node("Max", [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],  #输入：两个Abs节点的输出（即 abs (rmin) 和 abs (rmax)）
                                             [abs_max_name + ":0"], abs_max_name)
        nodes_list.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
           #将上述 “较大值” 除以 “量化范围的一半”（即quantize_range/2.0），最终得到缩放因子（scale = max (...) / (quantize_range/2.0)）。
        """
            生成量化范围常量（用于除法计算）：
               名称：self.fixed_qrange_int8_name（之前定义的固定名称，即 “fixed_quantization_range_int8”）。
               类型：FLOAT（浮点型，确保除法精度）。
               形状：[]（标量，无维度）。
               值：get_qrange_for_qType(qType) / 2.0——get_qrange_for_qType是工具函数，返回 INT8 的量化范围（即 255，因为 INT8 范围是 - 128~127，共 256 个值，量化范围取 255），除以 2 后为 127.5。
        """
        initializer_div = onnx.helper.make_tensor(self.fixed_qrange_int8_name, onnx_proto.TensorProto.FLOAT, [],
                                                  [get_qrange_for_qType(qType) / 2.0])
        self.model.add_initializer(initializer_div)  #将该常量添加到模型（self.model.add_initializer），供后续除法节点使用

        #生成Div节点（计算缩放因子 scale）
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [abs_max_node.output[0], self.fixed_qrange_int8_name],  # 算子类型：Div（执行除法运算，输入 1 ÷ 输入 2）。  输入：abs_max_node.output[0]（max (abs (rmin), abs (rmax))）和self.fixed_qrange_int8_name（量化范围常量 127.5）。                                                                                        
                                               [input_scale_name], scale_div_name)                            # 输出：input_scale_name（最终的缩放因子张量，即 scale = max (...) / 127.5）。                                                                    
        nodes_list.append(scale_div_node)

        # Zero point  #处理零点（zero_point）；生成 INT8 对称量化的零点常量初始化器
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType, [], [0]) #INT8 对称量化的零点固定为 0（因量化范围以 0 为中心，无需偏移）
        self.model.add_initializer(initializer_zp) 

        #返回动态量化参数的关键信息：
           # input_scale_name：计算得到的缩放因子张量名称。
           # self.fixed_zero_zp_name：固定零点张量名称（即 “fixed_zero_zp”）。
           # 两个空列表：分别对应缩放因子和零点的形状（因两者均为标量，形状为空）。
        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input to uint8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        #生成ReduceMin节点（计算输入张量的全局最小值）
        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name], [reduce_min_name + ":0"],
                                                reduce_min_name,
                                                keepdims=0)
        nodes_list.append(reduce_min_node)

        #生成ReduceMax节点（计算输入张量的全局最大值）
        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name], [reduce_max_name + ":0"],
                                                reduce_max_name,
                                                keepdims=0)
        nodes_list.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        initializer_qrange = onnx.helper.make_tensor(self.fixed_qrange_uint8_name, onnx_proto.TensorProto.FLOAT, [],
                                                     [get_qrange_for_qType(qType)])
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, onnx_proto.TensorProto.FLOAT, [], [0.0])
        self.model.add_initializer(initializer_qvalue)

        # Compute Scale  计算缩放因子；
        #   Subtract rmax and rmin  #生成Sub节点（执行减法运算）
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node("Sub", [reduce_max_node.output[0], reduce_min_node.output[0]],
                                               [scale_sub_name + ":0"], scale_sub_name)
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range  #生成Div节点（计算缩放因子 scale）
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [scale_sub_node.output[0], self.fixed_qrange_uint8_name],  # scale = (abs(max) - abs(min))/quantize_range
                                               [input_scale_name], scale_div_name)
        nodes_list.append(scale_div_node)

        # Compute zero point    #开始计算零点；
        #   Subtract zero and rmin  #生成Sub节点 (计算 “零值 - 最小值”（即 0 - min）)
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node("Sub", [self.fixed_zero_name, reduce_min_node.output[0]],
                                            [zp_sub_name + ":0"], zp_sub_name)
        nodes_list.append(zp_sub_node)
        #   Divide by scale  #生成Div节点 (计算零点的中间值)
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node("Div", [zp_sub_node.output[0], input_scale_name], [zp_div_name + ":0"],  #算子类型：Div（(0 - min) ÷ scale） ， 输入：zp_sub_node.output[0]（0 - min）、input_scale_name（之前计算的缩放因子）。
                                            zp_div_name)  #输出：zp_div_name + ":0"（零点的浮点中间值）。
        nodes_list.append(zp_div_node)
        #   Compute floor    #生成Floor节点 (对零点中间值取整)
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output, [zp_floor_name + ":0"], zp_floor_name)  #算子类型：Floor（向下取整，确保零点为整数，符合 UINT8 量化要求）。输入：zp_div_node.output（零点的浮点中间值）。
        nodes_list.append(zp_floor_node)
        #   Cast to integer  #生成Cast节点（将零点转换为 UINT8 类型）
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output, [input_zp_name], zp_cast_name, to=qType) #算子类型：Cast（类型转换); 输入：zp_floor_node.output（取整后的零点值）。输出：input_zp_name（最终的 UINT8 类型零点张量）。参数：to=qType（目标类型为 UINT8）。
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def _get_quantization_params(self, param_name, use_scale=None, use_zeropoint=None):
        '''
        Create initializers and inputs in the graph for zero point and scale of output.
        Zero point and scale values are obtained from self.quantization_params if specified.
            parameter param_name: Name of the quantization parameter.
            return: result, scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if use_scale is None or use_zeropoint is None:
            if self.quantization_params is None or param_name not in self.quantization_params:
                logging.info("Quantization parameters for tensor:\"{}\" not specified".format(param_name))
                return False, "", "", "", ""

            params = self.quantization_params[param_name]
            if params is None or len(params) != 2:
                raise ValueError("Quantization parameters should contain zero point and scale. "
                                 "Specified values for output {}: {}".format(param_name, params))

            zero_point_values = [params[0]]
            scale_values = [params[1]]
        else:
            zero_point_values = [use_zeropoint]
            scale_values = [use_scale]

        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = self.input_qType
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_point_shape, zero_point_values)
        self.model.add_initializer(init_zp)
        init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, scale_values)
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantize_input_nodes(self, node, input_index, qType, given_scale_name=None, given_zp_name=None):
        '''
        Given an input for a node (which is not a initializer), this function

        - add nodes to compute zero point and scale for this input if they don't exist.
        - add new QuantizeLinear node to quantize the input.

        :param node: node being quantized in NodeProto format.
        :param input_index: index of input in node.input.
        :param qType: type to quantize to.
        :param given_scale_name: if those inputs need to be quanitzed using this scale tensor.
        :param given_zp_name: if those inputs to be quantized using this zeropoint tensor.
        :return: List of newly created nodes in NodeProto format.
        '''
        input_name = node.input[input_index]
        output_name = input_name + "_quantized"
        ql_node_name = input_name + "_QuantizeLinear"

        if (given_scale_name is not None) and (given_zp_name is not None):
            data_found, scale_name, zp_name = (True, given_scale_name, given_zp_name)
        else:
            data_found, scale_name, zp_name, _, _ = self._get_quantization_params(input_name)

        nodes = []
        if data_found == True:
            qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name],
                                                 [output_name], ql_node_name)
        else:
            if self.static:
                return None
            # dynamic mode
            # Scale and Zero Points not available for this input. Add nodes to dynamically compute it
            if self.fuse_dynamic_quant and qType == onnx_proto.TensorProto.UINT8:
                scale_name = input_name + "_scale"
                zp_name = input_name + "_zero_point"
                qlinear_node = onnx.helper.make_node("DynamicQuantizeLinear", [input_name],
                                                     [output_name, scale_name, zp_name], ql_node_name)
            else:
                scale_name, zp_name, scale_shape, zp_shape = \
                    self._get_dynamic_input_quantization_params(input_name, nodes, qType)
                qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name],
                                                     [output_name], ql_node_name)

        self.quantized_value_map[input_name] = QuantizedValue(input_name, output_name, scale_name, zp_name, qType)
        return nodes + [qlinear_node]

    def find_quantized_value(self, input_name):
        if input_name in self.quantized_value_map:
            return self.quantized_value_map[input_name]
        if self.parent is not None:
            return self.parent.find_quantized_value(input_name)
        return None

"""
该方法，用于对模型中的偏置（bias）进行静态量化处理。
根据文档字符串说明，偏置量化的规则是：零点（Zero Point）固定为 0，缩放因子（Scale）等于输入的缩放因子乘以权重的缩放因子
"""
    def quantize_bias_static(self, bias_name, input_name, weight_name):
        '''
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        '''

        # Handle case where bias already in quantizatio map
        if bias_name in self.quantized_value_map:
            return self.quantized_value_map[bias_name].q_name

        # get scale for weight  #以下代码用于获取权重的缩放因子（scale）
        weight_scale_name = self.quantized_value_map[weight_name].scale_name  #从量化值映射表中，通过权重名称（weight_name）获取权重缩放因子的名称（scale_name）。
        weight_initializer = find_by_name(weight_scale_name, self.model.initializer()) #从模型的初始化器（initializer，存储模型中固定值的张量）中，找到名称为 weight_scale_name 的初始化器（即权重缩放因子的张量）。
        weight_scale = self.tensor_proto_to_array(weight_initializer) #将 ONNX 的张量协议格式（TensorProto）转换为 numpy 数组，得到权重的缩放因子数值（weight_scale）

        # get bias
        bias_initializer = find_by_name(bias_name, self.model.initializer())  #找偏置张量
        bias_data = self.tensor_proto_to_array(bias_initializer)
        quantized_bias_name = bias_name + "_quantized"

        # get scale for input
        if input_name in self.quantized_value_map:  #检查输入名称是否在量化值映射表中（即输入是否已被量化）
            input_scale_name = self.quantized_value_map[input_name].scale_name  #若输入已量化，则从映射表中直接获取输入缩放因子的名称
        elif input_name in self.quantization_params:
            _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
        else:
            raise ValueError("Expected {} to be in quantized value map for static quantization".format(input_name))

        inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())  #找到输入缩放因子对应的初始化器（即输入缩放因子的张量）
        input_scale = self.tensor_proto_to_array(inputscale_initializer)

        # calcuate scale for bias
        bias_scale = input_scale * weight_scale

        # quantize bias
        quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)  #将偏置数据（bias_data）转换为 numpy 数组；
                                                                                        #除以偏置缩放因子（bias_scale），得到浮点数形式的量化前中间值
                                                                                        #（round()）后转换为 32 位整数（np.int32）

        # update bias initializer  #更新模型中偏置的初始化器（添加量化后的偏置）
        bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
        packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
        self.model.initializer().extend([packed_bias_initializer])  #将量化后的偏置初始化器添加到模型的初始化器列表中（更新模型）

        # update scale initializer #添加量化偏置的缩放因子
        quantized_bias_scale_name = quantized_bias_name + "_scale"
        bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
        packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        self.model.initializer().extend([packed_bias_scale_initializer]) #将偏置缩放因子的初始化器添加到模型的初始化器列表中。

        # update zero initializer  #添加量化偏置的零点，固定为 0
        quantized_bias_zp_name = quantized_bias_name + "_zero_point"
        bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
        packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        self.model.initializer().extend([packed_bias_zp_initializer]) #将偏置零点的初始化器添加到模型的初始化器列表中。

        assert (bias_name not in self.quantized_value_map)
        quantized_value = QuantizedValue(bias_name, quantized_bias_name, quantized_bias_scale_name,
                                         quantized_bias_zp_name, QuantizedValueType.Initializer,
                                         0 if bias_scale_data.size > 1 else None)
        self.quantized_value_map[bias_name] = quantized_value #将创建的 QuantizedValue 对象存入量化值映射表，键为原始偏置名称（bias_name），表示该偏置已被量化。

        return quantized_bias_name  #返回量化后偏置的名称

    def contains_tensor(self, tensor_name):
        '''
        only check for value info and newly generated tensor names, initializers are checked seperately
        '''
        return (tensor_name in self.value_infos) or (tensor_name in self.tensor_names) or (tensor_name in self.generated_value_names)


    #用于量化 ONNX 模型中节点（node）的输入
    """
    node：待量化输入的节点（ONNX 的NodeProto格式）。
    indices：需要量化的输入索引列表（指定节点的哪些输入需要量化）。
    initializer_use_weight_qType：布尔值，若为True，初始化器（如权重）使用权重量化类型（weight_qType），否则使用输入量化类型（input_qType）。
    reduce_range：是否缩减量化范围（如将 8 位量化缩减为 7 位，减少精度损失）。
    op_level_per_channel：是否在操作级支持按通道量化（针对卷积等按通道处理的权重）。
    axis：按通道量化时的通道轴（默认-1，即最后一维）。
    from_subgraph：是否来自子图（影响节点添加方式）。
    """
    def quantize_inputs(self, node, indices, initializer_use_weight_qType=True, reduce_range=False, op_level_per_channel=False, axis=-1, from_subgraph=False):
        '''
        Given a node, this function quantizes the inputs as follows:
            - If input is an initializer, quantize the initializer data, replace old initializer
              with new initializer
            - Else, add QuantizeLinear nodes to perform quantization
            parameter node: node being quantized in NodeProto format.
            parameter indices: input indices to quantize.
            return: (List of quantized input names,
                     List of zero point names used for input quantization,
                     List of scale names used for input quantization,
                     List of new QuantizeLinear nodes created)
        '''

        scale_names = []  #用于存储输入量化的缩放因子（scale）名称。
        zero_point_names = [] #用于存储输入量化的零点（zero point）名称。
        quantized_input_names = [] #用于存储量化后的输入名称。
        nodes = [] #用于存储新创建的QuantizeLinear节点。

        for input_index in indices:
            node_input = node.input[input_index]  #获取当前索引对应的输入名称（node.input是节点的输入列表）

            # Find if this input is already quantized
            if node_input in self.quantized_value_map:
                quantized_value = self.quantized_value_map[node_input]
                scale_names.append(quantized_value.scale_name)
                zero_point_names.append(quantized_value.zp_name)
                quantized_input_names.append(quantized_value.q_name)
                continue

            # Quantize the input  #当前输入未量化，开始执行量化。
            initializer = find_by_name(node_input, self.model.initializer())  #从模型的初始化器（initializer，如权重等固定值张量）中查找该输入。
            if initializer is not None:
                if self.per_channel and op_level_per_channel:  #若类配置了按通道量化（self.per_channel）且当前操作支持按通道量化（op_level_per_channel），则按通道量化。
                    q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                        initializer.name, self.weight_qType if initializer_use_weight_qType else self.input_qType,
                        axis, reduce_range)
                else:
                    q_weight_name, zp_name, scale_name = self.quantize_weight(
                        initializer, self.weight_qType if initializer_use_weight_qType else self.input_qType,
                        reduce_range)

                quantized_input_names.append(q_weight_name) #将量化后的初始化器名称添加到quantized_input_names
                zero_point_names.append(zp_name)
                scale_names.append(scale_name)
            elif self.contains_tensor(node_input):  #若输入不是初始化器，但模型包含该张量（self.contains_tensor检查张量是否存在），则需要添加QuantizeLinear节点量化该张量。
                # Add QuantizeLinear node.
                qlinear_node = self.model.find_node_by_name(node_input + "_QuantizeLinear", self.new_nodes,
                                                            self.model.graph())  #检查是否已存在针对该输入的QuantizeLinear节点（名称格式为 “输入名 +_QuantizeLinear”）。
                if qlinear_node is None:
                    quantize_input_nodes = self._get_quantize_input_nodes(node, input_index, self.input_qType) #调用_get_quantize_input_nodes方法生成量化输入所需的节点（通常包含QuantizeLinear及相关节点）。
                    if quantize_input_nodes is None:
                        return (None, None, None, None)
                    if from_subgraph:
                        self.add_new_nodes(quantize_input_nodes)
                    else:
                        nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]  #新生成的节点列表中，最后一个节点即为QuantizeLinear节点。

                if qlinear_node.op_type == "QuantizeLinear":  #若节点类型是QuantizeLinear（ONNX 标准量化节点）。
                    quantized_input_names.extend(qlinear_node.output) #将QuantizeLinear的输出（量化后的值）添加到quantized_input_names。
                    scale_names.append(qlinear_node.input[1]) #QuantizeLinear的第 2 个输入（input[1]）是缩放因子，其名称添加到scale_names。
                    zero_point_names.append(qlinear_node.input[2]) #QuantizeLinear的第 3 个输入（input[2]）是零点，其名称添加到zero_point_names。
                else:
                    quantized_input_names.append(qlinear_node.output[0])
                    scale_names.append(qlinear_node.output[1])
                    zero_point_names.append(qlinear_node.output[2])
            elif self.parent is not None:  #若输入既不是初始化器，也不是模型中的张量，但存在父节点（self.parent），则委托父节点处理量化。
                (parent_quantized_input_names, parent_zero_point_names, parent_scale_names, _) = self.parent.quantize_inputs(
                    node,
                    [input_index],
                    initializer_use_weight_qType=initializer_use_weight_qType,
                    reduce_range=reduce_range,
                    op_level_per_channel=op_level_per_channel,
                    axis=axis,
                    from_subgraph=True)
                quantized_input_names.append(parent_quantized_input_names[0])
                scale_names.append(parent_scale_names[0])
                zero_point_names.append(parent_zero_point_names[0])
                # node should not be add this child level here
            else:
                raise ValueError('Invalid tensor name to quantize: {} @graph scope{}'.format(node_input, self.graph_scope))

        return (quantized_input_names, zero_point_names, scale_names, nodes) #返回四个列表：量化后的输入名称、零点名称、缩放因子名称、新创建的量化节点。


"""用于对模型中的权重（weight）进行量化处理。它会生成量化后的权重、缩放因子（scale）和零点（zero point），并更新模型初始化器和量化映射表
    weight：待量化的权重，以 ONNX 的TensorProto格式表示。
    qType：量化后的数据类型（如UINT8、INT8等）。
    reduce_range：是否缩减量化范围（如 8 位量化缩减为 7 位，减少精度损失）。
    keep_float_weight：布尔值，若为True则仅生成缩放因子和零点，不量化权重本身；若为False则同时量化权重。"""
    def quantize_weight(self, weight, qType, reduce_range=False, keep_float_weight=False):
        '''
            :param weight: TensorProto initializer
            :param qType: type to quantize to
            :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                      If keep_float_weight is False, quantize the weight, or don't quantize the weight.
            :return: quantized weight name, zero point name, scale name
        '''
        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (quantized_value.q_name, quantized_value.zp_name, quantized_value.scale_name)

        q_weight_name = weight.name + "_quantized"
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = self.tensor_proto_to_array(weight)
        #输入：展平的权重数据（flatten().tolist()）、量化类型（qType）、是否对称量化（self.is_weight_symmetric）、是否缩减范围（self.reduce_range与reduce_range的逻辑与）。
        _, _, zero_point, scale, q_weight_data = quantize_data(weight_data.flatten().tolist(),
                                                               qType, self.is_weight_symmetric,
                                                               self.reduce_range and reduce_range)
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, [], [scale]) #创建缩放因子的初始化器
        zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], [zero_point])  #创建零点的初始化器
        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:  #量化权重数据并更新模型
            q_weight_data = np.asarray(q_weight_data,
                                       dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[qType]).reshape(weight.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)
            self.model.initializer().extend([q_weight_initializer])

        # Log entry for this quantized weight #记录该权重的量化信息到映射表
        quantized_value = QuantizedValue(weight.name, q_weight_name, scale_name, zp_name,
                                         QuantizedValueType.Initializer, None)
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name


"""用于对权重进行按通道量化（per-channel quantization）,对每个通道的权重单独计算缩放因子和零点，再将所有通道的量化数据组合成完整权重张量。
与普通量化（对整个权重张量用同一套缩放因子和零点）不同，按通道量化为每个通道单独计算量化参数，能更好地适应不同通道的数值分布，常用于卷积层等按通道处理的权重
    weight_name：待量化的权重名称。
    weight_qType：量化后的数据类型（如INT8、UINT8等）。
    channel_axis：通道所在的轴（如卷积层权重通常为[out_channels, in_channels, kH, kW]，channel_axis=0表示按输出通道量化）。
    reduce_range：是否缩减量化范围（如 8 位量化用 7 位表示，减少精度损失）。
    keep_float_weight：是否保留浮点权重（False则替换为量化权重，True仅生成量化参数）。
"""
    def quantize_weight_per_channel(self, weight_name, weight_qType, channel_axis, reduce_range=True,
                                    keep_float_weight=False):
        # Find if this input is already quantized
        if weight_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight_name]
            return (quantized_value.q_name, quantized_value.zp_name, quantized_value.scale_name)

        initializer = find_by_name(weight_name, self.model.initializer())  #weight_name的权重初始化器（TensorProto格式）
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = self.tensor_proto_to_array(initializer) #将 ONNX 的TensorProto格式权重转换为 numpy 数组，得到原始权重数据（weights）
        channel_count = weights.shape[channel_axis] #获取通道数量：权重数组在channel_axis维度上的大小（如weights.shape=(32, 16, 3, 3)
        rmin_list = []
        rmax_list = []
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = [] #存储每个通道的量化后数据
        for i in range(channel_count):
            per_channel_data = weights.take(i, channel_axis)  #提取第i个通道的权重数据：通过take(i, channel_axis)从weights中沿channel_axis轴取出索引为i的子数组（单个通道的权重）。
            rmin, rmax, zero_point, scale, quantized_per_channel_data = quantize_data(
                per_channel_data.flatten().tolist(), weight_qType,
                self.is_weight_symmetric or weight_qType == onnx_proto.TensorProto.INT8, self.reduce_range and reduce_range)
            rmin_list.append(rmin)
            rmax_list.append(rmax)
            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)

        # combine per_channel_data into one  #将所有通道的量化数据组合成一个完整的权重数组（恢复原始形状）
        reshape_dims = list(weights.shape)  # deep copy  #复制原始权重的形状（如(32, 16, 3, 3)），用于重塑单个通道的数据。
        reshape_dims[channel_axis] = 1  # only one per channel for reshape #将形状中通道轴的维度改为 1（如channel_axis=0时，reshape_dims变为(1, 16, 3, 3)），以便单个通道数据能拼接成原始形状。
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims) #初始化总量化权重数组：将第一个通道的量化数据转换为 numpy 数组，重塑为reshape_dims（单通道形状）。
        for i in range(1, len(quantized_per_channel_data_list)): #遍历剩余通道（从第 1 个开始），将其量化数据拼接到总数组中。
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims) #将第i个通道的量化数据转换为 numpy 数组，重塑为reshape_dims（单通道形状）。
            quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis) #沿channel_axis轴拼接当前通道数据与总数组，逐步恢复原始权重的形状（如 32 个通道拼接后恢复为(32, 16, 3, 3)）。

        q_weight_name = weight_name + "_quantized"
        zp_name = weight_name + "_zero_point"
        scale_name = weight_name + "_scale"

        quantized_value = QuantizedValue(weight_name, q_weight_name, scale_name, zp_name,
                                         QuantizedValueType.Initializer, None) #创建QuantizedValue对象（量化信息封装）
        self.quantized_value_map[weight_name] = quantized_value

        # Update packed weight, zero point, and scale initializers  #更新模型初始化器，添加量化后的权重、零点和缩放因子
        zero_scale_shape = [initializer.dims[channel_axis]] #定义零点和缩放因子的形状：长度为通道数的一维数组（如 32 个通道则形状为[32]，与普通量化的标量形状不同）。
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape,
                                                    scale_list)
        zero_initializer = onnx.helper.make_tensor(zp_name, weight_qType, zero_scale_shape, zero_point_list)

        self.model.initializer().extend([scale_initializer, zero_initializer])  #将缩放因子和零点的初始化器添加到模型的初始化器列表中（更新模型）

        if not keep_float_weight:
            quantized_weights = np.asarray(
                quantized_weights, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight_qType]).reshape(initializer.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
            self.model.initializer().extend([q_weight_initializer])

        return (q_weight_name, zp_name, scale_name)  #返回量化后的权重名称、零点名称和缩放因子名称

    def _dequantize_value(self, value_name):
        '''
        Given a value (input/output) which is quantized, add a DequantizeLinear node to dequantize
        it back to float32
            parameter value_name: value to dequantize
            parameter new_nodes_list: List of new nodes created before processing current node
            return: None if there is already a DequantizeLinear node that dequantizes it
                    A DequantizeLinear node otherwise
        '''
        if (value_name in self.quantized_value_map) and (value_name not in self.generated_value_names):
            quantized_value = self.quantized_value_map[value_name]
            # Add DequantizeLinear Node for this input
            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_node = self.model.find_node_by_name(dqlinear_name, self.new_nodes, self.model.graph())
            if dqlinear_node is None:
                dqlinear_inputs = [quantized_value.q_name, quantized_value.scale_name, quantized_value.zp_name]
                dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [value_name],
                                                        dqlinear_name)
                return dequantize_node
            else:
                # DQ op is already present, assert it's output matches the input of current node
                assert (value_name == dqlinear_node.output[0])
        return None

    def _dequantize_outputs(self):
        '''
        Dequantize output if it is quantized
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        '''

        for output in self.model.graph().output:
            dequantize_node = self._dequantize_value(output.name)
            if dequantize_node is not None:
                self.new_nodes.append(dequantize_node)

"""核心功能是计算模型中张量的量化参数（缩放因子 scale 和零点 zero point）。它会先根据特定节点（Clip、Relu）的输出范围调整输入范围，
再基于张量的数值范围和量化类型，通过公式计算每个张量的量化参数。"""
    def calculate_quantization_params(self):
        if self.tensors_range is None:  #检查张量范围字典（self.tensors_range）是否为空：该字典存储了每个张量的数值范围（键为张量名称，值为 (rmin, rmax)，即张量的最小值和最大值），是计算量化参数的前提。
            return

        # adjust tensor_ranges for input of Clip and Relu node
        # 针对 Clip 和 Relu 节点的输入张量，调整其数值范围 —— 因为这两类节点会改变张量的数值范围（如 Relu 会将负数截断为 0），需用节点输出的实际范围反向修正输入范围，确保量化参数准确。
        for node in self.model.nodes():  #遍历模型中的所有节点（self.model.nodes() 返回模型的节点列表，每个节点为 ONNX 的 NodeProto 格式）。
            if node.op_type not in ['Clip', 'Relu']:  #仅处理 Clip（数值裁剪）和 Relu节点.
                continue
            if not self.should_quantize(node):
                continue
            if len(self.model.input_name_to_nodes()[node.input[0]]) != 1:  #若输入张量被多个节点使用（列表长度 >1），则不能仅用当前节点的输出范围修正其范围（避免影响其他节点），因此跳过。
                continue
            if node.input[0] not in self.tensors_range.keys() or node.output[0] not in self.tensors_range.keys(): #检查输入张量（node.input[0]）和输出张量（node.output[0]）的范围是否都在 self.tensors_range 中 —— 只有两者都有范围数据，才能进行修正。
                continue
            """核心修正逻辑：将 Clip/Relu 节点输出张量的范围赋值给输入张量的范围。
                   例：Relu 节点输入范围为 (-5, 10)，输出范围为 (0, 10)（负数被截断），则修正输入范围为 (0, 10)，确保后续计算输入量化参数时，基于实际有效的数值范围（而非原始范围）。
            """
            self.tensors_range[node.input[0]] = self.tensors_range[node.output[0]]

        quantization_params = {} #初始化量化参数字典，用于存储每个张量的量化参数
        for tensor_name in self.tensors_range.keys():  #遍历 self.tensors_range 中的所有张量名称，逐个计算量化参数
            rmin, rmax = self.tensors_range[tensor_name]
            qmin, qmax = get_qmin_qmax_for_qType(self.input_qType) #根据量化类型（self.input_qType，如 UINT8、INT8）获取该类型的量化最小值（qmin）和最大值（qmax）：
                                                                       #例：UINT8 的 qmin=0，qmax=255；INT8 的 qmin=-128，qmax=127。

            quantization_params[tensor_name] = compute_scale_zp(rmin, rmax,
                                                                qmin, qmax,
                                                                self.is_activation_symmetric)

        return quantization_params


"""
这段代码是 ONNXQuantizer 类的静态方法 CleanGraphInitializers，核心功能是清理 ONNX 图（及子图）中未使用的初始化器（如量化模型后遗留的无用权重、缩放因子 / 零点等），
同时处理子图的递归清理，并返回清理后的图和 “找不到的张量名列表”。
"""
    # static method
    def CleanGraphInitializers(graph, model):
        '''
        Clean unused initializers including which is caused by quantizing the model.
            return cleaned graph, and list of tensor names from this graph and all its subgraphes
            that can not be found in this graph and its subgraphes
        '''
        requesting_tensor_names = {}
        requesting_tensor_names.update({input_name: 1 for node in graph.node for input_name in node.input if input_name})
        requesting_tensor_names.update({g_out.name: 1 for g_out in graph.output if g_out.name})

        new_nodes = []
        for node in graph.node:
            node_2_add = node
            graph_attrs = [attr for attr in node.attribute if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS]
            if len(graph_attrs) > 0:
                kwargs = {}
                for attr in node.attribute:
                    kv = {}
                    if attr.type == onnx.AttributeProto.GRAPH:
                        cleaned_sub_graph, sub_requesting_tensor_names = ONNXQuantizer.CleanGraphInitializers(attr.g, model)
                        kv = {attr.name: cleaned_sub_graph}
                        requesting_tensor_names.update({gn: 1 for gn in sub_requesting_tensor_names})
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        cleaned_graphes = []
                        for subgraph in attr.graphs:
                            cleaned_sub_graph, sub_requesting_tensor_names = ONNXQuantizer.CleanGraphInitializers(subgraph, model)
                            cleaned_graphes.extend([cleaned_sub_graph])
                            requesting_tensor_names.update({gn: 1 for gn in sub_requesting_tensor_names})
                        kv = {attr.name: cleaned_graphes}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node_2_add = onnx.helper.make_node(node.op_type, node.input, node.output, name=node.name, **kwargs)
            new_nodes.extend([node_2_add])

        graph.ClearField('node')
        graph.node.extend(new_nodes)

        generated_names = {}
        generated_names.update({output_name: 1 for node in graph.node for output_name in node.output if output_name})
        for gn in generated_names:
            requesting_tensor_names.pop(gn, None)

        name_to_input = {}
        for input in graph.input:
            name_to_input[input.name] = input

        unused_ini_tensors = []
        for ini_tensor in graph.initializer:
            if ini_tensor.name in requesting_tensor_names:
                requesting_tensor_names.pop(ini_tensor.name, None)
            else:
                # mark it to remove, remove here directly will cause mis-behavier
                unused_ini_tensors.append(ini_tensor)

        for ini_tensor in unused_ini_tensors:
            graph.initializer.remove(ini_tensor)
            if ini_tensor.name in name_to_input:
                try:
                    graph.input.remove(name_to_input[ini_tensor.name])
                except StopIteration:
                    if model.ir_version < 4:
                        print("Warning: invalid weight name {} found in the graph (not a graph input)".format(ini_tensor.name))

        for input in graph.input:
            if input.name in requesting_tensor_names:
                requesting_tensor_names.pop(input.name, None)

        return graph, requesting_tensor_names
