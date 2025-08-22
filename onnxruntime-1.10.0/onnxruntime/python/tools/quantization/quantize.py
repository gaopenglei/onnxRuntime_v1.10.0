# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import onnx
import onnx.numpy_helper
import struct
import logging
import numpy as np

from pathlib import Path

from onnx import onnx_pb as onnx_proto
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

from .quant_utils import QuantizationMode, QuantizedValueType, QuantizedInitializer, QuantizedValue
from .quant_utils import find_by_name, get_elem_index, get_mul_node, generate_identified_filename, attribute_to_kwarg
from .quant_utils import QuantType, QuantFormat

from .registry import QLinearOpsRegistry, IntegerOpsRegistry

from .onnx_model import ONNXModel
from .onnx_quantizer import ONNXQuantizer
from .qdq_quantizer import QDQQuantizer
from .calibrate import CalibrationDataReader, create_calibrator, CalibrationMethod 


def optimize_model(model_path: Path):
    '''
        Generate model that applies graph optimization (constant folding,etc.) （生成应用了图优化（如常量折叠等）的模型）
        parameter model_path: path to the original onnx model
        return: optimized onnx model
    '''
    opt_model_path = generate_identified_filename(model_path, "-opt")
    sess_option = SessionOptions()  #用于配置 ONNX Runtime 推理会话的参数。SessionOptions 是 ONNX Runtime 中用于设置会话选项的类，可配置优化级别、日志、线程数等.
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC #启用基础优化（如常量折叠、冗余节点删除等）
    _ = InferenceSession(model_path.as_posix(), sess_option, providers=['CPUExecutionProvider']) #内部执行优化过程, 自动将优化后的模型保存到 opt_model_path
    optimized_model = onnx.load(opt_model_path.as_posix()) #使用 ONNX 库的 load 函数，从优化后模型的路径加载模型，得到一个 ONNX 模型对象（onnx.ModelProto）
    return optimized_model


def load_model(model_path: Path, optimize=True):
    if optimize:
        #optimize the original model
        onnx_model = ONNXModel(optimize_model(Path(model_path)))
        # to support GEMM
        onnx_model.replace_gemm_with_matmul()
        return onnx_model.model

    return onnx.load(Path(model_path))


def quantize(model,
             per_channel=False,
             nbits=8,
             quantization_mode=QuantizationMode.IntegerOps,
             static=False,
             force_fusions=False,
             symmetric_activation=False,
             symmetric_weight=False,
             quantization_params=None,
             nodes_to_quantize=None,
             nodes_to_exclude=None,
             op_types_to_quantize=[]):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file
    :param model: ModelProto to quantize
    :param per_channel: quantize weights per channel
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param quantization_mode: Can be one of the QuantizationMode types.
        IntegerOps:
            the function will use integer ops. Only ConvInteger and MatMulInteger ops are supported now.
        QLinearOps:
            the function will use QLinear ops. Only QLinearConv and QLinearMatMul ops are supported now.
    :param static:
        True: The inputs/activations are quantized using static scale and zero point values
              specified through quantization_params.
        False: The inputs/activations are quantized using dynamic scale and zero point values
               computed while running the model.
    :param symmetric_activation:
        True: activations are quantized into signed integers.
        False: activations are quantized into unsigned integers.
    :param symmetric_weight:
        True: weights are quantized into signed integers.
        False: weights are quantized into unsigned integers.
    :param quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Should be specified when static is set to True.
        The quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
    :return: ModelProto with quantization
    '''
    logging.warning("onnxruntime.quantization.quantize is deprecated.\n\
         Please use quantize_static for static quantization, quantize_dynamic for dynamic quantization.")
    if nbits == 8 or nbits == 7:
        mode = quantization_mode
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(model)

        if not op_types_to_quantize or len(op_types_to_quantize) == 0:
            op_types_to_quantize = list(QLinearOpsRegistry.keys()) if static else list(IntegerOpsRegistry.keys())

        quantizer = ONNXQuantizer(copy_model, per_channel, nbits == 7, mode, static, symmetric_weight,
                                  symmetric_activation, quantization_params, nodes_to_quantize, nodes_to_exclude,
                                  op_types_to_quantize)

        quantizer.quantize_model()
        return quantizer.model.model
    else:
        raise ValueError('Only 8 and 7 bit quantization is currently supported')


#是 ONNX Runtime 中实现静态量化（Post-Training Quantization, PTQ） 的核心函数 quantize_static，基于校准数据将 FP32 模型量化为低精度（如 INT8）模型。
# 核心是通过 “校准数据收集→范围计算→量化参数生成→模型结构转换” 的流程，在几乎不损失精度的前提下，将 FP32 模型转换为低精度模型。
"""
参数涵盖量化所需的核心配置：
输入输出路径（model_input/model_output）、校准数据读取器（calibration_data_reader）；
量化格式（quant_format）、目标算子类型（op_types_to_quantize）；
量化参数（per_channel 按通道量化、reduce_range 缩减量化范围等）；
校准方法（calibrate_method）及其他扩展选项（extra_options）。
"""
def quantize_static(model_input,
                    model_output,
                    calibration_data_reader: CalibrationDataReader,
                    quant_format=QuantFormat.QOperator,
                    op_types_to_quantize=[],
                    per_channel=False,
                    reduce_range=False,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QUInt8,
                    nodes_to_quantize=[],
                    nodes_to_exclude=[],
                    optimize_model=True,
                    use_external_data_format=False,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options = {}):

    '''
        Given an onnx model and calibration data reader, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param calibration_data_reader: a calibration data reader. It enumerates calibration data and generates inputs for the original model.
    :param quant_format: QuantFormat{QOperator, QDQ}.
        QOperator format quantizes the model with quantized operators directly.
        QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
    :param op_types: operators to quantize
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param activation_type: quantization data type of activation
    :param weight_type: quantization data type of weight
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    :param optimize_model: optimize model before quantization.
    :param use_external_data_format: option used for large size (>2GB) model. Set to False by default. 
    :param calibrate_method: 
        Current calibration methods supported are MinMax and Entropy. 
        Please use CalibrationMethod.MinMax or CalibrationMethod.Entropy as options.
    :param extra_options:
        key value pair dictionary for various options in different case. Current used:
            extra.Sigmoid.nnapi = True/False  (Default is False)
            ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
            WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
            EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                          Dyanmic mode currently is supported. Will support more in future.
            DisableShapeInference = True/False : in dynamic quantize mode, shape inference is not must have
                                                 and if it cause some issue, you could disable it.
            ForceQuantizeNoInputCheck = True/False : By default, some latent operators like maxpool, transpose, do not quantize
                                                     if their input is not quantized already. Setting to True to force such operator
                                                     always quantize input and so generate quantized output. Also the True behavior
                                                     could be disabled per node using the nodes_to_exclude.
            MatMulConstBOnly = True/False: Default is False. If enabled, only MatMul with const B will be quantized.
            AddQDQPairToWeight = True/False : Default is False which quantizes floating-point weight and feeds it to 
                                              soley inserted DeQuantizeLinear node. If True, it remains floating-point weight and 
                                              inserts both QuantizeLinear/DeQuantizeLinear nodes to weight.
            OpTypesToExcludeOutputQuantizatioin = list of op type : Default is []. If any op type is specified, it won't quantize  
                                                                    the output of ops with this specific op types.
            DedicatedQDQPair = True/False : Default is False. When inserting QDQ pair, multiple nodes can share a single QDQ pair as their inputs.
                                            If True, it will create identical and dedicated QDQ pair for each node. 
            QDQOpTypePerChannelSupportToAxis = dictionary : Default is {}. Set channel axis for specific op type, for example: {'MatMul': 1},
                                                            and it's effective only when per channel quantization is supported and per_channel is True.
                                                            If specific op type supports per channel quantization but not explicitly specified with channel axis,
                                                            default channel axis will be used.
    '''

    mode = QuantizationMode.QLinearOps  # 设置量化模式为 QLinearOps（线性量化模式），这是静态量化的标准模式，对应 ONNX 规范中的量化算子体系。

"""
1.如果未指定 op_types_to_quantize（需要量化的算子类型），则默认使用 QLinearOpsRegistry 中注册的所有支持量化的算子类型（如 Conv、MatMul 等）。
2.QLinearOpsRegistry 是 ONNX Runtime 维护的 “支持量化的算子注册表”，确保只对兼容量化的算子进行处理。
"""
    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

"""调用 load_model 函数加载原始 ONNX 模型：
  1.第一个参数：将输入路径 model_input 转换为 Path 对象；
  2.第二个参数 optimize_model：若为 True，则在加载时先对模型进行图优化（如常量折叠、算子融合），为量化做准备。
返回加载后的模型对象（onnx.ModelProto）。"""
    model = load_model(Path(model_input), optimize_model)

"""调用 create_calibrator 创建校准器（Calibrator 实例），用于计算量化所需的激活值范围（scale 和 zero point）：
参数包括：原始模型、待量化算子类型、校准方法（MinMax 或 Entropy）"""
    calibrator = create_calibrator(model, op_types_to_quantize, calibrate_method=calibrate_method)

 """调用校准器的 collect_data 方法，通过 calibration_data_reader（校准数据读取器）加载校准数据，执行模型推理并收集关键节点的激活值数据（用于后续计算量化范围）。
(校准数据通常是少量代表性的真实输入（如测试集的子集），需覆盖模型常见的输入分布。) """
    calibrator.collect_data(calibration_data_reader)

"""调用校准器的 compute_range 方法，基于收集的激活值数据计算每个张量的量化范围（最大值 / 最小值）：
对于 MinMax 校准：取激活值的最大 / 最小值作为范围；
对于 Entropy 校准：通过信息熵最小化确定最优范围（更适合减少精度损失）。
返回的 tensors_range 是一个字典，记录每个张量的量化范围，将用于后续量化参数（scale/zero point）的计算。"""
    tensors_range = calibrator.compute_range()

"""根据 quant_format（量化格式）选择不同的量化器：
       QuantFormat.QOperator：使用 ONNXQuantizer，直接将原始算子替换为量化算子（如 QLinearConv 替换 Conv），生成 “量化算子直接调用” 的模型；
       QuantFormat.QDQ：使用 QDQQuantizer，在原始算子的输入 / 输出处插入 QuantizeLinear（量化）和 DequantizeLinear（反量化）节点，生成 “量化 - 反量化对” 模式的模型（更易兼容不同推理引擎）。
   量化器初始化参数包含：原始模型、量化配置（per_channel/reduce_range）、量化类型（weight_type/activation_type）、量化范围（tensors_range）、待量化 / 排除的节点列表等，控制量化过程的细节。"""
    if quant_format is QuantFormat.QOperator:
        quantizer = ONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options)
    else:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options)

    quantizer.quantize_model()  #执行模型量化
    quantizer.model.save_model_to_file(model_output, use_external_data_format)  #将量化后的模型（quantizer.model）保存到 model_output 路径,
                                                                                #use_external_data_format：若为 True，则将大权重存储为外部文件（适用于模型大小超过 2GB 的场景），否则嵌入到 ONNX 文件中。


#这段代码是 ONNX Runtime v1.10.0 中实现动态量化（Dynamic Quantization） 的核心函数 quantize_dynamic，
#无需校准数据即可将 FP32 模型量化为低精度模型（如 INT8），核心是在推理时动态计算激活值的量化参数。
def quantize_dynamic(model_input: Path,
                     model_output: Path,
                     op_types_to_quantize=[],
                     per_channel=False,
                     reduce_range=False,
                     activation_type=QuantType.QUInt8,
                     weight_type=QuantType.QUInt8,
                     nodes_to_quantize=[],
                     nodes_to_exclude=[],
                     optimize_model=True,
                     use_external_data_format=False,
                     extra_options = { }):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param activation_type: quantization data type of activation
    :param weight_type: quantization data type of weight
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    :parma use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        :param extra_options:
        key value pair dictionary for various options in different case. Current used:
            extra.Sigmoid.nnapi = True/False  (Default is False)
            ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
            WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
            EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                          Dyanmic mode currently is supported. Will support more in future.
            DisableShapeInference = True/False : in dynamic quantize mode, shape inference is not must have
                                                 and if it cause some issue, you could disable it.
            ForceQuantizeNoInputCheck = True/False : By default, some latent operators like maxpool, transpose, do not quantize
                                                     if their input is not quantized already. Setting to True to force such operator
                                                     always quantize input and so generate quantized output. Also the True behavior
                                                     could be disabled per node using the nodes_to_exclude.
            MatMulConstBOnly = True/False: Default is False. If enabled, only MatMul with const B will be quantized.
    '''

    mode = QuantizationMode.IntegerOps  # 设置量化模式为 IntegerOps（整数算子模式），对应 ONNX 早期的 IntegerOps 算子体系（如 ConvInteger、MatMulInteger）。
                                        # 动态量化在 ONNX Runtime v1.10.0 中默认使用该模式（与静态量化的 QLinearOps 模式区分），算子需显式处理整数计算逻辑。

"""若未手动指定 op_types_to_quantize（待量化的算子类型），则默认使用 IntegerOpsRegistry 中注册的所有支持动态量化的算子（如 Conv、MatMul、Gemm 等）。
IntegerOpsRegistry 是 ONNX Runtime 维护的 “支持 IntegerOps 模式的算子注册表”，确保仅对兼容算子进行量化。"""
    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)
    quantizer = ONNXQuantizer(
        model,
        per_channel,
        reduce_range,
        mode,
        False,  #static
        weight_type,
        activation_type,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options)
"""
上述创建 ONNX 量化器实例，传入动态量化所需的配置参数，关键参数解析：
    model：加载后的原始模型；
    per_channel：是否按通道量化权重（动态量化中通常为 False，即按张量量化）；
    reduce_range：是否用 7 位量化权重（减少精度损失，尤其适用于非 VNNI 指令集的 CPU）；
    mode：已设置的 IntegerOps 量化模式；
    False（static 参数）：明确当前为动态量化（静态量化需设为 True 并传入校准范围）；
    None（tensors_range 参数）：动态量化无需提前计算激活值范围，因此传 None；
    其余参数：控制量化数据类型、节点过滤规则、扩展配置。
"""
    quantizer.quantize_model()
"""
调用量化器的 quantize_model 方法，执行动态量化的核心逻辑：
  1.遍历模型计算图，识别 op_types_to_quantize 中的算子；
  2.对权重进行离线量化（提前计算权重的 scale/zero point，转换为低精度整数）；
  3.在激活值的输入 / 输出处插入 QuantizeLinear（量化）和 DeQuantizeLinear（反量化）节点（动态推理时，激活值会实时计算量化参数）；
  4.将原始算子（如 Conv）替换为对应的整数算子（如 ConvInteger），确保低精度推理流程通顺。
"""
    quantizer.model.save_model_to_file(model_output, use_external_data_format)


def quantize_qat(model_input: Path,
                 model_output: Path,
                 op_types_to_quantize=[],
                 per_channel=False,
                 reduce_range=False,
                 activation_type=QuantType.QUInt8,
                 weight_type=QuantType.QUInt8,
                 nodes_to_quantize=[],
                 nodes_to_exclude=[],
                 use_external_data_format=False):
    '''
        Given a quantize-aware traning onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param activation_type: quantization data type of activation
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    :parma use_external_data_format: option used for large size (>2GB) model. Set to False by default. 
    '''

    mode = QuantizationMode.IntegerOps

    #optimize the original model
    optimized_model = optimize_model(Path(model_input))

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    quantizer = ONNXQuantizer(
        optimized_model,
        per_channel,
        reduce_range,
        mode,
        False,  #static
        weight_type,
        activation_type,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)




"""
##解惑
1. 动态量化时，在激活值的输入 / 输出处插入 QuantizeLinear（量化）和 DeQuantizeLinear（反量化）节点，是指在激活函数前后插入量化/反量化节点吗？
答：在动态量化中，“在激活值的输入 / 输出处插入 QuantizeLinear 和 DeQuantizeLinear 节点”并不完全等同于在激活函数（如 ReLU、Sigmoid 等）前后插入，
而是指在算子（如 Conv、MatMul 等）的输入 / 输出张量（即激活值张量）的流动路径上插入。
具体来说，这里的 “激活值” 指的是神经网络中 layer 之间传递的张量（即中间特征图），而非特指激活函数本身。插入节点的核心目的是：在算子执行前将输入
激活值从 FP32 量化为低精度（如 INT8），算子执行后再反量化回 FP32，以匹配后续算子的输入要求。

举例说明：Conv 层的动态量化节点插入
假设原始计算流为：
    输入张量（FP32） → Conv 算子（FP32） → 输出张量（FP32）
    
    动态量化后，计算流变为：
    输入张量（FP32） → QuantizeLinear（量化为 INT8） → ConvInteger（整数卷积，INT8） → DeQuantizeLinear（反量化为 FP32） → 输出张量（FP32）
    
    这里的 QuantizeLinear 和 DeQuantizeLinear 节点插在 Conv 算子的输入和输出处，对应的是 Conv 层的输入激活值和输出激活值，与是否有激活函数无关。
若存在激活函数（如 ReLU），节点如何插入？
    假设原始计算流为：
    输入张量 → Conv → ReLU → 输出张量
    
    动态量化后，计算流为：
    输入张量 → QuantizeLinear → ConvInteger → DeQuantizeLinear → ReLU → QuantizeLinear → 下一层整数算子 → ...

可见：
 （1）Conv 算子的输入 / 输出处仍会插入量化 / 反量化节点（处理 Conv 的激活值）；
 （2）激活函数（ReLU）本身不直接插入量化节点，但其输出若作为下一个量化算子的输入，则会在下一个算子的输入处再次插入 QuantizeLinear。
 
核心结论：
    动态量化中，量化 / 反量化节点的插入位置由算子的输入 / 输出激活值张量决定，而非激活函数本身。
激活函数通常以 FP32 执行，仅在其输出作为下一个量化算子的输入时，才会在该算子前插入量化节点。
#################################################################################################################

2.静态量化，不在算子的输入输出处添加量化及反量化节点，它是怎么工作的？
答：静态量化（Post-Training Quantization, PTQ）并非完全不添加量化 / 反量化节点，而是根据量化格式的不同，有两种实现方式：
    QOperator 格式：直接将原始算子替换为量化算子（如 QLinearConv 替换 Conv），量化参数（scale/zero point）作为算子属性内置，无需显式的量化 / 反量化节点；
    QDQ 格式：与动态量化类似，会在算子输入 / 输出处插入 QuantizeLinear/DeQuantizeLinear 节点，但量化参数是通过校准数据提前计算好的（而非动态生成）。
关键区别：静态量化的核心是 “离线计算量化参数”，其核心特征是：通过校准数据（少量代表性输入）提前计算所有激活值和权重的量化参数（scale/zero point），推理时直接使用这些预计算的参数，无需动态调整。无论是否显式插入量化节点，量化参数都是固定的。

两种静态量化格式的工作原理
  1. QOperator 格式（无显式量化节点）
    算子替换：将原始 FP32 算子（如 Conv、MatMul）直接替换为对应的量化算子（如 QLinearConv、QLinearMatMul）。
    量化参数内置：量化算子的属性中包含预计算的 scale/zero point（权重和激活值的量化参数均通过校准获得）。
    计算流程：
       原始流程：输入（FP32）→ Conv（FP32）→ 输出（FP32）
       量化后：输入（FP32）→ QLinearConv（内置量化参数，直接计算 INT8）→ 输出（FP32）（量化和反量化的逻辑被封装在 QLinearConv 内部，无需显式节点）

  2. QDQ 格式（有显式量化节点，但参数固定）
    插入节点：在算子输入前插入 QuantizeLinear（用预计算的 scale/zero point 将 FP32 激活值量化为 INT8），输出后插入 DeQuantizeLinear（反量化回 FP32）。
    量化参数固定：QuantizeLinear/DeQuantizeLinear 的 scale/zero point 是通过校准数据提前计算好的，推理时不再变化。
    计算流程：
       输入（FP32）→ QuantizeLinear（用预计算参数量化为 INT8）→ ConvInteger（INT8 计算）→ DeQuantizeLinear（反量化为 FP32）→ 输出

静态量化为何不需要动态计算？
    静态量化通过 “校准数据” 覆盖了模型可能的输入分布，提前计算出所有激活值的量化范围（scale/zero point）。因此，推理时无论输入如何变化，都使用这些预计算的参数进行量化，
避免了动态量化中实时计算的开销，同时精度通常更高（因参数经过数据校准）。

总结：
静态量化的核心是 “离线校准 + 固定量化参数”，而非是否显式插入量化节点：
    QOperator 格式：量化逻辑封装在算子内部，无显式量化节点；
    QDQ 格式：显式插入量化节点，但参数通过校准固定；
  两种方式均依赖校准数据提前计算量化参数，推理时直接使用，效率和精度通常优于动态量化。
#################################################################################################################

3.动态量化与静态量化的差异：
*********************************************************************************************************************************************
对比维度	   *             动态量化（Dynamic Quantization）	               *  静态量化（Static Quantization）
*********************************************************************************************************************************************
量化参数       *     - 权重参数：离线提前计算（量化阶段固定）                  *   - 权重参数：离线提前计算（量化阶段固定）
计算时机       *     - 激活参数：推理时实时动态计算（每批输入可能不同）        *   - 激活参数：离线通过校准数据提前计算（推理时固定不变）
*********************************************************************************************************************************************
是否依赖       *     不依赖。无需提前准备代表性数据，直接对模型权重量化即可    *   强依赖。必须提供少量（通常几十～几百批）覆盖真实输入分布的 “校准数据”，
校准数据       *                                                               *   否则激活值量化参数不准确，精度损失大。
*********************************************************************************************************************************************
推理阶段       *     较高                                                      *   极低
计算开销       *                                                               *   
*********************************************************************************************************************************************
精度表现       *     较低                                                      *   较高
*********************************************************************************************************************************************
量化节点       *     必须在激活值的输入 / 输出处显式插入                       *   分两种格式，量化节点逻辑不同：
插入逻辑       *     QuantizeLinear（量化）和 DeQuantizeLinear（反量化）节点： *   1. QOperator 格式：无显式量化节点，将原始 FP32 算子（如 Conv）替换为量化算子（如 QLinearConv），量化参数内置在算子属性中；
               *     推理时，先通过 QuantizeLinear 用实时计算的参数量化激活值，*   2. QDQ 格式：显式插入 QuantizeLinear/DeQuantizeLinear 节点，但参数是离线校准后的固定值。
               *     再进入整数算子计算，最后用 DeQuantizeLinear 反量化输出。  *   
*********************************************************************************************************************************************                                                                                 
部署复杂度     *     低。无需准备校准数据，量化流程简单，仅需调用框架 API 对   *   较高。需额外步骤：
               *     模型权重量化，部署时无需额外处理输入数据分布。            *   1. 筛选 / 生成代表性校准数据（需匹配真实输入分布）；
               *                                                               *   2. 选择合适的校准算法（如 min-max、KL 散度）；
               *                                                               *   3. 验证校准效果（避免过校准 / 欠校准）。
*********************************************************************************************************************************************
#################################################################################################################

4.模型的初始化器指的是什么？
（1）在ONNXRuntime（以及ONNX标准）中，初始化器（Initializer）指的是模型中的常量张量，主要包括模型的权重、偏置等固定参数。在模型加载时就确定值的张量，在推理过程中保持不变。
（2）常见的初始化器类型：
     - 权重矩阵（Weight matrices）
     - 偏置向量（Bias vectors） 
     - 卷积核参数（Convolution kernels）
     - 批归一化参数（BatchNorm scale/bias）
     - 量化参数（Scale/Zero-point for quantization）
     - 常数张量（Constant tensors）
（3） 初始化器的内容示例
     Conv层的权重:
     - 名称: "conv1.weight"
     - 形状: [64, 3, 7, 7]  # [输出通道, 输入通道, 高, 宽]
     - 数据: 浮点数权重值
     
     偏置参数:
     - 名称: "conv1.bias" 
     - 形状: [64]
     - 数据: 偏置值
     
     量化参数:
     - 名称: "conv1_scale"
     - 形状: [1] 
     - 数据: [0.023] # scale值
（4） 在计算图中的位置
    输入张量(Input) + 初始化器张量(Weight) -> 算子(Conv) -> 输出张量
         ↓                    ↓                    ↓           ↓
      动态变化              固定不变            计算操作      结果


"""
