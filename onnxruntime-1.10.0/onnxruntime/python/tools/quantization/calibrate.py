#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import numpy as np
import onnx
import onnxruntime
from onnx import helper, TensorProto, ModelProto
from onnx import onnx_pb as onnx_proto
from six import string_types
from enum import Enum

from .quant_utils import QuantType, smooth_distribution
from .registry import QLinearOpsRegistry

import abc
import itertools


class CalibrationMethod(Enum):
    MinMax = 0
    Entropy = 1
    Percentile = 2

class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_next') and callable(subclass.get_next) or NotImplemented)

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError

"""
定义用于模型校准（Calibration）的基类CalibraterBase，主要用于量化过程中收集张量（tensor）的动态范围信息（如最小值、最大值）。
校准是模型量化（将浮点模型转换为低精度整数模型）的关键步骤，直接影响量化模型的精度."""
"""
CalibraterBase是一个抽象基类，提供了校准流程的基础框架，包括模型加载、图增强（augment_graph）、校准张量选择、推理会话创建等通用功能。
具体的校准逻辑（如图增强方式、数据收集、范围计算）需要通过子类实现。"""
class CalibraterBase:
    def __init__(self, model, op_types_to_calibrate=[], augmented_model_path='augmented_model.onnx'):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.

        model：待校准的 ONNX 模型，可为模型路径（字符串）或onnx.ModelProto对象。
        op_types_to_calibrate：需要校准的算子类型列表，默认校准所有浮点（float32/float16）张量相关的算子。
        '''
        if isinstance(model, string_types):
            self.model = onnx.load(model)
        elif isinstance(model, ModelProto):
            self.model = model
        else:
            raise ValueError('model should be either model path or onnx.ModelProto.')

        self.op_types_to_calibrate = op_types_to_calibrate
        self.augmented_model_path = augmented_model_path

        # augment graph
        self.augment_model = None
        self.augment_graph()

        # Create InferenceSession
        self.infer_session = None
        self.execution_providers = ['CPUExecutionProvider']
        self._create_inference_session()

    def set_execution_providers(self, execution_providers=['CPUExecutionProvider']):
        '''
        reset the execution providers to execute the collect_data. It triggers to re-creating inference session.
        重置推理会话的执行提供者（如 CPU、GPU 等），并重新创建推理会话。
        '''
        self.execution_providers = execution_providers
        self._create_inference_session()

    def _create_inference_session(self):
        '''
        create an OnnxRuntime InferenceSession.
        '''
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.infer_session = onnxruntime.InferenceSession(self.augmented_model_path,
                                                          sess_options=sess_options,
                                                          providers=self.execution_providers)

    def select_tensors_to_calibrate(self, model):
        '''
        select all quantization_candidates op type nodes' input/output tensors. 
        returns:
            tensors (set): set of tensor name.
            value_infos (dict): tensor name to value info.
        '''
        """
        核心逻辑：
          1.收集模型中所有的张量信息：包括graph.value_info（中间张量）、graph.input（输入张量）、graph.output（输出张量）。
          2.排除初始化器（initializer，模型中的常量张量，无需校准）。
          3.对每个节点，若其算子类型在op_types_to_calibrate中（或列表为空时匹配所有算子），则检查其输入 / 输出张量：
              仅保留类型为 float32（TensorProto.FLOAT）或 float16（TensorProto.FLOAT16）的张量。
              最终得到需要校准的张量集合。
        """
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})
        initializer = set(init.name for init in model.graph.initializer)

        tensors_to_calibrate = set()
        tensor_type_to_calibrate = set([TensorProto.FLOAT, TensorProto.FLOAT16])

        for node in model.graph.node:
            if len(self.op_types_to_calibrate) == 0 or node.op_type in self.op_types_to_calibrate:
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos.keys():
                        vi = value_infos[tensor_name]
                        if vi.type.HasField('tensor_type') and (
                                vi.type.tensor_type.elem_type in tensor_type_to_calibrate) and (
                                    tensor_name not in initializer):
                            tensors_to_calibrate.add(tensor_name)

        return tensors_to_calibrate, value_infos

'''
下面这些抽象方法（需子类实现）：
这些方法定义了校准的核心流程，但具体逻辑由子类实现（如基于最小 - 最大范围的校准、移动平均校准等）。

  augment_graph：增强原始模型图。
    作用：通常在图中插入额外节点（如Identity节点），用于捕获待校准张量的数值，保存增强后的模型到augmented_model_path。
  collect_data：收集校准数据。
    作用：通过CalibrationDataReader读取输入数据，执行增强后的模型，收集待校准张量的实际数值（如推理过程中的中间张量值）。可多次调用以累积多批数据。
  compute_range：计算张量的动态范围。
    作用：基于collect_data收集的数据，计算每个待校准张量的 [min, max] 范围（用于量化时确定缩放因子）。
'''
    def get_augment_model(self):
        '''
        return: augmented onnx model
        '''
        return self.augment_model

    def augment_graph(self):
        '''
        abstract method: augment the input model to prepare for collecting data. It will:
            1. save augmented model to augmented_model_path.
            2. set the self.augment_model
        '''
        raise NotImplementedError

    def collect_data(self, data_reader: CalibrationDataReader):
        '''
        abstract method: collect the tensors that will be used for range computation. It can be called multiple times.
        '''
        raise NotImplementedError

    def compute_range(self, data_reader: CalibrationDataReader):
        '''
        abstract method: compute the [min, max] range for the tensors to calibrate based on the collected data.
        '''
        raise NotImplementedError

"""
该类是 CalibraterBase 的子类，实现了最小 - 最大校准（Min-Max Calibration）逻辑，用于计算待量化张量的动态范围（最小值和最大值），
是模型量化中最常用的校准方法之一."""
class MinMaxCalibrater(CalibraterBase):
    def __init__(self, model, op_types_to_calibrate=[], augmented_model_path='augmented_model.onnx'):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        '''
        super(MinMaxCalibrater, self).__init__(model, op_types_to_calibrate, augmented_model_path) #调用父类 CalibraterBase 的构造方法，完成模型加载、图增强触发等基础初始化。
        self.intermediate_outputs = [] #存储增强模型运行时产生的中间输出（主要是 ReduceMin 和 ReduceMax 节点的结果）。
        self.calibrate_tensors_range = None #最终存储每个待校准张量的 [min, max] 范围（字典类型，键为张量名，值为 (min, max) 元组）。
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(output.name for output in self.model.graph.output) #存储原始模型输出的名称集合（用于过滤掉非校准相关的输出）。

    def augment_graph(self): #增强原始模型图（核心是插入 ReduceMin 和 ReduceMax 节点，用于收集张量的最小值和最大值）。
        '''
        Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output
        :return: augmented ONNX model
        '''
        """先复制原始模型（避免修改原模型），再调用 onnx.shape_inference.infer_shapes 进行形状推断，确保所有张量的形状信息完整（后续创建 Reduce 节点需要依赖形状）。
        """
        model = onnx_proto.ModelProto()
        model.CopyFrom(self.model)
        model = onnx.shape_inference.infer_shapes(model)

        added_nodes = [] #added_nodes 存储新增的 ReduceMin/ReduceMax 节点；
        added_outputs = [] #added_outputs 存储这些节点的输出信息。
        tensors, value_infos = self.select_tensors_to_calibrate(model)  #获取需要校准的张量集合（tensors）和它们的类型信息（value_infos）。

        for tensor in tensors:  #遍历每个待校准的张量，为其创建 ReduceMin 和 ReduceMax 节点：

            # When doing ReduceMax/ReduceMin, ORT can't reduce on dim with value of 0 if 'keepdims' is false.
            # To make the code simple, we always let keepdims to be 1.
            keepdims = 1  #设置 Reduce 节点保留维度（避免因维度为 0 导致 ONNX Runtime 报错）

            # dim could be:
            #   [dim_param: "batch_size", dim_value: 256, dim_value: 36, dim_value: 64],
            #   [dim_value: 0],
            #   ...
            # Please see the definition of TensorShapeProto https://github.com/onnx/onnx/blob/master/onnx/onnx.proto#L651
            dim = value_infos[tensor].type.tensor_type.shape.dim  #获取当前张量的形状信息（来自 value_infos）
            shape = (1,) if len(dim) == 1 else tuple(1 for i in range(len(dim))) #构造 Reduce 节点输出的形状（全为 1 的维度，与原始张量维度数一致，例如原始形状为 (256, 36)，则 Reduce 输出形状为 (1, 1)）。

            # 为每个张量创建 ReduceMin 和 ReduceMax 节点：
            # Adding ReduceMin nodes
            reduce_min_name = tensor + '_ReduceMin'
            reduce_min_node = onnx.helper.make_node('ReduceMin', [tensor], [tensor + '_ReduceMin'], reduce_min_name, keepdims=keepdims)

            added_nodes.append(reduce_min_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_min_node.output[0], TensorProto.FLOAT, shape))

            # Adding ReduceMax nodes
            reduce_max_name = tensor + '_ReduceMax'
            reduce_max_node = onnx.helper.make_node('ReduceMax', [tensor], [tensor + '_ReduceMax'], reduce_max_name, keepdims=keepdims)

            added_nodes.append(reduce_max_node)
            added_outputs.append(helper.make_tensor_value_info(reduce_max_node.output[0], TensorProto.FLOAT, shape))

        model.graph.node.extend(added_nodes)
        model.graph.output.extend(added_outputs)
        onnx.save(model, self.augmented_model_path)
        self.augment_model = model

    def clear_collected_data(self):  #清空 self.intermediate_outputs 中收集的中间输出数据（用于重置校准状态，例如多轮数据收集后清理）。
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        self.compute_range()
        self.clear_collected_data()

    def merge_range(self, old_range, new_range):  #合并新旧两组张量范围（用于多轮数据收集时，累积全局的 min 和 max）
        if not old_range:
            return new_range

        for key, value in old_range.items(): 
            min_value = min(value[0], new_range[key][0])
            max_value = max(value[1], new_range[key][1])
            new_range[key] = (min_value, max_value)

        return new_range

    def compute_range(self):  #计算每个待校准张量的 [min, max] 范围。
        ''' 
        Compute the min-max range of tensor
        :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        '''

        if len(self.intermediate_outputs) == 0:
            return self.calibrate_tensors_range

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in self.intermediate_outputs
        ]

        merged_output_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_output_dict.setdefault(k, []).append(v)
        added_output_names = output_names[self.num_model_outputs:]
        calibrate_tensor_names = [
            added_output_names[i].rpartition('_')[0] for i in range(0, len(added_output_names), 2)
        ]  #output names

        merged_added_output_dict = dict(
            (i, merged_output_dict[i]) for i in merged_output_dict if i not in self.model_original_outputs)

        pairs = []
        for i in range(0, len(added_output_names), 2):
            min_value = 0
            max_value = 0
            min_value_array = min(merged_added_output_dict[added_output_names[i]])
            max_value_array = max(merged_added_output_dict[added_output_names[i + 1]])
            if type(min_value_array) == int or min_value_array.size > 0:
                min_value = float(min_value_array)
            if type(max_value_array) == int or max_value_array.size > 0:
                max_value = float(max_value_array)

            pairs.append(tuple([min_value, max_value]))

        new_calibrate_tensors_range = dict(zip(calibrate_tensor_names, pairs))
        if self.calibrate_tensors_range:
            self.calibrate_tensors_range = self.merge_range(self.calibrate_tensors_range, new_calibrate_tensors_range)
        else:
            self.calibrate_tensors_range = new_calibrate_tensors_range 

        return self.calibrate_tensors_range

class HistogramCalibrater(CalibraterBase):
    def __init__(self,
                 model,
                 op_types_to_calibrate=[],
                 augmented_model_path='augmented_model.onnx',
                 method='percentile',
                 num_quantized_bins=128,
                 percentile=99.99):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param method: A string. One of ['entropy', 'percentile'].
        :param num_quantized_bins: number of quantized bins. Default 128.
        :param percentile: A float number between [0, 100]. Default 99.99.
        '''
        super(HistogramCalibrater, self).__init__(model, op_types_to_calibrate, augmented_model_path)
        self.intermediate_outputs = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(output.name for output in self.model.graph.output)
        self.collector = None
        self.method = method
        self.num_quantized_bins = num_quantized_bins
        self.percentile = percentile

    def augment_graph(self):
        '''
        make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        '''
        model = onnx_proto.ModelProto()
        model.CopyFrom(self.model)
        model = onnx.shape_inference.infer_shapes(model)

        added_nodes = []
        added_outputs = []
        tensors, value_infos = self.select_tensors_to_calibrate(model) 

        for tensor in tensors:
            added_outputs.append(value_infos[tensor])

        model.graph.node.extend(added_nodes)
        model.graph.output.extend(added_outputs)
        onnx.save(model, self.augmented_model_path)
        self.augment_model = model

    def clear_collected_data(self):
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader):
        '''
        Entropy Calibrator collects operators' tensors as well as generates tensor histogram for each operator. 
        '''
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))


        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in self.intermediate_outputs
        ]

        merged_dict = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)

        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict if i not in self.model_original_outputs)

        if not self.collector:
            self.collector = HistogramCollector(method=self.method,
                                                num_quantized_bins=self.num_quantized_bins,
                                                percentile=self.percentile)
        self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_range(self):
        ''' 
        Compute the min-max range of tensor
        :return: dictionary mapping: {added node names: (ReduceMin, ReduceMax) pairs }
        '''
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        return self.collector.compute_collection_result()

class EntropyCalibrater(HistogramCalibrater):
    def __init__(self,
                 model,
                 op_types_to_calibrate=[],
                 augmented_model_path='augmented_model.onnx',
                 method='entropy',
                 num_quantized_bins=128):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param method: A string. One of ['entropy', 'percentile'].
        :param num_quantized_bins: number of quantized bins. Default 128.
        '''
        super(EntropyCalibrater, self).__init__(model, op_types_to_calibrate, augmented_model_path,
                                                method=method, num_quantized_bins=num_quantized_bins)

class PercentileCalibrater(HistogramCalibrater):
    def __init__(self,
                 model,
                 op_types_to_calibrate=[],
                 augmented_model_path='augmented_model.onnx',
                 method='percentile',
                 num_quantized_bins=2048,
                 percentile=99.999):
        '''
        :param model: ONNX model to calibrate. It can be a ModelProto or a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param method: A string. One of ['entropy', 'percentile'].
        :param num_quantized_bins: number of quantized bins. Default 128.
        :param percentile: A float number between [0, 100]. Default 99.99.
        '''
        super(PercentileCalibrater, self).__init__(model, op_types_to_calibrate, augmented_model_path,
                                                   method=method, num_quantized_bins=num_quantized_bins,
                                                   percentile=percentile)

class CalibrationDataCollector(metaclass=abc.ABCMeta):
    """
    Base class for collecting data for calibration-based quantization.
    """

    @abc.abstractmethod
    def collect(self, name_to_arr):
        """
        Generate informative data based on given data.
            name_to_arr : dict 
                tensor name to NDArray data 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_collection_result(self):
        """
        Get the optimal result among collection data.  
        """
        raise NotImplementedError

class HistogramCollector(CalibrationDataCollector):
    """
    Collecting histogram for each tensor. Percentile and Entropy method are supported.

    ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    ref: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/_modules/
                 pytorch_quantization/calib/histogram.html
    """
    def __init__(self, method, num_quantized_bins, percentile):
        self.histogram_dict = {}
        self.method = method
        self.num_quantized_bins= num_quantized_bins
        self.percentile = percentile

    def get_histogram_dict(self):
        return self.histogram_dict

    def collect(self, name_to_arr):
        # TODO: Currently we have different collect() for percentile and percentile method respectively.
        #       Need unified collect in the future.
        if self.method == 'entropy':
            return self.collect_for_entropy(name_to_arr)
        elif self.method == 'percentile':
            return self.collect_for_percentile(name_to_arr)
        else:
            raise ValueError('Only \'entropy\' or \'percentile\' method are supported')

    def collect_for_percentile(self, name_to_arr):
        for tensor, data_arr in name_to_arr.items():
            data_arr = np.asarray(data_arr)
            data_arr = data_arr.flatten()
            data_arr = np.absolute(data_arr) # only consider absolute value

            if tensor not in self.histogram_dict:
                # first time it uses num_quantized_bins to compute histogram.
                hist, hist_edges = np.histogram(data_arr, bins=self.num_quantized_bins)
                self.histogram_dict[tensor] = (hist, hist_edges)
            else:
                old_histogram = self.histogram_dict[tensor]
                old_hist = old_histogram[0]
                old_hist_edges = old_histogram[1]
                temp_amax = np.max(data_arr)
                if temp_amax > old_hist_edges[-1]:
                    # increase the number of bins
                    width = old_hist_edges[1] - old_hist_edges[0]
                    # NOTE: np.arange may create an extra bin after the one containing temp_amax
                    new_bin_edges = np.arange(old_hist_edges[-1] + width, temp_amax + width, width)
                    old_hist_edges = np.hstack((old_hist_edges, new_bin_edges))
                hist, hist_edges = np.histogram(data_arr, bins=old_hist_edges)
                hist[:len(old_hist)] += old_hist
                self.histogram_dict[tensor] = (hist, hist_edges)

    def collect_for_entropy(self, name_to_arr):
        for tensor, data_arr in name_to_arr.items():
            data_arr = np.asarray(data_arr)
            data_arr = data_arr.flatten()

            if data_arr.size > 0:
                min_value = np.min(data_arr)
                max_value = np.max(data_arr)
            else:
                min_value = 0
                max_value = 0

            threshold = max(abs(min_value), abs(max_value))

            if tensor in self.histogram_dict:
                old_histogram = self.histogram_dict[tensor]
                self.histogram_dict[tensor] = self.merge_histogram(old_histogram, data_arr, min_value, max_value, threshold)
            else:
                hist, hist_edges = np.histogram(data_arr, self.num_quantized_bins, range=(-threshold, threshold))
                self.histogram_dict[tensor] = (hist, hist_edges, min_value, max_value, threshold)

    def merge_histogram(self, old_histogram, data_arr, new_min, new_max, new_threshold):

        (old_hist, old_hist_edges, old_min, old_max, old_threshold) = old_histogram

        if new_threshold <= old_threshold:
            new_hist, _ = np.histogram(data_arr, len(old_hist), range=(-old_threshold, old_threshold))
            return (new_hist + old_hist, old_hist_edges, min(old_min, new_min), max(old_max, new_max), old_threshold)
        else:
            if old_threshold == 0:
                hist, hist_edges = np.histogram(data_arr, len(old_hist), range=(-new_threshold, new_threshold))
                hist += old_hist
            else:
                old_num_bins = len(old_hist)
                old_stride = 2 * old_threshold / old_num_bins
                half_increased_bins = int((new_threshold - old_threshold) // old_stride + 1) 
                new_num_bins = old_num_bins + 2 * half_increased_bins
                new_threshold = half_increased_bins * old_stride + old_threshold
                hist, hist_edges = np.histogram(data_arr, new_num_bins, range=(-new_threshold, new_threshold))
                hist[half_increased_bins:new_num_bins-half_increased_bins] += old_hist
            return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_threshold)

    def compute_collection_result(self):
        if not self.histogram_dict or len(self.histogram_dict) == 0:
            raise ValueError("Histogram has not been collected. Please run collect() first.")

        if self.method == 'entropy':
            return self.compute_entropy()
        elif self.method == 'percentile':
            return self.compute_percentile()
        else:
            raise ValueError('Only \'entropy\' or \'percentile\' method are supported')

    def compute_percentile(self):
        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        histogram_dict = self.histogram_dict
        percentile = self.percentile

        thresholds_dict = {} # per tensor thresholds

        for tensor, histogram in histogram_dict.items():
            hist = histogram[0]
            hist_edges = histogram[1]
            total = hist.sum()
            cdf = np.cumsum(hist/total)
            idx = np.searchsorted(cdf, percentile/100)
            thresholds_dict[tensor] = (float(hist_edges[idx]), float(hist_edges[idx]))

        return thresholds_dict

    def compute_entropy(self):
        histogram_dict = self.histogram_dict
        num_quantized_bins = self.num_quantized_bins

        thresholds_dict = {} # per tensor thresholds

        for tensor, histogram in histogram_dict.items():
            optimal_threshold = self.get_entropy_threshold(histogram, num_quantized_bins)
            thresholds_dict[tensor] = optimal_threshold

        return thresholds_dict

    def get_entropy_threshold(self, histogram, num_quantized_bins):
        """Given a dataset, find the optimal threshold for quantizing it.
        The reference distribution is `q`, and the candidate distribution is `p`.
        `q` is a truncated version of the original distribution.
        Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        """
        from scipy.stats import entropy
        import copy

        hist, hist_edges, _, _, _ = histogram
        num_bins = hist.size
        zero_bin_index = num_bins // 2
        num_half_quantized_bin = num_quantized_bins // 2
        
        kl_divergence = np.zeros(zero_bin_index - num_half_quantized_bin + 1)
        thresholds = [(0, 0) for i in range(kl_divergence.size)] 

        for i in range(num_half_quantized_bin, zero_bin_index + 1, 1):
            start_index = zero_bin_index - i 
            end_index = zero_bin_index + i + 1 if (zero_bin_index + i + 1) <= num_bins else num_bins

            thresholds[i - num_half_quantized_bin] = (float(hist_edges[start_index]), float(hist_edges[end_index]))

            sliced_distribution = copy.deepcopy(hist[start_index:end_index])

            # reference distribution p
            p = sliced_distribution.copy() # a copy of np array
            left_outliers_count = sum(hist[:start_index]) 
            right_outliers_count = sum(hist[end_index:])
            p[0] += left_outliers_count
            p[-1] += right_outliers_count

            # nonzeros[i] incidates whether p[i] is non-zero
            nonzeros = (p != 0).astype(np.int64)
            
            # quantize p.size bins into quantized bins (default 128 bins) 
            quantized_bins = np.zeros(num_quantized_bins, dtype=np.int64)
            num_merged_bins = sliced_distribution.size // num_quantized_bins

            # merge bins into quantized bins
            for index in range(num_quantized_bins):
                start = index * num_merged_bins 
                end = start + num_merged_bins
                quantized_bins[index] = sum(sliced_distribution[start:end]) 
            quantized_bins[-1] += sum(sliced_distribution[num_quantized_bins * num_merged_bins:])

            # in order to compare p and q, we need to make length of q equals to length of p
            # expand quantized bins into p.size bins
            q = np.zeros(p.size, dtype=np.int64)
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins

                norm = sum(nonzeros[start:end])
                if norm != 0:
                    q[start:end] = float(quantized_bins[index]) / float(norm)
            
            p = smooth_distribution(p)
            q = smooth_distribution(q)

            if isinstance(q, np.ndarray):
                kl_divergence[i - num_half_quantized_bin] = entropy(p, q)
            else:
                kl_divergence[i - num_half_quantized_bin] = float('inf')

        min_kl_divergence_idx = np.argmin(kl_divergence)
        optimal_threshold = thresholds[min_kl_divergence_idx] 

        return optimal_threshold


def create_calibrator(model,
                      op_types_to_calibrate=[],
                      augmented_model_path='augmented_model.onnx',
                      calibrate_method=CalibrationMethod.MinMax):
    if calibrate_method == CalibrationMethod.MinMax:
        return MinMaxCalibrater(model, op_types_to_calibrate, augmented_model_path)
    elif calibrate_method == CalibrationMethod.Entropy:
        return EntropyCalibrater(model, op_types_to_calibrate, augmented_model_path)
    elif calibrate_method == CalibrationMethod.Percentile:
        return PercentileCalibrater(model, op_types_to_calibrate, augmented_model_path)

    raise ValueError('Unsupported calibration method {}'.format(calibrate_method))
