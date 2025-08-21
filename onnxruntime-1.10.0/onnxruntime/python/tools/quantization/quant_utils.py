import logging
import numpy
import onnx

from enum import Enum
from onnx import onnx_pb as onnx_proto
from pathlib import Path

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"

type_to_name = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
}

# Quantization mode
# IntegerOps: Use IntegerOps in quantized model. Only ConvInteger and MatMulInteger ops are supported now.
# QLinearOps: Use QLinearOps in quantized model. Only QLinearConv and QLinearMatMul ops are supported now.


class QuantizationMode(Enum):  #定义枚举类 QuantizationMode，继承自 Python 标准库的 Enum 类
    IntegerOps = 0
    QLinearOps = 1

    def __str__(self):  #重写 __str__ 方法（字符串转换方法）。作用：当打印 QuantizationMode 枚举成员时，返回成员的名称（如 print(QuantizationMode.QLinearOps) 会输出 "QLinearOps"）。
        return self.name  #self.name 是 Enum 类的内置属性，存储枚举成员的名称（如 QLinearOps）

    @staticmethod  #定义静态方法装饰器，标记后续方法 from_string 无需依赖类实例，可直接通过类名调用
    def from_string(mode): #定义静态方法 from_string，接收一个字符串参数 mode， “将字符串转换为对应的 QuantizationMode 枚举成员”。
        try:
            return QuantizationMode[mode]
        except KeyError:
            raise ValueError()


class QuantizedValueType(Enum):
    Input = 0
    Initializer = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(v):
        try:
            return QuantizedValueType[v]
        except KeyError:
            raise ValueError()


class QuantType(Enum):
    QInt8 = 0
    QUInt8 = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(t):
        try:
            return QuantType[t]
        except KeyError:
            raise ValueError()


class QuantFormat(Enum):
    QOperator = 0
    QDQ = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(format):
        try:
            return QuantFormat[format]
        except KeyError:
            raise ValueError()

ONNX_TYPE_TO_NP_TYPE = {
    onnx_proto.TensorProto.INT8: numpy.dtype('int8'),
    onnx_proto.TensorProto.UINT8:  numpy.dtype('uint8')
}

# 这个函数用于将 numpy 数组量化为指定的整数类型（INT8 或 UINT8），是 ONNX Runtime 中处理量化操作的核心函数之一。
"""qType：目标量化类型（如 ONNX 的 INT8 或 UINT8）     arr：需要量化的原始 numpy 数组
   scale：量化缩放因子（用于将浮点数映射到整数范围）   zero_point：量化零点（用于偏移映射，使 0 值在量化后保持一致）"""
"""该函数通过 “缩放 - 偏移 - 裁剪 - 类型转换” 四步，将原始浮点数组量化为指定的整数类型，确保量化结果在目标类型的有效范围内，是模型量化（降低精度以提升推理速度）的关键操作"""
def quantize_nparray(qType, arr, scale, zero_point, low=None, high=None):
    assert qType in ONNX_TYPE_TO_NP_TYPE, \
        "Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(qType)
    dtype = ONNX_TYPE_TO_NP_TYPE[qType]
    cliplow = max(0 if dtype == numpy.uint8 else -127, -127 if low is None else low) # 如果数据类型是numpy.uint8（无符号 8 位整数），则取值为 0，否则取值为 - 127
                                                                                     # 如果low参数为None，则取值为 - 127，否则使用low参数本身的值
    cliphigh = min(255 if dtype == numpy.uint8 else 127, 255 if high is None else high)
    arr_fp32 = numpy.asarray((arr.astype(numpy.float32) / scale).round() + zero_point)  """执行量化的核心转换，步骤如下：
                                                                                              将原始数组arr转换为float32类型（确保计算精度）。
                                                                                              除以缩放因子scale（将浮点数范围映射到整数范围）。
                                                                                              使用round()四舍五入为整数。
                                                                                              加上零点zero_point（偏移量，使量化后的数据分布更合理）。
                                                                                              最终转换为 numpy 数组，得到浮点型的量化中间结果。 """
    numpy.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)  #使用numpy.clip将量化中间结果arr_fp32限制在cliplow和cliphigh范围内, 结果直接存储回arr_fp32（避免创建新数组）。
    return arr_fp32.astype(dtype)   #将裁剪后的浮点数组转换为目标量化类型（dtype，即uint8或int8），并返回最终的量化数组。


"""
这个函数用于计算量化过程中的缩放因子（scale）和零点（zero point），是量化操作的核心辅助函数。
它基于原始数据范围和目标量化范围，建立原始值r与量化值q之间的映射关系r = s*(q-z)（其中s是 scale，z是 zero point）。
"""
# rmin：原始数据（未量化）的最小值; rmax：原始数据（未量化）的最大值; qmin：目标量化类型可表示的最小值（如 int8 的 - 128，uint8 的 0）; qmax：目标量化类型可表示的最大值（如 int8 的 127，uint8 的 255）;symmetric：是否使用对称量化（默认 False）
def compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=False):
    '''
    Calculate the scale s and zero point z for the quantization relation 
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values. 

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    详细说明函数功能：计算量化映射关系r = s*(q-z)中的s（scale）和z（zero point），
    确保原始范围[rmin, rmax]能被量化范围[qmin, qmax]近似表示，且z必须在[qmin, qmax]范围内。
    若symmetric=True，则原始范围会被对称化为[-absmax, absmax]（absmax是原始值绝对值的最大值）。

    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :return: zero and scale [z, s]

    '''
    
    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = min(rmin, 0)   #确保原始数据范围[rmin, rmax]包含 0 值。
    rmax = max(rmax, 0)   #这是为了保证零点z（对应原始值 0 的量化值）能落在量化范围[qmin, qmax]内，避免 0 无法被正确量化


"""接下来是对称量化处理，若启用对称量化（symmetric=True）：
      计算原始范围中绝对值的最大值absmax（取rmin和rmax绝对值中更大的那个）
      将原始范围调整为对称区间[-absmax, absmax]，确保量化后的分布关于 0 对称（常见于 int8 量化，能简化计算）"""
    if symmetric:
        absmax = max(abs(rmin), abs(rmax))
        rmin = -absmax
        rmax = +absmax

"""计算缩放因子，缩放因子s的计算公式：
       当原始范围不为单点（rmax != rmin）时，scale = (原始范围跨度) / (量化范围跨度)，表示 “每个量化单位对应的原始值大小”
       当原始范围为单点（rmax == rmin，所有值相同）时，避免除零错误，默认scale=1.0"""
    scale = (rmax - rmin) / float(qmax-qmin) if rmax!=rmin else 1.0
"""计算零点，零点z是原始值 0 对应的量化值，推导过程：
      由量化公式r = s*(q - z)，当r=0时，0 = s(z - z)不直接适用，需通过边界计算
      原始最小值rmin对应量化最小值qmin时，rmin = s(qmin - z)，变形得z = qmin - rmin/s
      用round()取整，确保z是整数（量化值必须为整数）"""
    zero_point = round(qmin - rmin/scale)

    return [zero_point, scale]


"""
该函数是量化流程的 “统筹者”，先计算原始数据的分布范围，再确定量化范围，接着计算量化所需的 scale 和 zero point，
最后调用具体量化函数完成转换，最终返回量化过程中的所有关键信息，为后续的模型推理或部署提供支持。

参数：data：需要量化的原始数据（通常是浮点型数组）
      qType：目标量化类型（支持 UINT8 和 INT8）
      symmetric：是否使用对称量化（主要用于 INT8 类型）
      reduce_range：是否缩小量化范围（可选参数，用于调整量化精度）
"""
def quantize_data(data, qType, symmetric, reduce_range=False):
    '''
    :param data: data to quantize
    :param qType: data type to quantize to. Supported types UINT8 and INT8
    :param symmetric: whether symmetric quantization is used or not. This is applied to INT8.
    :return: minimum, maximum, zero point, scale, and quantized weights

    To pack weights, we compute a linear transformation
    
    - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
    - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
        `m = max(abs(rmin), abs(rmax))`

    and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation

    :math:`r = S(q-z)`, where
    
    - *r*: real original value
    - *q*: quantized value
    - *S*: scale
    - *z*: zero point

    量化是通过线性变换实现的：
       UINT8 模式：将原始范围[rmin, rmax]映射到[0, 255]（2^8-1）
       INT8 模式（对称量化）：将原始范围[-m, m]（m是原始值绝对值的最大值）映射到[-127, 127]（-(2^7-1) 到 2^7-1）
    量化后的值通过公式r = s*(q-z)还原为原始值（r为原始值，q为量化值，S为 scale，z为 zero point）
    '''

    rmin = 0
    rmax = 0
    zero_point = 0
    scale = 1.0
    if len(data):   # 仅当输入数据data非空时（长度大于 0），才执行量化计算；若数据为空，则直接使用初始值
        rmin = min(data)
        rmax = max(data)
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range)   #根据目标量化类型（qType）和是否缩小范围（reduce_range），获取量化值的最小值（qmin）和最大值（qmax）

        zero_point, scale = compute_scale_zp(rmin, rmax, qmin, qmax, symmetric)

    quantized_data = quantize_nparray(qType, numpy.asarray(data), scale, zero_point)

    return rmin, rmax, zero_point, scale, quantized_data

def get_qmin_qmax_for_qType(qType, reduce_range=False):
    '''
    Return qmin and qmax, the minimum and maximum value representable by the given qType
    :parameter qType: onnx.onnx_pb.TensorProto.UINT8 or onnx.onnx_pb.TensorProto.UINT8
    :return: qmin, qmax
    '''
    if qType == onnx_proto.TensorProto.UINT8:
        (qmin, qmax) = (0,127) if reduce_range else (0,255)
    elif qType == onnx_proto.TensorProto.INT8:
        (qmin, qmax) = (-64,64) if reduce_range else (-127,127)
    else:
        raise ValueError("Unexpected data type {} requested. Only INT8 and UINT8 are supported.".format(qType))
    return qmin, qmax

def get_qrange_for_qType(qType, reduce_range=False):
    '''
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    '''
    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range)
    return  qmax - qmin


class QuantizedInitializer:   #该类用于统一管理权重的量化状态
    '''
        Represents a linearly quantized weight input from ONNX operators
        表示来自 ONNX 算子的线性量化权重输入。线性量化是指通过缩放因子（scale）和零点（zero point）将浮点数映射到整数的量化方式（如 FP32→INT8）。
    '''
    def __init__(self,
                 name,
                 initializer,
                 rmins,
                 rmaxs,
                 zero_points,
                 scales,
                 data=[],
                 quantized_data=[],
                 axis=None):
        self.name = name    #存储量化权重的名称（通常对应 ONNX 模型中权重张量的名称，如 conv1.weight），用于在模型中定位该权重。
        self.initializer = initializer  # TensorProto initializer in ONNX graph  # 存储原始的 ONNX 初始化器对象（TensorProto 类型），包含未量化的权重数据（FP32 格式）及张量形状、数据类型等元信息。
        self.rmins = rmins  # List of minimum range for each axis  """#存储每个轴（axis）上的最小值范围。
                                                                         #在按通道量化（per-channel quantization）时，每个通道（轴维度）会计算独立的量化范围，因此 rmins 是一个列表（长度等于轴的维度数）；
                                                                         #若为按张量量化（per-tensor），则列表仅含一个元素。"""
        self.rmaxs = rmaxs  # List of maximum range for each axis  #存储每个轴上的最大值范围，与 rmins 对应，共同确定量化的取值区间（[rmin, rmax]）。量化参数（scale/zero point）基于此范围计算。
        # 1D tensor of zero points computed for each axis. scalar if axis is empty
        self.zero_points = zero_points   """存储零点（量化后整数的偏移量），用于将浮点数映射到整数时消除偏移。
                                              #按通道量化时，是长度等于轴维度的 1D 张量（每个通道一个零点）；
                                              #按张量量化时，是单个标量（整个张量共用一个零点）。"""
        self.scales = scales  # 1D tensor of scales computed for each axis. scalar if axis is empty  #存储缩放因子（量化的比例系数），用于将浮点数范围映射到整数范围（如 INT8 的 [-128, 127]）。
                                                                                                          #与 zero_points 结构一致：按通道量化时为 1D 张量，按张量量化时为标量。
        self.data = data  # original data from initializer TensorProto   #存储从 initializer 中提取的原始浮点数据（通常为 FP32 类型的权重值），便于后续校验或重新计算
        self.quantized_data = quantized_data  # weight-packed data from data   #存储量化后的整数数据（如 INT8 类型）
        # Scalar to specify which dimension in the initializer to weight pack.
        self.axis = axis                                                               ##量化轴信息
        # If empty, single zero point and scales computed from a single rmin and rmax   """存储量化的轴索引（如卷积核权重中通常以输出通道为轴，即 axis=0）。
                                                                                           """按通道量化时，axis 指明哪个维度作为 “通道”（如 4D 卷积核 [out_channels, in_channels, h, w] 中，axis=0 表示按输出通道量化）；
                                                                                              按张量量化时，axis 为 None（不指定轴）。"""
                                                                                        

 
class QuantizedValue:   #关联量化值与其量化参数
    '''
    Represents a linearly quantized value (input\output\intializer)
    '''
    def __init__(self,
                 name,
                 new_quantized_name,
                 scale_name,
                 zero_point_name,
                 quantized_value_type,
                 axis=None):
        self.original_name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis


class BiasToQuantize:  #表示神经网络中需要被量化的偏置项（bias），并关联其在计算中依赖的输入和权重信息
    '''
    Represents a bias to be quantized

    该类用于建立偏置项与关联输入、权重的映射关系，因为：
       偏置的量化不能独立进行，必须结合输入和权重的量化参数（scale）计算自身的量化参数（具体公式为 bias_scale = input_scale × weight_scale）；
       通过名称关联，量化器（如 ONNXQuantizer）可以在量化过程中快速查找偏置对应的输入和权重的量化信息，确保计算正确；
       统一管理待量化的偏置项，避免遗漏或错误关联（尤其在复杂模型中，一个偏置可能对应特定的输入和权重）。
    '''
    def __init__(self, bias_name, input_name, weight_name):
        self.bias_name = bias_name
        self.input_name = input_name
        self.weight_name = weight_name


def attribute_to_kwarg(attribute):
    '''
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    '''
    if (attribute.type == 0):
        raise ValueError('attribute {} does not have type specified.'.format(attribute.name))

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    if (attribute.type == 1):
        value = attribute.f
    elif (attribute.type == 2):
        value = attribute.i
    elif (attribute.type == 3):
        value = attribute.s
    elif (attribute.type == 4):
        value = attribute.t
    elif (attribute.type == 5):
        value = attribute.g
    elif (attribute.type == 6):
        value = attribute.floats
    elif (attribute.type == 7):
        value = attribute.ints
    elif (attribute.type == 8):
        value = attribute.strings
    elif (attribute.type == 9):
        value = attribute.tensors
    elif (attribute.type == 10):
        value = attribute.graphs
    else:
        raise ValueError('attribute {} has unsupported type {}.'.format(attribute.name, attribute.type))

    return {attribute.name: value}


def find_by_name(item_name, item_list):
    '''
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    '''
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None


def get_elem_index(elem_name, elem_list):
    '''
    Helper function to return index of an item in a node list
    '''
    elem_idx = -1
    for i in range(0, len(elem_list)):
        if elem_list[i] == elem_name:
            elem_idx = i
    return elem_idx


def get_mul_node(inputs, output, name):
    '''
    Helper function to create a Mul node.
        parameter inputs: list of input names.
        parameter output: output name.
        parameter name: name of the node.
        return: Mul node in NodeProto format.
    '''
    return onnx.helper.make_node("Mul", inputs, [output], name)


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    '''
    Helper function to generate a identifiable filepath by concatenating the given identifier as a suffix.   
    '''
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)

"""
这段代码是一个校准表写入工具函数，功能是将模型量化校准过程中生成的 calibration_cache（校准缓存，存储张量量化范围信息）
以三种格式（JSON、FlatBuffers、纯文本）写入文件。
该函数在静态量化流程中通常用于 “保存校准结果”，后续量化时可直接加载校准文件，避免重复执行耗时的校准数据收集步骤。
"""
#参数： calibration_cache（校准缓存，通常是字典类型，键为张量名称，值为该张量的量化范围 [rmin, rmax]）。
def write_calibration_table(calibration_cache):
    '''
    Helper function to write calibration table to files.   
    '''

    import json
    import flatbuffers
    import onnxruntime.quantization.CalTableFlatBuffers.TrtTable as TrtTable
    import onnxruntime.quantization.CalTableFlatBuffers.KeyValue as KeyValue

    logging.info("calibration cache: {}".format(calibration_cache))

    with open("calibration.json", 'w') as file:
        file.write(json.dumps(calibration_cache))  # use `json.loads` to do the reverse

    # Serialize data using FlatBuffers
    builder = flatbuffers.Builder(1024)
    key_value_list = []
    for key in sorted(calibration_cache.keys()):
        values = calibration_cache[key]
        value = str(max(abs(values[0]), abs(values[1])))

        flat_key = builder.CreateString(key)
        flat_value = builder.CreateString(value)

        KeyValue.KeyValueStart(builder)
        KeyValue.KeyValueAddKey(builder, flat_key)
        KeyValue.KeyValueAddValue(builder, flat_value)
        key_value = KeyValue.KeyValueEnd(builder)

        key_value_list.append(key_value)

    TrtTable.TrtTableStartDictVector(builder, len(key_value_list))
    for key_value in key_value_list:
        builder.PrependUOffsetTRelative(key_value)
    main_dict = builder.EndVector(len(key_value_list))

    TrtTable.TrtTableStart(builder)
    TrtTable.TrtTableAddDict(builder, main_dict)
    cal_table = TrtTable.TrtTableEnd(builder)

    builder.Finish(cal_table)
    buf = builder.Output()

    with open("calibration.flatbuffers", 'wb') as file:
        file.write(buf)

    # Deserialize data (for validation)
    if False:
        cal_table = TrtTable.TrtTable.GetRootAsTrtTable(buf, 0)
        dict_len = cal_table.DictLength()
        for i in range(dict_len):
            key_value = cal_table.Dict(i)
            logging.info(key_value.Key())
            logging.info(key_value.Value())

    # write plain text
    with open("calibration.cache", 'w') as file:
        for key in sorted(calibration_cache.keys()):
            value = calibration_cache[key]
            s = key + ' ' + str(max(abs(value[0]), abs(value[1])))
            file.write(s)
            file.write('\n')


"""
1.该函数实现了离散分布平滑函数，用于处理量化校准（如熵校准）中可能出现的 “零概率” 问题，避免后续计算（如 KL 散度）时因零值导致的数值异常。
2.典型应用于深度学习模型量化的熵校准（Entropy Calibration） 流程：
 (1)熵校准中，需统计激活值的直方图（离散分布），并计算 “量化后分布” 与 “原始分布” 的 KL 散度，选择 KL 最小的量化范围；
 (2)若直方图中存在零值，计算 KL 散度时会出现 log(0)（无穷大），导致无法比较；
 (3)调用 smooth_distribution 后，零值被替换为小的正值，非零值扣除对应比例，既保留分布特征，又避免数值异常。
"""
def smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
         https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    """
    import numpy as np

    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    if not n_nonzeros:
        # raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        return -1
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)

    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0

    return hist
