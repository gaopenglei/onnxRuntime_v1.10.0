// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/quantize_linear.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"

/*
这段代码是 ONNX Runtime（ORT）中 CPU 平台的量化（QuantizeLinear）与反量化（DequantizeLinear）算子实现，核心功能是完成浮点张量与低精度整数张量之间的转换，
是模型量化推理的关键组件。代码分为「辅助准备函数」「反量化算子」「量化算子」三大部分.
*/

namespace onnxruntime {
/*
该函数是量化（QuantizeLinear）和反量化（DequantizeLinear）的共用工具函数，核心作用是区分「逐张量（per-tensor）」和「逐通道（per-channel）」两种量化模式，
并计算后续循环所需的块参数（block_count/broadcast_dim/block_size），确保量化参数（scale/zero_point）与输入张量形状兼容.
*/
static void PrepareForQDQ(const TensorShape& input_shape,   //输入张量的形状（如 [N, C, H, W]）
                          const Tensor& scale,              //量化 / 反量化的缩放因子张量（per-tensor 时为标量，per-channel 时为 1D 张量）
                          const Tensor* zero_point_ptr,     //量化 / 反量化的零点张量（可选， nullptr 表示无零点）
                          int64_t axis,                     //per-channel 模式下的目标轴（如通道轴 C，支持负轴，如 -1 表示最后一维）
                          int64_t& block_count,             //per-channel 时对应轴之前的维度总元素数
                          int64_t& broadcast_dim,           //per-channel 时对应目标轴的维度大小
                          int64_t& block_size) {            //per-channel 时对应轴之后的维度总元素数
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear  //逐张量（per-tensor）模式, 逐张量模式是最简单的量化方式 ——整个张量共用一个 scale 和 zero_point（scale 为标量或 1 元素向量，zero_point 同理）。
    block_count = 1;     //仅1个块（整个张量为1个块）
    broadcast_dim = 1;   // 广播维度大小为1（无通道区分）
    block_size = static_cast<size_t>(input_shape.Size());  // 块内元素数 = 输入张量总元素数

    // enforce that zero point are scalars   // 强制检查：zero_point 必须为 null 或「标量/1元素向量」（与 scale 模式一致）
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),  //ORT 内置工具函数，判断张量是否为「标量」（维度数为 0）或「1D 且元素数为 1 的向量」（如形状 [1]）
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel QuantizeLinear/DequantizeLinear, 逐通道模式为每个通道（指定轴的每个维度）分配独立的 scale 和 zero_point（如卷积层的输出通道，每个通道对应一个 scale），需先处理轴的正负问题，再计算块参数。
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());   // 步骤1：将负轴转为正轴（如 axis=-1 对应最后一维，axis=-2 对应倒数第二维）
     // 步骤2：计算块参数（以输入形状 [N, C, H, W]、axis=1 为例）
    block_count = input_shape.SizeToDimension(axis_no_neg);  // axis之前的维度乘积：N（axis=1 前只有 N, 有N组通道）
    broadcast_dim = input_shape[axis_no_neg];                // axis维度的大小：C（通道数）;
    block_size = input_shape.SizeFromDimension(axis_no_neg + 1); // axis之后的维度乘积：H*W;
   /* 示例：
      输入张量: [2, 64, 8, 8], axis=1 (通道轴)
        - axis_no_neg = 1
        - block_count = 2 (batch size)
        - broadcast_dim = 64 (通道数)  
        - block_size = 64 (8*8, 每个通道的大小)
   */

    // if an axis was specified, ensure the scale and zero point are compatible   //步骤3：检查 scale 形状合法性（必须是 1D 且大小 = 通道数）
    ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                "scale must be 1D tensor with size ",
                broadcast_dim);

    // 步骤4：检查 zero_point 形状合法性（同 scale，或为 null）
    ORT_ENFORCE(zero_point_ptr == nullptr || (zero_point_ptr->Shape().NumDimensions() == 1 && zero_point_ptr->Shape()[0] == broadcast_dim),
                "x_zero_point must be null or 1D tensor with size ",
                broadcast_dim);
  }
}

/*
反量化的核心是将低精度整数张量（如 int8/uint8）恢复为浮点张量，公式为：Y = (X - ZeroPoint) * Scale（X 是量化后整数，Y 是反量化后浮点数）。
代码分为「算子注册」和「计算逻辑（Compute 方法）」两部分。
*/
// 宏：注册 ONNX 13 版本及以上的 DequantizeLinear 算子（指定数据类型 T）
#define REGISTER_DEQUANTIZELINEAR(T)                              \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                 \
      DequantizeLinear,                                           \
      13,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

// 宏：注册 ONNX 10~12 版本的 DequantizeLinear 算子（兼容旧版本）
#define REGISTER_DEQUANTIZELINEAR_VERSIONED(T)                    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DequantizeLinear,                                           \
      10,                                                         \
      12,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

// 注册具体数据类型的算子（int8/uint8/int32，覆盖版本 10~13）
REGISTER_DEQUANTIZELINEAR(int8_t)
REGISTER_DEQUANTIZELINEAR(uint8_t)
REGISTER_DEQUANTIZELINEAR(int32_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t)

//模板类 DequantizeLinear<T> 的 Compute 方法是反量化的核心，实现公式 Y = (X - ZeroPoint) * Scale，支持 per-tensor 和 per-channel 模式
// formula is Y = (X - ZeroPoint) * Scale
template <typename T>
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {  //OpKernelContext* ctx：ORT 算子上下文，用于获取输入张量、创建输出张量、管理计算资源（如线程池）。
  //步骤 1：获取输入 / 输出张量
  auto& x = *ctx->Input<Tensor>(0);   //获取输入张量：0-量化后的数据（X），1-缩放因子（Scale），2-零点（ZeroPoint，可选）
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto* x_zero_point = ctx->Input<Tensor>(2);

  // 获取输入张量形状，创建输出张量 Y（形状与 X 完全一致，类型为 float）
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);  //创建输出张量，索引 0 表示第一个输出，形状与输入 x_shape 一致（反量化不改变张量形状）。

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;

  //步骤 2：调用 PrepareForQDQ 获取块参数
  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, N, broadcast_dim, block_size); // 调用辅助函数，填充 N、broadcast_dim、block_size

  //步骤 3：获取张量数据指针（内存访问）
  const float* scale = x_scale.template Data<float>();  // 获取 Scale 数据指针（Scale 始终是 float 类型）
  const T* input = x.template Data<T>();  // 获取输入 X 数据指针（类型为 T，如 int8_t）
  float* output = y.template MutableData<float>(); // 获取输出 Y 数据指针（反量化后为 float 类型，MutableData 表示可写）
  /*
     template Data<T>()：ORT 张量的模板方法，返回张量数据的 const 指针（确保只读）；MutableData<T>() 返回可写指针。
     注意：Scale 始终是 float 类型（ONNX 规范），与量化类型 T 无关。
  */

  //步骤 4：特殊检查（int32_t 量化的零点约束）
  const T* zero_point = x_zero_point ? x_zero_point->template Data<T>() : nullptr;  // 获取 ZeroPoint 数据指针（若存在）.
  if (std::is_same<T, int32_t>::value) {   //特殊约束：若量化类型是 int32_t，ZeroPoint 必须为 null 或全 0。 原因：int32 通常用于「伪量化」或中间存储，无需零点（ZeroPoint 会增加计算开销且无精度收益），因此强制约束。
    ORT_ENFORCE(zero_point == nullptr ||
                    std::all_of(zero_point,
                                zero_point + x_zero_point->Shape().Size(),
                                [](int32_t zp) { return zp == 0; }),
                "DequantizeLinear with type int32 should have no zero point or all zero points should be 0");
  }

  //步骤 5：三重循环执行反量化（内存友好型顺序）;循环顺序说明：按「块数量 → 通道 → 元素」循环，符合 CPU 内存局部性原理（连续访问相邻内存，减少缓存 miss），提升效率。
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      auto zp = zero_point ? static_cast<int32_t>(zero_point[bd]) : 0;
      auto sc = scale[bd];

      for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++) {
        *output++ = static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc; //static_cast<int32_t>(*input++)：将量化整数 T 转为 int32_t（避免 T 为 int8_t 时减零点溢出）；
      }                                                                           //减 zp（当前通道的零点）；乘 sc（当前通道的缩放因子）；
    }
  }

  return Status::OK();
}


/*
量化的核心是将浮点张量转换为低精度整数张量，公式为 Y = round(X / Scale) + ZeroPoint（实际需钳位到整数范围，如 int8_t 需钳位到 [-128, 127]）。
代码结构与反量化完全对称，分为「算子注册」和「计算逻辑」。
*/
#define REGISTER_QUANTIZELINEAR(T)                                    \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                     \
      QuantizeLinear,                                                 \
      13,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \  // 量化前：float
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \  // 量化后：T（如 int8）
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED(T)                          \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      QuantizeLinear,                                                 \
      10,                                                             \
      12,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);

REGISTER_QUANTIZELINEAR(int8_t)
REGISTER_QUANTIZELINEAR(uint8_t)
REGISTER_QUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t)

// formula is Y = X / Scale + ZeroPoint
//模板类 QuantizeLinear<T> 的 Compute 方法实现量化核心逻辑，公式为 Y = round(X / Scale) + ZeroPoint（实际钳位逻辑在 ParQuantizeLinear 中）。
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  //步骤 1：获取输入 / 输出张量
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);    //输入顺序：QuantizeLinear 的 ONNX 规范输入顺序为「X（float）→ Scale → ZeroPoint」，输出为 Y（量化整数）。

  //步骤 2：调用 PrepareForQDQ 获取块参数
  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, N, broadcast_dim, block_size);

  //步骤 3：获取张量数据指针
  const T* zero_point = y_zero_point != nullptr ? y_zero_point->template Data<T>() : nullptr;
  const float* scale = y_scale.template Data<float>();
  const float* input = x.template Data<float>();
  T* output = y.template MutableData<T>();

  //步骤 4：批量量化（调用 ParQuantizeLinear）
  /*
    ParQuantizeLinear：ORT 内部批量量化函数（「Par」表示 Parallel），核心功能：
      计算 X / Scale（浮点除法）；
      四舍五入（round）到最近整数；
      加 ZeroPoint；
      钳位到 T 的取值范围（如 int8_t 钳位到 [-128, 127]，uint8_t 钳位到 [0, 255]）；
      利用线程池并行处理，提升大张量的量化效率。
  */
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {  // 循环顺序：block_count（N）→ broadcast_dim（通道）
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      T zp = zero_point != nullptr ? zero_point[bd] : 0;
      ParQuantizeLinear(input, output, static_cast<size_t>(block_size), scale[bd], zp, ctx->GetOperatorThreadPool());   // 批量处理 block_size 个元素（并行优化，传入线程池）
      input += block_size;
      output += block_size;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
