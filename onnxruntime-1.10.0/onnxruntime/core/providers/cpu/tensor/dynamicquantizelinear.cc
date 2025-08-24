// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamicquantizelinear.h"

#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

/*
这段代码是 ONNX Runtime（ORT）中 CPU 平台的动态量化算子（DynamicQuantizeLinear）实现，针对 uint8_t 数据类型。
与静态量化（需提前通过校准获取 scale 和 zero_point）不同，动态量化在算子执行时实时计算输入张量的量化参数（scale 和 zero_point），
再完成浮点到整数的量化，适用于输入数据分布动态变化的场景。
*/
namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    DynamicQuantizeLinear,  // 算子名称（需与 ONNX 规范中的算子名一致）
    11,                     // 算子支持的 ONNX 版本（从 ONNX 11 版本开始支持该算子）
    uint8_t,              // 量化后的数据类型（此处为无符号 8 位整数）
    KernelDefBuilder()    // 内核定义构建器（约束算子输入输出类型、设备类型等）
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),   // 约束输出 Y（T2）的类型为 uint8_t
    DynamicQuantizeLinear<uint8_t>);  // 对应的内核类（模板类，实例化为 uint8_t 版本）

// formula is Y = X / Scale + ZeroPoint // 量化公式：Y = round(X / Scale) + ZeroPoint（X是输入浮点，Y是量化后uint8整数，需钳位到[0,255]）
//该方法实现 **「实时计算量化参数 → 输出参数 → 量化输入张量」** 的完整流程，
template <typename T>
Status DynamicQuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto x_ptr = ctx->Input<Tensor>(0);
  ORT_ENFORCE(x_ptr != nullptr);
  auto& x = *x_ptr;
  const auto* x_data = x.template Data<float>();
  const auto num_of_elements = x.Shape().Size();

  auto& y = *ctx->Output(0, x.Shape());
  std::vector<int64_t> shape({});
  auto& y_scale = *ctx->Output(1, shape);
  auto& y_zeropoint = *ctx->Output(2, shape);

  float scale;
  T zero_point;
  GetQuantizationParameter(x_data, num_of_elements, scale, zero_point, ctx->GetOperatorThreadPool());

  auto* output_scale = y_scale.template MutableData<float>();
  *output_scale = scale;

  auto* output_zp = y_zeropoint.template MutableData<T>();
  *output_zp = zero_point;

  // quantize the data
  auto* output = y.template MutableData<T>();
  ParQuantizeLinear(x_data, output, num_of_elements, scale, zero_point, ctx->GetOperatorThreadPool());

  return Status::OK();
}

}  // namespace onnxruntime
