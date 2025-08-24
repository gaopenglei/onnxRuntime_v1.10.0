// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamicquantizelinear.h"

#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

/*
��δ����� ONNX Runtime��ORT���� CPU ƽ̨�Ķ�̬�������ӣ�DynamicQuantizeLinear��ʵ�֣���� uint8_t �������͡�
�뾲̬����������ǰͨ��У׼��ȡ scale �� zero_point����ͬ����̬����������ִ��ʱʵʱ������������������������scale �� zero_point����
����ɸ��㵽�������������������������ݷֲ���̬�仯�ĳ�����
*/
namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    DynamicQuantizeLinear,  // �������ƣ����� ONNX �淶�е�������һ�£�
    11,                     // ����֧�ֵ� ONNX �汾���� ONNX 11 �汾��ʼ֧�ָ����ӣ�
    uint8_t,              // ��������������ͣ��˴�Ϊ�޷��� 8 λ������
    KernelDefBuilder()    // �ں˶��幹������Լ����������������͡��豸���͵ȣ�
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),   // Լ����� Y��T2��������Ϊ uint8_t
    DynamicQuantizeLinear<uint8_t>);  // ��Ӧ���ں��ࣨģ���࣬ʵ����Ϊ uint8_t �汾��

// formula is Y = X / Scale + ZeroPoint // ������ʽ��Y = round(X / Scale) + ZeroPoint��X�����븡�㣬Y��������uint8��������ǯλ��[0,255]��
//�÷���ʵ�� **��ʵʱ������������ �� ������� �� ��������������** ���������̣�
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
