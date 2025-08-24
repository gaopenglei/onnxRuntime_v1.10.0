// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/quantize_linear.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"

/*
��δ����� ONNX Runtime��ORT���� CPU ƽ̨��������QuantizeLinear���뷴������DequantizeLinear������ʵ�֣����Ĺ�������ɸ���������;�����������֮���ת����
��ģ����������Ĺؼ�����������Ϊ������׼�������������������ӡ����������ӡ����󲿷�.
*/

namespace onnxruntime {
/*
�ú�����������QuantizeLinear���ͷ�������DequantizeLinear���Ĺ��ù��ߺ������������������֡���������per-tensor�����͡���ͨ����per-channel������������ģʽ��
���������ѭ������Ŀ������block_count/broadcast_dim/block_size����ȷ������������scale/zero_point��������������״����.
*/
static void PrepareForQDQ(const TensorShape& input_shape,   //������������״���� [N, C, H, W]��
                          const Tensor& scale,              //���� / ����������������������per-tensor ʱΪ������per-channel ʱΪ 1D ������
                          const Tensor* zero_point_ptr,     //���� / �������������������ѡ�� nullptr ��ʾ����㣩
                          int64_t axis,                     //per-channel ģʽ�µ�Ŀ���ᣨ��ͨ���� C��֧�ָ��ᣬ�� -1 ��ʾ���һά��
                          int64_t& block_count,             //per-channel ʱ��Ӧ��֮ǰ��ά����Ԫ����
                          int64_t& broadcast_dim,           //per-channel ʱ��ӦĿ�����ά�ȴ�С
                          int64_t& block_size) {            //per-channel ʱ��Ӧ��֮���ά����Ԫ����
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear  //��������per-tensor��ģʽ, ������ģʽ����򵥵�������ʽ ����������������һ�� scale �� zero_point��scale Ϊ������ 1 Ԫ��������zero_point ͬ����
    block_count = 1;     //��1���飨��������Ϊ1���飩
    broadcast_dim = 1;   // �㲥ά�ȴ�СΪ1����ͨ�����֣�
    block_size = static_cast<size_t>(input_shape.Size());  // ����Ԫ���� = ����������Ԫ����

    // enforce that zero point are scalars   // ǿ�Ƽ�飺zero_point ����Ϊ null �򡸱���/1Ԫ������������ scale ģʽһ�£�
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),  //ORT ���ù��ߺ������ж������Ƿ�Ϊ����������ά����Ϊ 0����1D ��Ԫ����Ϊ 1 ��������������״ [1]��
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel QuantizeLinear/DequantizeLinear, ��ͨ��ģʽΪÿ��ͨ����ָ�����ÿ��ά�ȣ���������� scale �� zero_point������������ͨ����ÿ��ͨ����Ӧһ�� scale�������ȴ�������������⣬�ټ���������
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());   // ����1��������תΪ���ᣨ�� axis=-1 ��Ӧ���һά��axis=-2 ��Ӧ�����ڶ�ά��
     // ����2��������������������״ [N, C, H, W]��axis=1 Ϊ����
    block_count = input_shape.SizeToDimension(axis_no_neg);  // axis֮ǰ��ά�ȳ˻���N��axis=1 ǰֻ�� N, ��N��ͨ����
    broadcast_dim = input_shape[axis_no_neg];                // axisά�ȵĴ�С��C��ͨ������;
    block_size = input_shape.SizeFromDimension(axis_no_neg + 1); // axis֮���ά�ȳ˻���H*W;
   /* ʾ����
      ��������: [2, 64, 8, 8], axis=1 (ͨ����)
        - axis_no_neg = 1
        - block_count = 2 (batch size)
        - broadcast_dim = 64 (ͨ����)  
        - block_size = 64 (8*8, ÿ��ͨ���Ĵ�С)
   */

    // if an axis was specified, ensure the scale and zero point are compatible   //����3����� scale ��״�Ϸ��ԣ������� 1D �Ҵ�С = ͨ������
    ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                "scale must be 1D tensor with size ",
                broadcast_dim);

    // ����4����� zero_point ��״�Ϸ��ԣ�ͬ scale����Ϊ null��
    ORT_ENFORCE(zero_point_ptr == nullptr || (zero_point_ptr->Shape().NumDimensions() == 1 && zero_point_ptr->Shape()[0] == broadcast_dim),
                "x_zero_point must be null or 1D tensor with size ",
                broadcast_dim);
  }
}

/*
�������ĺ����ǽ��;��������������� int8/uint8���ָ�Ϊ������������ʽΪ��Y = (X - ZeroPoint) * Scale��X ��������������Y �Ƿ������󸡵�������
�����Ϊ������ע�᡹�͡������߼���Compute �������������֡�
*/
// �꣺ע�� ONNX 13 �汾�����ϵ� DequantizeLinear ���ӣ�ָ���������� T��
#define REGISTER_DEQUANTIZELINEAR(T)                              \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                 \
      DequantizeLinear,                                           \
      13,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

// �꣺ע�� ONNX 10~12 �汾�� DequantizeLinear ���ӣ����ݾɰ汾��
#define REGISTER_DEQUANTIZELINEAR_VERSIONED(T)                    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DequantizeLinear,                                           \
      10,                                                         \
      12,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

// ע������������͵����ӣ�int8/uint8/int32�����ǰ汾 10~13��
REGISTER_DEQUANTIZELINEAR(int8_t)
REGISTER_DEQUANTIZELINEAR(uint8_t)
REGISTER_DEQUANTIZELINEAR(int32_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t)

//ģ���� DequantizeLinear<T> �� Compute �����Ƿ������ĺ��ģ�ʵ�ֹ�ʽ Y = (X - ZeroPoint) * Scale��֧�� per-tensor �� per-channel ģʽ
// formula is Y = (X - ZeroPoint) * Scale
template <typename T>
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {  //OpKernelContext* ctx��ORT ���������ģ����ڻ�ȡ��������������������������������Դ�����̳߳أ���
  //���� 1����ȡ���� / �������
  auto& x = *ctx->Input<Tensor>(0);   //��ȡ����������0-����������ݣ�X����1-�������ӣ�Scale����2-��㣨ZeroPoint����ѡ��
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto* x_zero_point = ctx->Input<Tensor>(2);

  // ��ȡ����������״������������� Y����״�� X ��ȫһ�£�����Ϊ float��
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);  //����������������� 0 ��ʾ��һ���������״������ x_shape һ�£����������ı�������״����

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;

  //���� 2������ PrepareForQDQ ��ȡ�����
  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, N, broadcast_dim, block_size); // ���ø������������ N��broadcast_dim��block_size

  //���� 3����ȡ��������ָ�루�ڴ���ʣ�
  const float* scale = x_scale.template Data<float>();  // ��ȡ Scale ����ָ�루Scale ʼ���� float ���ͣ�
  const T* input = x.template Data<T>();  // ��ȡ���� X ����ָ�루����Ϊ T���� int8_t��
  float* output = y.template MutableData<float>(); // ��ȡ��� Y ����ָ�루��������Ϊ float ���ͣ�MutableData ��ʾ��д��
  /*
     template Data<T>()��ORT ������ģ�巽���������������ݵ� const ָ�루ȷ��ֻ������MutableData<T>() ���ؿ�дָ�롣
     ע�⣺Scale ʼ���� float ���ͣ�ONNX �淶�������������� T �޹ء�
  */

  //���� 4�������飨int32_t ���������Լ����
  const T* zero_point = x_zero_point ? x_zero_point->template Data<T>() : nullptr;  // ��ȡ ZeroPoint ����ָ�루�����ڣ�.
  if (std::is_same<T, int32_t>::value) {   //����Լ���������������� int32_t��ZeroPoint ����Ϊ null ��ȫ 0�� ԭ��int32 ͨ�����ڡ�α���������м�洢��������㣨ZeroPoint �����Ӽ��㿪�����޾������棩�����ǿ��Լ����
    ORT_ENFORCE(zero_point == nullptr ||
                    std::all_of(zero_point,
                                zero_point + x_zero_point->Shape().Size(),
                                [](int32_t zp) { return zp == 0; }),
                "DequantizeLinear with type int32 should have no zero point or all zero points should be 0");
  }

  //���� 5������ѭ��ִ�з��������ڴ��Ѻ���˳��;ѭ��˳��˵�������������� �� ͨ�� �� Ԫ�ء�ѭ�������� CPU �ڴ�ֲ���ԭ���������������ڴ棬���ٻ��� miss��������Ч�ʡ�
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      auto zp = zero_point ? static_cast<int32_t>(zero_point[bd]) : 0;
      auto sc = scale[bd];

      for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++) {
        *output++ = static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc; //static_cast<int32_t>(*input++)������������ T תΪ int32_t������ T Ϊ int8_t ʱ������������
      }                                                                           //�� zp����ǰͨ������㣩���� sc����ǰͨ�����������ӣ���
    }
  }

  return Status::OK();
}


/*
�����ĺ����ǽ���������ת��Ϊ�;���������������ʽΪ Y = round(X / Scale) + ZeroPoint��ʵ����ǯλ��������Χ���� int8_t ��ǯλ�� [-128, 127]����
����ṹ�뷴������ȫ�Գƣ���Ϊ������ע�᡹�͡������߼�����
*/
#define REGISTER_QUANTIZELINEAR(T)                                    \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                     \
      QuantizeLinear,                                                 \
      13,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \  // ����ǰ��float
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \  // ������T���� int8��
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
//ģ���� QuantizeLinear<T> �� Compute ����ʵ�����������߼�����ʽΪ Y = round(X / Scale) + ZeroPoint��ʵ��ǯλ�߼��� ParQuantizeLinear �У���
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  //���� 1����ȡ���� / �������
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);    //����˳��QuantizeLinear �� ONNX �淶����˳��Ϊ��X��float���� Scale �� ZeroPoint�������Ϊ Y��������������

  //���� 2������ PrepareForQDQ ��ȡ�����
  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, N, broadcast_dim, block_size);

  //���� 3����ȡ��������ָ��
  const T* zero_point = y_zero_point != nullptr ? y_zero_point->template Data<T>() : nullptr;
  const float* scale = y_scale.template Data<float>();
  const float* input = x.template Data<float>();
  T* output = y.template MutableData<T>();

  //���� 4���������������� ParQuantizeLinear��
  /*
    ParQuantizeLinear��ORT �ڲ�����������������Par����ʾ Parallel�������Ĺ��ܣ�
      ���� X / Scale�������������
      �������루round�������������
      �� ZeroPoint��
      ǯλ�� T ��ȡֵ��Χ���� int8_t ǯλ�� [-128, 127]��uint8_t ǯλ�� [0, 255]����
      �����̳߳ز��д�������������������Ч�ʡ�
  */
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {  // ѭ��˳��block_count��N���� broadcast_dim��ͨ����
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      T zp = zero_point != nullptr ? zero_point[bd] : 0;
      ParQuantizeLinear(input, output, static_cast<size_t>(block_size), scale[bd], zp, ctx->GetOperatorThreadPool());   // �������� block_size ��Ԫ�أ������Ż��������̳߳أ�
      input += block_size;
      output += block_size;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
