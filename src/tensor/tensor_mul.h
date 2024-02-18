#pragma once

#include "sslib/blocked_allocator.h"
#include "device_tensor.h"
#include "device_memory_heap.h"
#include "tensor_common.h"

INFER_FLOW_BEGIN

using std::vector;
using std::map;
using sslib::BlockedAllocator;

class TensorMul
{
public:
    TensorMul();
    virtual ~TensorMul();

    //vector-matrix multiplication
    static bool Gemv(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, bool is_b_column_major = false,
        VectorMatrixMulAlg alg = VectorMatrixMulAlg::Auto);

    //U: An auxiliary tensor
    static bool Gemv_AX(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X,
        VectorMatrixMulAlg alg = VectorMatrixMulAlg::Auto,
        DeviceTensor *U = nullptr, const DeviceTensor *bias = nullptr);
    static bool Gemv_XA(DeviceTensor &Y, const DeviceTensor &X, const DeviceTensor &A,
        VectorMatrixMulAlg alg = VectorMatrixMulAlg::Auto);

    //GEMM
    static bool Gemm(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, float alpha = 1.0f, float beta = 0,
        bool is_b_column_major = false, bool is_sub_level = false,
        MatrixMulAlg alg = MatrixMulAlg::Auto);

#   ifdef USE_CUTLASS
    static bool Gemm_Cutlass(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, float alpha = 1.0f, float beta = 0,
        bool is_b_column_major = false, bool is_sub_level = false);
#   endif //USE_CUTLASS

    static bool Gemm_Alg1(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, float alpha = 1.0f, float beta = 0,
        bool is_b_column_major = false, bool is_sub_level = false);

    static bool Gemm_Alg2(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, float alpha, float beta,
        bool is_b_column_major, bool is_sub_level = false);

    static bool Gemm_Bruce(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, float alpha, float beta,
        bool is_b_column_major, bool is_sub_level = false);

    static bool GemmSparse(DeviceTensor &C, const DeviceTensor &A,
        const DeviceSparseMatrix &B, float alpha, float beta);

protected:
    static bool Gemm_Alg3(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, float alpha, float beta,
        bool is_b_column_major, bool is_sub_level = false);

    static bool Gemv_AX_QuantHalf(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_QuantQ8(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);

    static bool Gemv_Alg1(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, bool is_b_column_major = false);

    static bool Gemv_Alg2(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, bool is_b_column_major = false);

    static bool Gemv_XA_Alg3(DeviceTensor &Y, const DeviceTensor &X,
        const DeviceTensor &A);

    static bool Gemv_AX_Alg3(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X, const DeviceTensor *bias = nullptr);

    static bool Gemv_AX_Q8GL(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q8_B32T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q8_B32T2(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q6_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q5_B32T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q5_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q4B16(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q4_B32T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q4_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q3H_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q3_B32T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX_Q2_B32T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);

    static bool Gemv_AX8_Q8_B32T2(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);

    static bool Gemv_AX8_Q6_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX8_Q5_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX8_Q4_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX8_Q4_B32T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);
    static bool Gemv_AX8_Q3H_B64T1(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X);

    //U: An auxiliary tensor
    static bool Gemv_AX_Alg4(DeviceTensor &Y, const DeviceTensor &A,
        const DeviceTensor &X, DeviceTensor &U);

    static bool GemvCheckN(int cx, int min_cx, int m);
};

INFER_FLOW_END
