#pragma once

#include "common/cuda_util.h"
#ifdef __GNUC__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunknown-pragmas"
#else
#   pragma warning(push)
#   pragma warning(disable:4505)
#endif
#   include <cublas_v2.h>
#ifdef __GNUC__
#   pragma GCC diagnostic pop
#else
#   pragma warning(pop)
#endif
#include "device_tensor.h"
#include "device_memory_heap.h"

INFER_FLOW_BEGIN

class CublasEngine
{
public:
    CublasEngine() {};
    virtual ~CublasEngine();
    void Clear();

    bool Init();

    bool GemmEx(DeviceTensor &C, const DeviceTensor &A, const DeviceTensor &B,
        float alpha = 1.0f, float beta = 0, bool is_b_column_major = false,
        int alg_id = CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    bool GemmExBatch(DeviceTensor &C, const DeviceTensor &A, const DeviceTensor &B,
        DeviceMemoryHeap &heap, float alpha = 1.0f, float beta = 0,
        bool is_b_column_major = false,
        int alg_id = CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    bool GemmEx_F16(int m, int n, int k, const half *A, const half *B, half *C,
        float alpha, float beta, bool is_b_column_major, bool is_comp32,
        int alg_id);
    bool GemmEx_F32(int m, int n, int k, const float *A, const float *B, float *C,
        float alpha, float beta, bool is_b_column_major, int alg_id);

    bool Gemv(DeviceTensor &Y, const DeviceTensor &X, const DeviceTensor &A,
        float alpha = 1.0f, float beta = 0, bool is_b_column_major = false);

    bool Gemv_F16(int m, int n, const half *X, const half *B, half *Y,
        float alpha, float beta, bool is_b_column_major);
    bool Gemv_F32(int m, int n, const float *X, const float *B, float *Y,
        float alpha, float beta, bool is_b_column_major);

protected:
    cublasHandle_t handle_ = nullptr;

protected:
    bool GemmExBatch_F16(int m, int n, int k, const half **array_a,
        const half **array_b, half **array_c, float alpha, float beta,
        bool is_b_column_major, bool is_comp32, int batch_size, int alg_id);
    bool GemmExBatch_F32(int m, int n, int k, const float **array_a,
        const float **array_b, float **array_c, float alpha, float beta,
        bool is_b_column_major, int batch_size, int alg_id);

protected:
    cublasStatus_t GemmEx_Inner_FP16_Comp16(cublasOperation_t trans_a, cublasOperation_t trans_b,
        int m, int n, int k, const half *A, int lda, const half *B, int ldb, half *C, int ldc,
        const float *alpha, const float *beta, int alg_id);
    cublasStatus_t GemmEx_Inner_FP16_Comp32(cublasOperation_t trans_a, cublasOperation_t trans_b,
        int m, int n, int k, const half *A, int lda, const half *B, int ldb, half *C, int ldc,
        const float *alpha, const float *beta, int alg_id);

    cublasStatus_t GemmEx_Inner_FP32(cublasOperation_t trans_a, cublasOperation_t trans_b,
        int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *C, int ldc,
        const float *alpha, const float *beta, int alg_id);

    cublasStatus_t GemmExBatch_Inner_FP16_Comp16(cublasOperation_t trans_a,
        cublasOperation_t trans_b, int m, int n, int k, const half **array_a, int lda,
        const half **array_b, int ldb, half **array_c, int ldc,
        const float *alpha, const float *beta, int batch_size, int alg_id);
    cublasStatus_t GemmExBatch_Inner_FP16_Comp32(cublasOperation_t trans_a,
        cublasOperation_t trans_b, int m, int n, int k, const half **array_a, int lda,
        const half **array_b, int ldb, half **array_c, int ldc,
        const float *alpha, const float *beta, int batch_size, int alg_id);

    cublasStatus_t GemmExBatch_Inner_FP32(cublasOperation_t trans_a,
        cublasOperation_t trans_b, int m, int n, int k, const float **array_a, int lda,
        const float **array_b, int ldb, float **array_c, int ldc,
        const float *alpha, const float *beta, int batch_size, int alg_id);
};

INFER_FLOW_END
