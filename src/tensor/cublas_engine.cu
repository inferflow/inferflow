#include "cublas_engine.h"
#include "sslib/log.h"
#include <cuda_profiler_api.h>

INFER_FLOW_BEGIN

using namespace sslib;

CublasEngine::~CublasEngine()
{
    Clear();
}

void CublasEngine::Clear()
{
    cublasDestroy(handle_);
}

bool CublasEngine::Init()
{
    auto ret_code = cublasCreate(&handle_);
    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasCreate", ret_code);
        return false;
    }

    //ret_code = cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH);
    ret_code = cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    //ret_code = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasSetMathMode", ret_code);
        return false;
    }

    return true;
}

bool CublasEngine::GemmEx(DeviceTensor &C, const DeviceTensor &A, const DeviceTensor &B,
    float alpha, float beta, bool is_b_column_major, bool is_comp32, int alg_id)
{
    bool ret = true;
    int m = A.Rows(), k = A.Columns();
    int n = is_b_column_major ? B.Rows() : B.Columns();

    bool is_fp16 = A.data_type == ElementType::F16 && B.data_type == ElementType::F16
        && C.data_type == ElementType::F16;
    bool is_fp32 = A.data_type == ElementType::F32 && B.data_type == ElementType::F32
        && C.data_type == ElementType::F32;

    //cudaDeviceSynchronize();
    //cudaProfilerStart();

    if (is_fp16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();
        ret = GemmEx_F16(m, n, k, a_data, b_data, c_data, alpha, beta,
            is_b_column_major, is_comp32, alg_id);
    }
    else if (is_fp32)
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();
        ret = GemmEx_F32(m, n, k, a_data, b_data, c_data, alpha, beta,
            is_b_column_major, alg_id);
    }
    else
    {
        LogError("Not implemented yet (should both be fp16 or fp32)!");
        ret = false;
    }

    //cudaProfilerStop();
    return ret;
}

bool CublasEngine::GemmExBatch(DeviceTensor &C, const DeviceTensor &A, const DeviceTensor &B,
    DeviceMemoryHeap &heap, float alpha, float beta, bool is_b_column_major, int alg_id)
{
    bool ret = true;
    if (is_b_column_major)
    {
        if (A.ne[2] != B.ne[2] || A.ne[0] != B.ne[0])
        {
            LogError("A and B are not compatible: %d vs. %d and %d vs. %d",
                A.ne[2], B.ne[2], A.ne[0], B.ne[0]);
            return false;
        }

        if (A.ne[2] != C.ne[2] || C.ne[1] != A.ne[1] || C.ne[0] != B.ne[1])
        {
            LogError("C (%d, %d, %d) is not compatible with A (%d, %d, %d) and B (%d, %d, %d).",
                C.ne[0], C.ne[1], C.ne[2], A.ne[0], A.ne[1], A.ne[2], B.ne[0], B.ne[1], B.ne[2]);
            return false;
        }
    }
    else
    {
        if (A.ne[2] != B.ne[2] || A.ne[0] != B.ne[1])
        {
            LogError("A and B are not compatible: %d vs. %d and %d vs. %d",
                A.ne[2], B.ne[2], A.ne[0], B.ne[1]);
            return false;
        }

        if (A.ne[2] != C.ne[2] || C.ne[1] != A.ne[1] || C.ne[0] != B.ne[0])
        {
            LogError("C (%d, %d, %d) is not compatible with A (%d, %d, %d) and B (%d, %d, %d).",
                C.ne[0], C.ne[1], C.ne[2], A.ne[0], A.ne[1], A.ne[2], B.ne[0], B.ne[1], B.ne[2]);
            return false;
        }
    }

    int batch_size = A.ne[2];
    int m = A.ne[1], k = A.ne[0];
    int n = is_b_column_major ? B.ne[1] : B.ne[0];

    bool is_fp16 = A.data_type == ElementType::F16 && B.data_type == ElementType::F16
        && C.data_type == ElementType::F16;
    bool is_fp32 = A.data_type == ElementType::F32 && B.data_type == ElementType::F32
        && C.data_type == ElementType::F32;

    if (is_fp16)
    {
        vector<const half*> ha(batch_size), hb(batch_size);
        vector<half*> hc(batch_size);
        for (int depth = 0; depth < batch_size; depth++)
        {
            ha[depth] = (const half*)A.RowData(0, depth);
            hb[depth] = (const half*)B.RowData(0, depth);
            hc[depth] = (half*)C.RowData(0, depth);
        }

        int bytes = sizeof(half*) * batch_size;
        half **da = (half**)heap.New(bytes);
        half **db = (half**)heap.New(bytes);
        half **dc = (half**)heap.New(bytes);
        CudaUtil::HostToDeviceMemcpy(da, ha.data(), bytes);
        CudaUtil::HostToDeviceMemcpy(db, hb.data(), bytes);
        CudaUtil::HostToDeviceMemcpy(dc, hc.data(), bytes);

        bool is_comp32 = false;
        ret = GemmExBatch_F16(m, n, k, (const half**)da, (const half**)db, dc,
            alpha, beta, is_b_column_major, is_comp32, batch_size, alg_id);
    }
    else if (is_fp32)
    {
        vector<const float*> ha(batch_size), hb(batch_size);
        vector<float*> hc(batch_size);
        for (int depth = 0; depth < batch_size; depth++)
        {
            ha[depth] = (const float*)A.RowData(0, depth);
            hb[depth] = (const float*)B.RowData(0, depth);
            hc[depth] = (float*)C.RowData(0, depth);
        }

        int bytes = sizeof(float*) * batch_size;
        float **da = (float**)heap.New(bytes);
        float **db = (float**)heap.New(bytes);
        float **dc = (float**)heap.New(bytes);
        CudaUtil::HostToDeviceMemcpy(da, ha.data(), bytes);
        CudaUtil::HostToDeviceMemcpy(db, hb.data(), bytes);
        CudaUtil::HostToDeviceMemcpy(dc, hc.data(), bytes);

        ret = GemmExBatch_F32(m, n, k, (const float**)da, (const float**)db, dc,
            alpha, beta, is_b_column_major, batch_size, alg_id);
    }
    else
    {
        LogError("Not implemented yet (should both be fp16 or fp32)!");
        ret = false;
    }

    return ret;
}

bool CublasEngine::GemmEx_F16(int m, int n, int k, const half *A, const half *B, half *C,
    float alpha, float beta, bool is_b_column_major, bool is_comp32,
    int alg_id)
{
    cublasStatus_t ret_code = CUBLAS_STATUS_SUCCESS;
    if (is_b_column_major)
    {
        if (is_comp32)
        {
            ret_code = GemmEx_Inner_FP16_Comp32(CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                A, k, B, k, C, m,
                &alpha, &beta, alg_id);
            //ret_code = GemmEx_Inner_FP16_Comp32(CUBLAS_OP_T, CUBLAS_OP_N,
            //    n, m, k,
            //    B, n, A, k, C, n,
            //    &alpha, &beta, alg_id);
        }
        else
        {
            ret_code = GemmEx_Inner_FP16_Comp16(CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                A, k, B, k, C, m,
                &alpha, &beta, alg_id);
            //ret_code = GemmEx_Inner_FP16_Comp16(CUBLAS_OP_N, CUBLAS_OP_T,
            //    n, m, k,
            //    B, n, A, m, C, n,
            //    &alpha, &beta, alg_id);
        }
    }
    else
    {
        if (is_comp32)
        {
            ret_code = GemmEx_Inner_FP16_Comp32(CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                B, n, A, k, C, n,
                &alpha, &beta, alg_id);
        }
        else
        {
            ret_code = GemmEx_Inner_FP16_Comp16(CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                B, n, A, k, C, n,
                &alpha, &beta, alg_id);
        }
    }

    cudaDeviceSynchronize();

    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasGemmEx", ret_code);
        return false;
    }
    return true;
}

bool CublasEngine::GemmExBatch_F16(int m, int n, int k, const half **array_a,
    const half **array_b, half **array_c, float alpha, float beta,
    bool is_b_column_major, bool is_comp32, int batch_num, int alg_id)
{
    cublasStatus_t ret_code = CUBLAS_STATUS_SUCCESS;
    if (is_b_column_major)
    {
        if (is_comp32)
        {
            ret_code = GemmExBatch_Inner_FP16_Comp32(CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                array_a, k, array_b, k, array_c, m,
                &alpha, &beta, batch_num, alg_id);
        }
        else
        {
            ret_code = GemmExBatch_Inner_FP16_Comp16(CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                array_a, k, array_b, k, array_c, m,
                &alpha, &beta, batch_num, alg_id);
        }
    }
    else
    {
        if (is_comp32)
        {
            ret_code = GemmExBatch_Inner_FP16_Comp32(CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                array_b, n, array_a, k, array_c, n,
                &alpha, &beta, batch_num, alg_id);
        }
        else
        {
            ret_code = GemmExBatch_Inner_FP16_Comp16(CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                array_b, n, array_a, k, array_c, n,
                &alpha, &beta, batch_num, alg_id);
        }
    }

    cudaDeviceSynchronize();

    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasGemmBatchedEx", ret_code);
        return false;
    }
    return true;
}

bool CublasEngine::GemmExBatch_F32(int m, int n, int k, const float **array_a,
    const float **array_b, float **array_c, float alpha, float beta,
    bool is_b_column_major, int batch_size, int alg_id)
{
    cublasStatus_t ret_code = CUBLAS_STATUS_SUCCESS;
    if (is_b_column_major)
    {
        ret_code = GemmExBatch_Inner_FP32(CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            array_a, k, array_b, k, array_c, m,
            &alpha, &beta, batch_size, alg_id);
    }
    else
    {
        ret_code = GemmExBatch_Inner_FP32(CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            array_b, n, array_a, k, array_c, n,
            &alpha, &beta, batch_size, alg_id);
    }

    cudaDeviceSynchronize();

    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasGemmEx", ret_code);
        return false;
    }
    return true;
}

bool CublasEngine::GemmEx_F32(int m, int n, int k, const float *A, const float *B, float *C,
    float alpha, float beta, bool is_b_column_major, int alg_id)
{
    cublasStatus_t ret_code = CUBLAS_STATUS_SUCCESS;
    if (is_b_column_major)
    {
        ret_code = GemmEx_Inner_FP32(CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            A, k, B, k, C, m,
            &alpha, &beta, alg_id);
    }
    else
    {
        ret_code = GemmEx_Inner_FP32(CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            B, n, A, k, C, n,
            &alpha, &beta, alg_id);
    }

    cudaDeviceSynchronize();

    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasGemmEx", ret_code);
        return false;
    }
    return true;
}

bool CublasEngine::Gemv(DeviceTensor &Y, const DeviceTensor &X, const DeviceTensor &A,
    float alpha, float beta, bool is_b_column_major)
{
    bool ret = true;
    int m = A.Rows(), n = A.Columns();

    bool is_fp16 = A.data_type == ElementType::F16 && X.data_type == ElementType::F16
        && Y.data_type == ElementType::F16;
    bool is_fp32 = A.data_type == ElementType::F32 && X.data_type == ElementType::F32
        && Y.data_type == ElementType::F32;

    if (is_fp16)
    {
        const half *a_data = A.data_f16();
        const half *x_data = X.data_f16();
        half *y_data = Y.data_f16();
        //bool is_comp32 = false;
        ret = Gemv_F16(m, n, x_data, a_data, y_data, alpha, beta,
            is_b_column_major);
    }
    else if (is_fp32)
    {
        const float *a_data = A.data_f32();
        const float *x_data = X.data_f32();
        float *y_data = Y.data_f32();
        ret = Gemv_F32(m, n, x_data, a_data, y_data, alpha, beta,
            is_b_column_major);
    }
    else
    {
        ret = false;
    }

    return ret;
}

bool CublasEngine::Gemv_F16(int m, int n, const half *X, const half *A, half *Y,
    float alpha, float beta, bool is_b_column_major)
{
    (void)alpha; (void)beta; (void)m; (void)n; (void)X; (void)A; (void)Y;
    cublasStatus_t ret_code = CUBLAS_STATUS_SUCCESS;
    if (is_b_column_major)
    {
    }
    else
    {
        //int batch_size = 1;
        //ret_code = cublasHSHgemvBatched(handle_, CUBLAS_OP_N, n, m,
        //    &alpha, &A, n, &X, 1, &beta, &Y, 1, batch_size);
    }

    cudaDeviceSynchronize();

    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasGemmEx", ret_code);
        return false;
    }
    return true;
}

bool CublasEngine::Gemv_F32(int m, int n, const float *X, const float *A, float *Y,
    float alpha, float beta, bool is_b_column_major)
{
    cublasStatus_t ret_code = CUBLAS_STATUS_SUCCESS;
    if (is_b_column_major)
    {
    }
    else
    {
        ret_code = cublasSgemv(handle_, CUBLAS_OP_N, n, m,
            &alpha, A, n, X, 1, &beta, Y, 1);
    }

    cudaDeviceSynchronize();

    if (ret_code != CUBLAS_STATUS_SUCCESS) {
        LogError("cuBLAS error %d in calling cublasGemmEx", ret_code);
        return false;
    }
    return true;
}

cublasStatus_t CublasEngine::GemmEx_Inner_FP16_Comp16(cublasOperation_t trans_a, cublasOperation_t trans_b,
    int m, int n, int k, const half *A, int lda, const half *B, int ldb, half *C, int ldc,
    const float *alpha, const float *beta, int alg_id)
{
    cudaDataType_t AType = CUDA_R_16F;
    cudaDataType_t BType = CUDA_R_16F;
    cudaDataType_t CType = CUDA_R_16F;
    cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F_FAST_16F; //CUBLAS_COMPUTE_16F;
    //cudaDataType_t ComputeType = CUDA_R_16F;
    cublasStatus_t status = cublasGemmEx(handle_,
        trans_a, trans_b, m, n, k,
        alpha, A, AType, lda, B, BType, ldb,
        beta, C, CType, ldc,
        ComputeType, (cublasGemmAlgo_t)alg_id);
    return status;
}

cublasStatus_t CublasEngine::GemmEx_Inner_FP16_Comp32(cublasOperation_t trans_a, cublasOperation_t trans_b,
    int m, int n, int k, const half *A, int lda, const half *B, int ldb, half *C, int ldc,
    const float *alpha, const float *beta, int alg_id)
{
    cudaDataType_t AType = CUDA_R_16F;
    cudaDataType_t BType = CUDA_R_16F;
    cudaDataType_t CType = CUDA_R_16F;
    cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
    cublasStatus_t status = cublasGemmEx(handle_,
        trans_a, trans_b, m, n, k,
        alpha, A, AType, lda, B, BType, ldb,
        beta, C, CType, ldc,
        ComputeType, (cublasGemmAlgo_t)alg_id);
    return status;
}

cublasStatus_t CublasEngine::GemmEx_Inner_FP32(cublasOperation_t trans_a, cublasOperation_t trans_b,
    int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *C, int ldc,
    const float *alpha, const float *beta, int alg_id)
{
    cudaDataType_t AType = CUDA_R_32F;
    cudaDataType_t BType = CUDA_R_32F;
    cudaDataType_t CType = CUDA_R_32F;
    cudaDataType_t ComputeType = CUDA_R_32F;
    cublasStatus_t status = cublasGemmEx(handle_,
        trans_a, trans_b, m, n, k,
        alpha, A, AType, lda, B, BType, ldb,
        beta, C, CType, ldc,
        ComputeType, (cublasGemmAlgo_t)alg_id);
    return status;
}

cublasStatus_t CublasEngine::GemmExBatch_Inner_FP16_Comp16(cublasOperation_t trans_a,
    cublasOperation_t trans_b, int m, int n, int k, const half **array_a, int lda,
    const half **array_b, int ldb, half **array_c, int ldc,
    const float *alpha, const float *beta, int batch_size, int alg_id)
{
    cudaDataType_t AType = CUDA_R_16F;
    cudaDataType_t BType = CUDA_R_16F;
    cudaDataType_t CType = CUDA_R_16F;
    cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F_FAST_16F; //CUBLAS_COMPUTE_16F;
    cublasStatus_t status = cublasGemmBatchedEx(handle_,
        trans_a, trans_b, m, n, k,
        alpha, (const void**)array_a, AType, lda,
        (const void**)array_b, BType, ldb,
        beta, (void**)array_c, CType, ldc,
        batch_size, ComputeType,
        (cublasGemmAlgo_t)alg_id);
    return status;
}

cublasStatus_t CublasEngine::GemmExBatch_Inner_FP16_Comp32(cublasOperation_t trans_a,
    cublasOperation_t trans_b, int m, int n, int k, const half **array_a, int lda,
    const half **array_b, int ldb, half **array_c, int ldc,
    const float *alpha, const float *beta, int batch_size, int alg_id)
{
    cudaDataType_t AType = CUDA_R_16F;
    cudaDataType_t BType = CUDA_R_16F;
    cudaDataType_t CType = CUDA_R_16F;
    cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F;
    cublasStatus_t status = cublasGemmBatchedEx(handle_,
        trans_a, trans_b, m, n, k,
        alpha, (const void**)array_a, AType, lda,
        (const void**)array_b, BType, ldb,
        beta, (void**)array_c, CType, ldc,
        batch_size, ComputeType,
        (cublasGemmAlgo_t)alg_id);
    return status;
}

cublasStatus_t CublasEngine::GemmExBatch_Inner_FP32(cublasOperation_t trans_a,
    cublasOperation_t trans_b, int m, int n, int k, const float **array_a, int lda,
    const float **array_b, int ldb, float **array_c, int ldc,
    const float *alpha, const float *beta, int batch_size, int alg_id)
{
    cudaDataType_t AType = CUDA_R_32F;
    cudaDataType_t BType = CUDA_R_32F;
    cudaDataType_t CType = CUDA_R_32F;
    cudaDataType_t ComputeType = CUDA_R_32F;
    cublasStatus_t status = cublasGemmBatchedEx(handle_,
        trans_a, trans_b, m, n, k,
        alpha, (const void**)array_a, AType, lda,
        (const void**)array_b, BType, ldb,
        beta, (void**)array_c, CType, ldc,
        batch_size, ComputeType,
        (cublasGemmAlgo_t)alg_id);
    return status;
}

INFER_FLOW_END
