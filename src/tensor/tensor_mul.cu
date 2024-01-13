#include "tensor_mul.h"
#include <algorithm>
#include "kernels/gemm.h"
#include "kernels/gemm_wmma.h"
#include "kernels/gemv.h"
#include "tensor_opr.h"
#include "host_tensor_opr.h"
#ifdef USE_CUTLASS
#   include "sgemm_cutlass.h"
#   include "cutlass/cutlass.h"
#   include "cutlass/gemm/device/gemm.h"
#endif

INFER_FLOW_BEGIN

//static
bool TensorMul::Gemm(DeviceTensor &C, const DeviceTensor &A, const DeviceTensor &B,
    float alpha, float beta, bool is_b_column_major, bool is_sub_level, MatrixMulAlg alg)
{
    bool ret = true;
    switch (alg)
    {
        case MatrixMulAlg::Alg1:
            ret = Gemm_Alg1(C, A, B, alpha, beta, is_b_column_major, is_sub_level);
            break;
        case MatrixMulAlg::Alg2:
            ret = Gemm_Alg2(C, A, B, alpha, beta, is_b_column_major, is_sub_level);
            break;
        //case MatrixMulAlg::Alg3:
        //    ret = Gemm_Alg3(C, A, B, alpha, beta, is_b_column_major, is_sub_level);
        //    break;
        case MatrixMulAlg::Cutlass:
#ifdef USE_CUTLASS
            ret = Gemm_Cutlass(C, A, B, alpha, beta, is_b_column_major, is_sub_level);
#else
            LogError("Cutlass is not enabled.");
#endif //USE_CUTLASS
            break;
        case MatrixMulAlg::Bruce:
            ret = Gemm_Bruce(C, A, B, alpha, beta, is_b_column_major, is_sub_level);
            break;
        default:
            ret = Gemm_Alg2(C, A, B, alpha, beta, is_b_column_major, is_sub_level);
            break;
    }

    return ret;
}

//static
bool TensorMul::Gemv(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, bool is_b_column_major, VectorMatrixMulAlg alg)
{
    bool ret = true;
    int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();

    bool is_compatible = HostTensorOpr::IsCompatible_MulMat(a_rows, a_cols,
        b_rows, b_cols, c_rows, c_cols, is_b_column_major);
    if (!is_compatible) {
        return false;
    }

    if (C.data == nullptr || c_rows * c_cols != C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    if (A.ne[1] != 1 || A.ne[2] != 1 || C.ne[1] != 1 || C.ne[2] != 1)
    {
        LogError("Tensor A and C should be vectors");
        return false;
    }

    switch (alg)
    {
        case VectorMatrixMulAlg::Alg1:
            ret = Gemv_Alg1(C, A, B, is_b_column_major);
            break;
        case VectorMatrixMulAlg::Alg2:
            ret = Gemv_Alg1(C, A, B, is_b_column_major);
            break;
        case VectorMatrixMulAlg::Cublas:
        default:
            //ret = Gemv_Cublas(C, A, B, is_b_column_major);
            LogError("VectorMatrixMul: Algorithm %d is not impelemtned.", (int)alg);
            ret = false;
            break;
    }

    return ret;
}

//static
//  U: An auxiliary tensor
bool TensorMul::Gemv_AX(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X,
    VectorMatrixMulAlg alg, DeviceTensor *U, const DeviceTensor *bias)
{
    bool ret = true;
    int M = A.Rows(), N = A.Columns();
    if (X.size != N) {
        LogError("A and X are not compatible");
        return false;
    }
    if (Y.size != M) {
        LogError("A and Y are not compatible");
        return false;
    }

    switch (alg)
    {
        case VectorMatrixMulAlg::Alg3:
            ret = Gemv_AX_Alg3(Y, A, X, bias);
            break;
        case VectorMatrixMulAlg::Alg4:
            if (U == nullptr) {
                LogError("Null auxiliary tensor");
                return false;
            }
            ret = Gemv_AX_Alg4(Y, A, X, *U);
            break;
        default:
            LogError("Gemv_AX: Algorithm %d is not impelemtned.", (int)alg);
            ret = false;
            break;
    }

    return ret;
}

//static
bool TensorMul::Gemv_XA(DeviceTensor &Y, const DeviceTensor &X, const DeviceTensor &A,
    VectorMatrixMulAlg alg)
{
    bool ret = true;
    int M = A.Rows(), N = A.Columns();
    if (X.size != M) {
        LogError("A and X are not compatible");
        return false;
    }
    if (Y.size != N) {
        LogError("A and Y are not compatible");
        return false;
    }

    switch (alg)
    {
        case VectorMatrixMulAlg::Alg3:
            ret = Gemv_XA_Alg3(Y, X, A);
            break;
        default:
            LogError("Gemv_XA: Algorithm %d is not impelemtned.", (int)alg);
            ret = false;
            break;
    }

    return ret;
}

//static
bool TensorMul::Gemm_Alg1(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, float alpha, float beta,
    bool is_b_column_major, bool is_sub_level)
{
    (void)is_sub_level;
    int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();

    bool is_compatible = HostTensorOpr::IsCompatible_MulMat(a_rows, a_cols,
        b_rows, b_cols, c_rows, c_cols, is_b_column_major);
    if (!is_compatible) {
        return false;
    }

    if (C.data == nullptr || c_rows * c_cols != C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    int M = a_rows, K = a_cols;
    int N = is_b_column_major ? b_rows : b_cols;
    dim3 block(16, 16); //blocks of the target C
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        if (is_b_column_major)
        { //(M, K) * (N, K) = (M, N)
            MulMat_Y1_RowMajor_Kernel<half, half, half><<<grid, block>>>(
                M, N, K, alpha, a_data, b_data, beta, c_data);
        }
        else
        { //(M, K) * (K, N) = (M, N)
            MulMat_X1_RowMajor_Kernel<half, half, half><<<grid, block>>>(
                M, N, K, alpha, a_data, b_data, beta, c_data);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();

        if (is_b_column_major)
        { //(M, K) * (N, K) = (M, N)
            MulMat_Y1_RowMajor_Kernel<float, float, float><<<grid, block>>>(
                M, N, K, alpha, a_data, b_data, beta, c_data);
        }
        else
        { //(M, K) * (K, N) = (M, N)
            MulMat_X1_RowMajor_Kernel<float, float, float><<<grid, block>>>(
                M, N, K, alpha, a_data, b_data, beta, c_data);
        }
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemm_Alg1");
    return ret;
}

//static
bool TensorMul::Gemm_Alg2(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, float alpha, float beta,
    bool is_b_column_major, bool is_sub_level)
{
    int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();

    if (C.data == nullptr || c_rows * c_cols != C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    int r = 1;
    bool is_compatible = true;
    if (is_sub_level)
    {
        //r = A.ne[2] / B.ne[2];
        if (is_b_column_major)
        {
            if (A.ne[2] != r * B.ne[2] || A.ne[0] != B.ne[0])
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
            if (A.ne[2] != r * B.ne[2] || A.ne[0] != B.ne[1])
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
    }
    else
    {
        is_compatible = HostTensorOpr::IsCompatible_MulMat(a_rows, a_cols,
            b_rows, b_cols, c_rows, c_cols, is_b_column_major);
        if (!is_compatible) {
            LogError("Gemm_Alg2: C is not compatible with A and B.");
            return false;
        }
    }

    int depth = is_b_column_major && is_sub_level ? A.ne[2] : 1;
    //if (depth == 1 && A.ne[1] == 1 && !is_sub_level)
    //{
    //    bool ret = VectorMatrixMul_Alg2(C, A, B, is_b_column_major);
    //    return ret;
    //}

    int M = a_rows / depth, K = a_cols;
    int N = is_b_column_major ? b_rows * r / depth  : b_cols;
    bool a_is_zy_data = A.is_zy_data, b_is_zy_data = B.is_zy_data;

    int block_size = 8;
    dim3 block(block_size, block_size, 1);
    //dim3 grid((RB.width + block.x - 1) / block.x, (RA.height + block.y - 1) / block.y);
    dim3 grid((N + block_size - 1) / block_size, (M + block_size - 1) / block_size, A.ne[2]);

    //cout << "block: " << block << ", grid: " << grid << endl;

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        Gemm_Alg2_Kernel<half, half, half><<<grid, block>>>(M, N, K, depth,
            alpha, a_data, b_data, beta, c_data, is_b_column_major,
            a_is_zy_data, b_is_zy_data, r);
    }
    else
    {
        const float *a_data = (const float*)A.data;
        const float *b_data = (const float*)B.data;
        float *c_data = (float*)C.data;

        //Gemm_Alg2_Kernel<<<grid, block>>>(RC, RA, RB, alpha, beta);
        Gemm_Alg2_Kernel<float, float, float><<<grid, block>>>(M, N, K, depth,
            alpha, a_data, b_data, beta, c_data, is_b_column_major,
            a_is_zy_data, b_is_zy_data, r);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemm_Alg2");
    return ret;
}

#ifdef USE_CUTLASS
//static
bool TensorMul::Gemm_Cutlass(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, float alpha, float beta,
    bool is_b_column_major, bool is_sub_level)
{
    int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();

    bool is_compatible = IsCompatible_MulMat(a_rows, a_cols,
        b_rows, b_cols, c_rows, c_cols, is_b_column_major);
    if (!is_compatible) {
        return false;
    }

    if (C.data == nullptr || c_rows * c_cols != C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    int M = a_rows, N = b_cols, K = a_cols;

    bool ret = true;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        if (is_b_column_major)
        {
            //int lda = K, ldb = N, ldc = N;
            //cutlass::gemm::GemmCoord coord_b(N, K, 1);
            //ret = CutlassSgemm_RowColMajor<half, half, half>(M, N, K,
            //    alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
        else
        {
            int lda = K, ldb = N, ldc = N;
            ret = CutlassSgemm_RowMajor_Half(M, N, K,
                alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();

        if (is_b_column_major)
        {
            int lda = K, ldb = N, ldc = N;
            //cutlass::gemm::GemmCoord coord_b(N, K, 1);
            ret = CutlassSgemm_RowColMajor(M, N, K,
                alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
        else
        {
            int lda = K, ldb = N, ldc = N;
            ret = CutlassSgemm_RowMajor_Float(M, N, K,
                alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
        }
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    return ret;
}
#endif //USE_CUTLASS

//static
bool TensorMul::Gemm_Bruce(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, float alpha, float beta,
    bool is_b_column_major, bool is_sub_level)
{
    (void)is_sub_level;
    int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();

    bool is_compatible = HostTensorOpr::IsCompatible_MulMat(a_rows, a_cols,
        b_rows, b_cols, c_rows, c_cols, is_b_column_major);
    if (!is_compatible) {
        return false;
    }

    if (C.data == nullptr || c_rows * c_cols != C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    int M = a_rows, K = a_cols;
    int N = is_b_column_major ? b_rows : b_cols;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));
    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half),
        BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    cudaFuncSetAttribute(GemmHalf_Bruce_Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        if (is_b_column_major)
        { //(M, K) * (N, K) = (M, N)
            GemmHalf_Bruce_Kernel<<<grid, block, smem_max_size>>>(a_data, b_data, c_data, M, N, K);
        }
        else
        { //(M, K) * (K, N) = (M, N)
            LogWarning("is_b_column_major == false is NOT implemented for Gemm_Bruce");
            return false;
        }
    }
    else
    {
        LogWarning("Only FP16 is supported by Gemm_Bruce.");
        return false;
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemm_Bruce");
    return ret;
}

//static
bool TensorMul::GemmSparse(DeviceTensor &C, const DeviceTensor &A,
    const DeviceSparseMatrix &B, float alpha, float beta)
{
    int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();

    if (C.data == nullptr || c_rows * c_cols != C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    bool is_compatible = true;
    bool is_b_column_major = true;
    is_compatible = HostTensorOpr::IsCompatible_MulMat(a_rows, a_cols,
        b_rows, b_cols, c_rows, c_cols, is_b_column_major);
    if (!is_compatible) {
        LogError("GemmSparse: C is not compatible with A and B.");
        return false;
    }

    //B is column-major
    int M = a_rows, K = a_cols;
    int N = b_rows;
    int b_size = B.Size();
    bool a_is_zy_data = A.is_zy_data;

    dim3 block, grid;
    block.y = min(8, M);
    block.x = 128 / block.y;
    grid.y = (M + block.y - 1) / block.y;
    grid.x = (N + block.x - 1) / block.x;

    const SparseMatrixCell *b_cells = B.Cells();
    const int *row_offset_array = B.RowOffsetArray();
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        half *c_data = C.data_f16();

        GemmSparse_Alg1_Kernel<half, half><<<grid, block>>>(c_data, a_data, b_cells,
            row_offset_array, M, N, K, b_size, alpha, beta, a_is_zy_data);
    }
    else
    {
        const float *a_data = (const float*)A.data;
        float *c_data = (float*)C.data;

        GemmSparse_Alg1_Kernel<float, float><<<grid, block>>>(c_data, a_data, b_cells,
            row_offset_array, M, N, K, b_size, alpha, beta, a_is_zy_data);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemm_Alg2");
    return ret;
}

bool TensorMul::Gemv_Alg1(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, bool is_b_column_major)
{
    (void)is_b_column_major;
    //int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    //int c_rows = C.Rows(), c_cols = C.Columns();

    int K = b_rows, N = b_cols;
    int n_tile_len = min(4, Inferflow_MaxNTile);
    int thread_per_block = 256; //K >= 256 ? 256 : (K >= 64 ? 64 : (K >= 16 ? 16 : 4));
    dim3 block_grid(1, (N + n_tile_len - 1) / n_tile_len);
    int k_tile_len = (K + thread_per_block - 1) / thread_per_block;

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();
        Gemv_Alg1_Kernel<half, half, half><<<block_grid, thread_per_block>>>(
            c_data, a_data, b_data, K, N, k_tile_len, n_tile_len);
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();
        Gemv_Alg1_Kernel<float, float, float><<<block_grid, thread_per_block>>>(
            c_data, a_data, b_data, K, N, k_tile_len, n_tile_len);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemv_Alg1");
    return ret;
}

bool TensorMul::Gemv_Alg2(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, bool is_b_column_major)
{
    (void)is_b_column_major;
    //int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    //int c_rows = C.Rows(), c_cols = C.Columns();

    bool ret = C.AssignZero();
    if (!ret) {
        return false;
    }

    int K = b_rows, N = b_cols;
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, 1);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();
        Gemv_Alg2_Kernel<half, half, half><<<grid, block>>>(c_data, a_data, b_data, K, N);
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();
        Gemv_Alg2_Kernel<float, float, float><<<grid, block>>>(c_data, a_data, b_data, K, N);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    ret = CudaUtil::CheckReturnCode(status, "Gemv_Alg2");
    return ret;
}

// Y = X*A
bool TensorMul::Gemv_XA_Alg3(DeviceTensor &Y, const DeviceTensor &X,
    const DeviceTensor &A)
{
    int M = A.Rows(), N = A.Columns();
    if (X.size != M) {
        LogError("A and X are not compatible");
        return false;
    }
    if (Y.size != N) {
        LogError("A and Y are not compatible");
        return false;
    }

    const int warp_size = 32;
    //int stride = 4;
    int x_stride = N % 16 == 0 && N >= 1024 ? 16 : (N % 8 == 0 && N >= 512 ? 8 : 4);
    dim3 block(4, warp_size), grid(1, 1);
    grid.x = (N / (x_stride / 4) + block.x - 1) / block.x;
    //LogKeyInfo("M: %d, N: %d, block: (%d, %d), grid: (%d, %d)",
    //    M, N, block.x, block.y, grid.x, grid.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *x_data = X.data_f16();
        half *y_data = Y.data_f16();
        switch (x_stride)
	{
        case 16:
            GemvHalf_XA_Alg3_S16_Kernel<<<grid, block>>>(y_data, x_data, a_data, M, N);
            break;
        case 8:
            GemvHalf_XA_Alg3_S8_Kernel<<<grid, block>>>(y_data, x_data, a_data, M, N);
            break;
	case 4:
	default:
            GemvHalf_XA_Alg3_Kernel<<<grid, block>>>(y_data, x_data, a_data, M, N);
            break;
        }
    }
    else
    {
        /*const float *a_data = A.data_f32();
        const float *x_data = X.data_f32();
        float *y_data = Y.data_f32();
        Sgemv_Alg3_AX128_Kernel<<<grid, block>>>(y_data, a_data, x_data, M, N);*/
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemv_Alg3");
    return ret;
}

//static
bool TensorMul::Gemv_AX_QuantHalf(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X)
{
    bool ret = true;
    switch (A.data_type)
    {
    case ElementType::Q8_GL:
        ret = Gemv_AX_Q8GL(Y, A, X);
        break;
    case ElementType::Q8_B32T1:
        ret = Gemv_AX_Q8_B32T1(Y, A, X);
        break;
    case ElementType::Q8_B32T2:
        ret = Gemv_AX_Q8_B32T2(Y, A, X);
        break;
    case ElementType::Q6_B64T1:
        ret = Gemv_AX_Q6_B64T1(Y, A, X);
        break;
    case ElementType::Q5:
        ret = Gemv_AX_Q5(Y, A, X);
        break;
    case ElementType::Q4_B16:
        ret = Gemv_AX_Q4B16(Y, A, X);
        break;
    case ElementType::Q4_B32T1A:
    case ElementType::Q4_B32T1B:
        ret = Gemv_AX_Q4_B32T1(Y, A, X);
        break;
    case ElementType::Q3H_B64T1:
        ret = Gemv_AX_Q3H_B64T1(Y, A, X);
        break;
    case ElementType::Q3_B32T1A:
    case ElementType::Q3_B32T1B:
        ret = Gemv_AX_Q3_B32T1(Y, A, X);
        break;
    case ElementType::Q2_B32T1A:
    case ElementType::Q2_B32T1B:
        ret = Gemv_AX_Q2_B32T1(Y, A, X);
        break;
    default:
        LogError("Gemv_AX_QuantHalf: Element type %d is not supported so far.", A.data_type);
        ret = false;
        break;
    }

    ret = CudaUtil::DeviceSynchronize("Gemv_AX_QuantHalf");
    return ret;
}

//static
bool TensorMul::Gemv_AX_QuantQ8(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X)
{
    bool ret = true;
    switch (A.data_type)
    {
    case ElementType::Q8_B32T1:
        //ret = Gemv_AX_Q8_B32T1(Y, A, X);
        break;
    case ElementType::Q8_B32T2:
        ret = Gemv_AX8_Q8_B32T2(Y, A, X);
        break;
    case ElementType::Q6_B64T1:
        //ret = Gemv_AX8_Q6_B64T1(Y, A, X);
        break;
    case ElementType::Q5:
        //ret = Gemv_AX8_Q8Q5(Y, A, X);
        break;
    case ElementType::Q4_B16:
        //ret = Gemv_AX8_Q4_B16(Y, A, X);
        break;
    case ElementType::Q4_B32T1A:
    case ElementType::Q4_B32T1B:
        //ret = Gemv_AX8_Q4_B32T1(Y, A, X);
        break;
    case ElementType::Q3H_B64T1:
        //ret = Gemv_AX8_Q3H_B64T1(Y, A, X);
        break;
    case ElementType::Q3_B32T1A:
    case ElementType::Q3_B32T1B:
        //ret = Gemv_AX8_Q3_B32T1(Y, A, X);
        break;
    case ElementType::Q2_B32T1A:
    case ElementType::Q2_B32T1B:
        //ret = Gemv_AX8_Q2_B32T1(Y, A, X);
    default:
        LogError("Gemv_AX_QuantQ8: Element type %d is not supported so far.", A.data_type);
        ret = false;
        break;
    }

    ret = CudaUtil::DeviceSynchronize("Gemv_AX_QuantQ8");
    return ret;
}

// Y = A*X
bool TensorMul::Gemv_AX_Alg3(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X, const DeviceTensor *bias)
{
    int cy = A.Rows(), cx = A.Columns();
    if (X.size != cx) {
        LogError("A and X are not compatible");
        return false;
    }
    if (Y.size != cy) {
        LogError("A and Y are not compatible");
        return false;
    }

    if (X.IsQuantized())
    {
        bool ret = Gemv_AX_QuantQ8(Y, A, X);
        return ret;
    }
    else if (A.IsQuantized())
    {
        bool ret = Gemv_AX_QuantHalf(Y, A, X);
        return ret;
    }

    const int warp_size = 32;
    dim3 block(warp_size, 4);
    //dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    dim3 grid(1, (cy + block.y - 1) / block.y);

    if (A.data_type == ElementType::F32)
    {
        if (cx % 8 != 0) {
            LogError("N (%d) should be a multiple of 8", cx);
            return false;
        }

        block.x = cx % 512 == 0 ? 128 : (cx % 256 == 0 ? 64 : 32);
        block.y = 128 / block.x;
        grid.x = 1;
        grid.y = (cy + block.y - 1) / block.y;

        const float *a_data = A.data_f32();
        const float *x_data = X.data_f32();
        float *y_data = Y.data_f32();
        const float *bias_data = bias == nullptr ? nullptr : bias->data_f32();
        Sgemv_Alg3_AX128_Kernel<<<grid, block>>>(y_data, a_data, x_data, bias_data, cy, cx);
    }
    else if (A.data_type == ElementType::F16)
    { //FP16
        if (cx % 8 != 0) {
            LogError("N (%d) should be a multiple of 8", cx);
            return false;
        }

        block.x = cx % 1024 == 0 ? 128 : (cx % 512 == 0 ? 64 : 32);
        block.y = 128 / block.x;
        grid.x = 1; //(cx + block.x - 1) / block.x;
        grid.y = (cy + block.y - 1) / block.y;
        if (grid.y > 0xFFFF)
        {
            grid.z = (grid.y + 4095) / 4096;
            grid.y = 4096;
        }

        const half *a_data = A.data_f16();
        const half *x_data = X.data_f16();
        half *y_data = Y.data_f16();
        const half *bias_data = bias == nullptr ? nullptr : bias->data_f16();
        GemvHalf_AX_Alg3_Kernel<<<grid, block>>>(y_data, a_data, x_data, bias_data, cy, cx);
    }
    else
    {
        LogError("Gemv_AX_Alg3: Element type %d is not supported now.", A.data_type);
        return false;
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemv_AX_Alg3");
    return ret;
}

bool TensorMul::Gemv_AX_Q8GL(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X)
{
    (void)Y; (void)A; (void)X;
    /*int cy = A.Rows(), cx = A.Columns();
    if (cx % 32 != 0) {
        LogError("N (%d) should be a multiple of 32", cx);
        return false;
    }

    dim3 block, grid;
    block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    //const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    (void)x_data; (void)y_data;
    //GemvHalf_AX_Q8GL_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);*/
    return true;
}

//static
bool TensorMul::Gemv_AX_Q8_B32T1(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    bool ret = GemvCheckN(cx, 1024, 32); //1024 = 32 * 32 (i.e., warp size * Q4B32_CAPACITY)
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q8_B32T1_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

//static
bool TensorMul::Gemv_AX_Q8_B32T2(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    bool ret = GemvCheckN(cx, 1024, 32); //1024 = 32 * 32 (i.e., warp size * Q4B32_CAPACITY)
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q8_B32T2_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

bool TensorMul::Gemv_AX_Q6_B64T1(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    //2048 = 32 * 64 (i.e., warp size * Q6_B64_CAPACITY)
    bool ret = GemvCheckN(cx, 2048, Q6_B64_CAPACITY);
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 8192 == 0 ? 128 : (cx % 4096 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q6_B64T1_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

//static
bool TensorMul::Gemv_AX_Q5(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    bool ret = GemvCheckN(cx, 1024, 32); //1024 = 32 * 32 (i.e., warp size * Q4B32_CAPACITY)
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q5_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

bool TensorMul::Gemv_AX_Q4B16(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    if (cx % 512 != 0) { //512 = 32 * 16, where 32 is the warp size, and 16 is Q4B16_CAPACITY
        LogError("N (%d) should be a multiple of 1024", cx);
        return false;
    }

    dim3 block, grid;
    block.x = cx % 2048 == 0 ? 128 : (cx % 1024 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q4B16_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

bool TensorMul::Gemv_AX_Q4_B32T1(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    bool ret = GemvCheckN(cx, 1024, 32); //1024 = 32 * 32 (i.e., warp size * Q4B32_CAPACITY)
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q4_B32T1_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

bool TensorMul::Gemv_AX_Q3H_B64T1(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    //2048 = 32 * 64 (i.e., warp size * Q3H_B64_CAPACITY)
    bool ret = GemvCheckN(cx, 2048, Q3H_B64_CAPACITY);
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 8192 == 0 ? 128 : (cx % 4096 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q3H_B64T1_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

bool TensorMul::Gemv_AX_Q3_B32T1(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    bool ret = GemvCheckN(cx, 1024, 32); //1024 = 32 * 32 (i.e., warp size * Q3B32_CAPACITY)
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q3_B32T1_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

bool TensorMul::Gemv_AX_Q2_B32T1(DeviceTensor &Y, const DeviceTensor &A, const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    bool ret = GemvCheckN(cx, 1024, 32); //1024 = 32 * 32 (i.e., warp size * Q4B32_CAPACITY)
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const half *x_data = X.data_f16();
    half *y_data = Y.data_f16();
    GemvHalf_AX_Q2_B32T1_Kernel<<<grid, block>>>(y_data, a_data, x_data, cy, cx);
    return true;
}

//static
bool TensorMul::Gemv_AX8_Q8_B32T2(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X)
{
    int cy = A.Rows(), cx = A.Columns();
    bool ret = GemvCheckN(cx, 32, 32);
    Macro_RetFalseIf(!ret);

    dim3 block, grid;
    //block.x = cx % 4096 == 0 ? 128 : (cx % 2048 == 0 ? 64 : 32);
    block.x = 32;
    block.y = 128 / block.x;
    grid.x = 1;
    grid.y = (cy + block.y - 1) / block.y;

    const uint8_t *a_data = (const uint8_t*)A.data;
    const uint8_t *x_data = (const uint8_t*)X.data;
    half *y_data = Y.data_f16();
    Gemv_AX8_Q8_B32T2_Kernel<<<grid, block>>>(y_data, a_data, x_data, cx, cy);
    return true;
}

// Y = A*X
bool TensorMul::Gemv_AX_Alg4(DeviceTensor &Y, const DeviceTensor &A,
    const DeviceTensor &X, DeviceTensor &U)
{
    int M = A.Rows(), N = A.Columns();
    if (X.size != N) {
        LogError("A and X are not compatible");
        return false;
    }
    if (Y.size != M) {
        LogError("A and Y are not compatible");
        return false;
    }

    bool is_f16 = A.data_type == ElementType::F16;
    const int thread_num_per_block = 128;
    //const int warp_size = 32;
    const int stride = is_f16 ? 8 : 4;
    dim3 block, grid;
    if (is_f16) {
        block.x = N >= 1024 ? 128 : (N >= 512 ? 64 : 32);
    }
    else {
        block.x = N >= 512 ? 128 : (N >= 256 ? 64 : 32);
    }
    //block.x = 32;
    block.y = thread_num_per_block / block.x;
    grid.x = (N / stride + block.x - 1) / block.x;
    grid.y = (M + block.y - 1) / block.y;
    //LogKeyInfo("block: (%d, %d), grid: (%d, %d)", block.x, block.y, grid.x, grid.y);

    DeviceTensor aux_tensor(false);
    aux_tensor.data_type = Y.data_type;
    aux_tensor.data = U.data;

    if (is_f16)
    {
        const half *a_data = A.data_f16();
        const half *x_data = X.data_f16();

        int aux_cx = max(1, N / stride / thread_num_per_block);
        aux_tensor.SetStructure(aux_cx, M);
        half *aux_data = aux_tensor.data_f16();
        GemvHalf_AX_Alg4_Kernel<<<grid, block>>>(aux_data, a_data, x_data, M, N);
    }
    else //f32
    {
        //const float *a_data = A.data_f32();
        //const float *x_data = X.data_f32();
        //float *y_data = Y.data_f32();
        //Gemv_AX_Alg4_Kernel<<<grid, block>>>(y_data, a_data, x_data, M, N);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    auto status = cudaGetLastError();
    bool ret = CudaUtil::CheckReturnCode(status, "Gemv_AX_Alg4");
    Macro_RetFalseIf(!ret);

    ret = true; //TensorOpr::SumByRow(Y, aux_tensor);
    return ret;
}

//static
bool TensorMul::GemvCheckN(int cx, int min_cx, int m)
{
    if (cx < min_cx)
    {
        LogError("N (%d) should be at least %d", cx, min_cx);
        return false;
    }

    if (cx % m != 0) {
        LogError("N (%d) should be a multiple of %d", cx, m);
        return false;
    }

    return true;
}

INFER_FLOW_END
