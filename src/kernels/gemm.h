#pragma once

#include "kernels_common.h"
#include "tensor/tensor_common.h"

namespace inferflow
{

template <typename AType, typename BType, typename CType>
__global__ void MulMat_X1_RowMajor_Kernel(int M, int N, int K,
    float alpha, AType const *A, BType const *B,
    float beta, CType *C)
{
    //ci: row index of C; cj: column index of C
    int ci = threadIdx.x + blockIdx.x * blockDim.x;
    int cj = threadIdx.y + blockIdx.y * blockDim.y;

    if (ci < M && cj < N)
    {
        double accumulator = 0;
        for (int kk = 0; kk < K; ++kk) {
            accumulator += (float)A[kk + ci * K] * (float)B[cj + kk * N];
        }

        float new_value = alpha * (float)accumulator + beta * (float)C[cj + ci * N];
        C[cj + ci * N] = new_value;

        //if (ci == 0 && (cj == 625 || cj == 626 || cj == 627 || cj == 0)) {
        //    half v_half = __float2half(new_value);
        //    float new_value2 = v_half;
        //    printf("beta: %f, cj: %d, accumulator: %f, new_value: %f, %f\n", beta, cj, accumulator, new_value, new_value2);
        //}
    }
}

__global__ void MulMat_X1_ColumnMajor_Kernel(int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc)
{
    //ci: row index of C; cj: column index of C
    int ci = threadIdx.x + blockIdx.x * blockDim.x;
    int cj = threadIdx.y + blockIdx.y * blockDim.y;

    if (ci < M && cj < N)
    {
        double accumulator = 0;
        for (int kk = 0; kk < K; ++kk) {
            accumulator += A[ci + kk * lda] * B[kk + cj * ldb];
        }

        C[ci + cj * ldc] = alpha * (float)accumulator + beta * C[ci + cj * ldc];
    }
}

template <typename AType, typename BType, typename CType>
__global__ void MulMat_Y1_RowMajor_Kernel(int M, int N, int K,
    float alpha, AType const *A, BType const *B,
    float beta, CType *C)
{
    //ci: row index of C; cj: column index of C
    int ci = threadIdx.x + blockIdx.x * blockDim.x;
    int cj = threadIdx.y + blockIdx.y * blockDim.y;

    //printf("thread: (%d, %d); blockIdx: (%d, %d), blockDim: (%d, %d)\n",
    //    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);

    if (ci < M && cj < N)
    {
        double accumulator = 0;
        for (int kk = 0; kk < K; ++kk)
        {
            accumulator += (float)A[kk + ci * K] * (float)B[kk + cj * K];
        }

        C[cj + ci * N] = alpha * (float)accumulator + beta * (float)C[cj + ci * N];
    }
}

//depth: size of the z-dimension
template <typename AType, typename BType, typename CType>
__global__ void Gemm_Alg2_Kernel(int M, int N, int K, int depth,
    float alpha, AType const *A, BType const *B, float beta, CType *C,
    bool is_b_column_major, bool a_is_zy_data, bool b_is_zy_data,
    int r)
{
    //c_y: index of the y dimension of C; c_z: index of the z dimension of C
    int c_y = threadIdx.y + blockIdx.y * blockDim.y;
    int c_col = threadIdx.x + blockIdx.x * blockDim.x;
    int c_z = threadIdx.z + blockIdx.z * blockDim.z;
    int block_size = blockDim.x;

    // Do NOT return, because each thread is responsible for filling one element
    // in cache_a and cache_b (even with a zero value)
    //if (c_y >= M || c_col >= N) {
    //    return;
    //}

    // shared memory among all the threads in one block
    const int cache_len = Inferflow_MaxThreadPerBlock;
    __shared__ AType cache_a[cache_len];
    __shared__ BType cache_b[cache_len];

    if (blockDim.x != blockDim.y || blockDim.x * blockDim.y > cache_len) {
        return;
    }

    // t_row, t_col: thread row and column in the block
    // t_row is t_y, t_col is t_x
    int t_row = threadIdx.y, t_col = threadIdx.x;
    float c_value = 0;
    int offset_a = 0, offset_b = 0;

    //if (c_y == 1 && c_col == 0)
    //{
    //    printf("blockIdx: (%d, %d), c_y: %d, c_col: %d, t_row: %d, t_col: %d\n",
    //        blockIdx.x, blockIdx.y, c_y, c_col, t_row, t_col);
    //}

    // Loop over all the sub-matrices of A and B that are required to compute c_sub
    // Multiply each pair of sub-matrices together and accumulate the results
    for (int k1 = 0; k1 < K; k1 += block_size)
    {
        int row = a_is_zy_data ? (c_y * depth  + c_z) : (c_z * M  + c_y);
        offset_a = row * K + (k1 + t_col);
        //offset_a = a_is_zy_data ? (c_y * M * K  + c_z * K + (k1 + t_col)) : (c_z * M * K  + c_y * K + (k1 + t_col));
        cache_a[t_row * block_size + t_col] = c_y < M && k1 + t_col < K ? A[offset_a] : (AType)0;

        if (is_b_column_major)
        {
            row = b_is_zy_data ? (c_col * depth + c_z / r) : (c_z / r * N + c_col);
            offset_b = row * K + (k1 + t_row);
            //offset_b = c_z * N * K + c_col * K + (k1 + t_row);
            cache_b[t_col * block_size + t_row] = k1 + t_row < K && c_col < N ? B[offset_b] : (BType)0;
        }
        else
        {
            row = b_is_zy_data ? ((k1 + t_row) * depth + c_z / r) : (c_z / r * K + (k1 + t_row));
            offset_b = row * N + c_col;
            //offset_b = c_z * K * N + (k1 + t_row) * N + c_col;
            cache_b[t_col * block_size + t_row] = k1 + t_row < K && c_col < N ? B[offset_b] : (BType)0;
        }
        //printf("t_row_col: (%d, %d), offset: (%d, %d), cache_offset: (%d, %d), value: (%f, %f)\n",
        //    t_row, t_col, offset_a, offset_b, t_row * block_size + t_col, t_col * block_size + t_row,
        //    cache_a[t_row * block_size + t_col], cache_b[t_col * block_size + t_row]);

        // Synchronize to make sure the sub-matrices are loaded before starting the computation
        __syncthreads();

#       pragma unroll
        for (int idx = 0; idx < block_size; idx++)
        {
            offset_a = t_row * block_size + idx;
            offset_b = t_col * block_size + idx;
            //if (c_y == 0 && c_col == 0 && c_z == 0)
            //{
            //    printf("c_y & c_x: (%d, %d), cache_offset: (%d, %d), values: (%f, %f)\n",
            //        c_y, c_col, offset_a, offset_b, (float)cache_a[offset_a], (float)cache_b[offset_b]);
            //}

            //c_value += cache_a[t_row * block_size + idx] * cache_b[idx * block_size + t_col];
            c_value += (float)cache_a[offset_a] * (float)cache_b[offset_b];
        }

        // Synchronize to make sure that the preceding computation is done
        // before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write to C. Each thread writes one element
    if (c_z < depth  && c_y < M && c_col < N)
    {
        C[c_z * M * N + c_y * N + c_col] = alpha * c_value;
        //printf("c_y and c_x: (%d, %d), value: %f, N: %d\n", c_y, c_col, c_value, N);
    }
}

//Note: Column-major is assumed in B
template <typename AType, typename CType>
__global__ void GemmSparse_Alg1_Kernel(CType *C, AType const *A, SparseMatrixCell const *B,
    int const *row_offset_array, int M, int N, int K, int b_size, float alpha, float beta,
    bool a_is_zy_data)
{
    //c_y: index of the y dimension of C; c_x: index of the x dimension of C
    int c_y = threadIdx.y + blockIdx.y * blockDim.y;
    int c_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (c_x >= N || c_y >= M) {
        return;
    }

    float sum = 0;
    int start_idx = row_offset_array[c_x];
    int end_idx = c_x + 1 < N ? row_offset_array[c_x + 1] : b_size;

#   pragma unroll
    for (int idx = start_idx; idx < end_idx; idx++)
    {
        SparseMatrixCell cell = B[idx];
        int offset_a = c_y * K + cell.col;
        sum += (float)A[offset_a] * cell.score;
    }

    //printf("c_x: %d, c_y: %d, sum: %f\n", c_x, c_y, sum);
    C[c_y * N + c_x] += (CType)sum;
}

} //end of namespace
