#pragma once

#include "kernels_common.h"
#include "common/quant_cuda.h"

namespace inferflow
{

//larger values may exceed the amout of shared memory reserved for one thread block (1K)
#define Inferflow_MaxNTile 1

// kernel for matrix–vector multiplication (vector * matrix)
// A: a row vector of size K
// B: a K*N matrix
// C: a row vector of size K
template <typename AType, typename BType, typename CType>
__global__ void Gemv_Alg1_Kernel(CType *C, const AType *A, const BType *B,
    int K, int N, int k_tile_len, int n_tile_len)
{
    __shared__ float block_cache[Inferflow_MaxThreadPerBlock * Inferflow_MaxNTile];

    const int start_ri = (blockDim.x * blockIdx.x + threadIdx.x) * k_tile_len;
    const int start_cj = blockIdx.y * n_tile_len;
    const int end_ri = start_ri + k_tile_len;
    const int end_cj = start_cj + n_tile_len;

    //printf("threadIdx: (%d, %d), blockIdx: (%d, %d), blockDim: (%d, %d), k_tile: %d, n_tile: %d\n",
    //    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
    //    blockDim.x, blockDim.y, k_tile_len, n_tile_len);
    //printf("ri range: [%d, %d); cj range: [%d, %d)\n", start_ri, end_ri, start_cj, end_cj);

    for (int cj = start_cj; cj < end_cj && cj < N; cj++)
    {
        float sum = 0;
        for (int ri = start_ri; ri < end_ri && ri < K; ri++)
        {
            sum += ((float)A[ri] * (float)B[cj + N * ri]);
            //if (cj - start_cj == 2)
            //{
            //    printf("block.thread: (%d,%d).%d, ri: %d, cj: %d, B_offset: %d, sum: %f, cache_idx: %d\n",
            //        blockIdx.x, blockIdx.y, threadIdx.x, ri, cj, cj + N * ri, sum,
            //        (cj - start_cj) * Inferflow_MaxThreadPerBlock + threadIdx.x);
            //}
        }
        block_cache[(cj - start_cj) * Inferflow_MaxThreadPerBlock + threadIdx.x] = sum;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int cj = start_cj; cj < end_cj && cj < N; cj++)
        {
            for (int idx = 0; idx < blockDim.x; idx++)
            {
                sum += block_cache[(cj - start_cj) * Inferflow_MaxThreadPerBlock + idx];
                //if (cj == 2)
                //{
                //    printf("cj: %d; idx: %d, cache_idx: %d, sum: %f\n", cj, idx,
                //        (cj - start_cj) * Inferflow_MaxThreadPerBlock + idx, sum);
                //}
            }
            C[cj] = sum;
        }
    }
}

template <typename AType, typename BType, typename CType>
__global__ void Gemv_Alg2_Kernel(CType *C, const AType *A, const BType *B,
    int K, int N)
{
    //ci: row index of C; cj: column index of C
    int c_row = threadIdx.y + blockIdx.y * blockDim.y;
    int c_col = threadIdx.x + blockIdx.x * blockDim.x;
    int block_size = blockDim.x;

    // Do NOT return, because each thread is responsible for filling one element
    // in cache_a and cache_b (even with a zero value)
    //if (c_row >= M || c_col >= N) {
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
    int t_row = threadIdx.y, t_col = threadIdx.x;
    float c_value = 0;
    int offset_a = 0, offset_b = 0;

    for (int k1 = 0; k1 < K; k1 += block_size)
    {
        offset_a = k1 + t_col;
        cache_a[t_row * block_size + t_col] = c_row < 1 && k1 + t_col < K ? A[offset_a] : (AType)0;
        offset_b = (k1 + t_row) * N + c_col;
        cache_b[t_col * block_size + t_row] = k1 + t_row < K && c_col < N ? B[offset_b] : (BType)0;
        //printf("t_row_col: (%d, %d), offset: (%d, %d), cache_offset: (%d, %d), value: (%f, %f)\n",
        //    t_row, t_col, offset_a, offset_b, t_row * block_size + t_col, t_col * block_size + t_row,
        //    cache_a[t_row * block_size + t_col], cache_b[t_col * block_size + t_row]);

        // Synchronize to make sure the sub-matrices are loaded before starting the computation
        __syncthreads();

        for (int idx = 0; idx < block_size; idx++)
        {
            offset_a = t_row * block_size + idx;
            offset_b = t_col * block_size + idx;
            //if (c_row == 1 && c_col == 0)
            //{
            //    printf("c_row_col: (%d, %d), cache_offset: (%d, %d), values: (%f, %f)\n",
            //        c_row, c_col, offset_a, offset_b, cache_a[offset_a], cache_b[offset_b]);
            //}
            //c_value += cache_a[t_row * block_size + idx] * cache_b[idx * block_size + t_col];
            c_value += (float)cache_a[offset_a] * (float)cache_b[offset_b];
        }

        // Synchronize to make sure that the preceding computation is done
        // before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    if (c_row < 1 && c_col < N) {
        //atomicAdd(C + c_row * N + c_col, c_value);
        C[c_row * N + c_col] = c_value;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void Sgemv_Alg3_AX128_Kernel(float * __restrict__ Y,
    const float * __restrict__ A, const float * __restrict__ X,
    const float * __restrict__ bias_data, int M, int N)
{
    const int stride = 4;
    //row of A and Y
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    int cache_idx = threadIdx.x / WARP_SIZE;

    __shared__ float sum_cache[128];

    if (row_idx >= M) {
        return;
    }

    float sum = 0;
    {
        int iterations = max(1, N / blockDim.x / stride);
        const float *row_data = A + row_idx * N;

        #pragma unroll
        for(int idx = 0; idx < iterations; idx++)
        {
            int current_col_vec = (idx * blockDim.x + threadIdx.x % blockDim.x);
            float4 current_val = ((const float4*)row_data)[current_col_vec];
            float4 current_x = ((const float4*)X)[current_col_vec];
            sum += current_val.x * current_x.x;
            sum += current_val.y * current_x.y;
            sum += current_val.z * current_x.z;
            sum += current_val.w * current_x.w;
        }

        sum = WarpReduceSum(sum, WARP_SIZE);
    }

    if (lane_id == 0)
    {
        if (blockDim.x <= WARP_SIZE) {
            Y[row_idx] = bias_data == nullptr ? sum : (sum + bias_data[row_idx]);
            return;
        }

        sum_cache[cache_idx] = sum;
        __syncthreads();

        if (cache_idx == 0)
        {
            sum = 0;
            #pragma unroll
            for (int idx = 0; idx < blockDim.x / WARP_SIZE; idx++) {
                sum += sum_cache[idx];
            }
            Y[row_idx] = bias_data == nullptr ? sum : (sum + bias_data[row_idx]);
        }    
    }
}

// Y = X*A (X: 1*M, A: M*N, Y: 1*N)
__global__ void GemvHalf_XA_Alg3_S8_Kernel(half * __restrict__ Y,
    const half * __restrict__ X, const half * __restrict__ A,
    int M, int N)
{
    const int y_stride = 4;
    const int x_stride = 8;
    const int warp_num_per_block = 4;
    int tx = threadIdx.x;
    int ty = threadIdx.y; //here: blockDim.y == 1 and blockIdx.y == 0
    int thread_id = threadIdx.x * WARP_SIZE + threadIdx.y;
    int col_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    //int warp_id = col_idx / WARP_SIZE;
    const float4 *A4 = reinterpret_cast<const float4*>(A);

    __shared__ float sum_cache[warp_num_per_block][8];

    if (col_idx >= N) {
        return;
    }

    float sum[8] = {};
    int iterations = max(1, M / WARP_SIZE / y_stride);
    #pragma unroll
    for(int iter_idx = 0; iter_idx < iterations; iter_idx++)
    {
        int row_idx = iter_idx * WARP_SIZE * y_stride + ty * y_stride + tx;
        if (row_idx < M)
        {
            const float4 *row_data = A4 + row_idx * (N / x_stride);
            float4 float4_a = row_data[col_idx / x_stride];

            half x_h = X[row_idx];
            float x_f = (float)x_h;
            const half *data_array = (const half*)&float4_a;
            #pragma unroll
            for (int idx = 0; idx < 8; idx++)
            {
                sum[idx] += x_f * (float)data_array[idx];
            }
        }
    }

    #pragma unroll
    for (int idx = 0; idx < 8; idx++)
    {
        sum[idx] = WarpReduceSum(sum[idx], WARP_SIZE);
    }

    if (thread_id % WARP_SIZE == 0)
    {
        #pragma unroll
        for (int idx = 0; idx < 16; idx++)
        {
            sum_cache[thread_id / WARP_SIZE][idx] = sum[idx];
        }
    }

    __syncthreads();
    if (ty == 0 && tx == 0)
    {
        float4 res;
        #pragma unroll 
        for (int idx = 0; idx < 16; idx++)
        {
            float sum = 0;
            sum += sum_cache[0][idx];
            sum += sum_cache[1][idx];
            sum += sum_cache[2][idx];
            sum += sum_cache[3][idx];
            ((half*)&res)[idx] = (half)sum;
        }

        float4 *Y4 = reinterpret_cast<float4*>(Y);
        Y4[col_idx / x_stride] = res;
    }
}

// Y = X*A (X: 1*M, A: M*N, Y: 1*N)
__global__ void GemvHalf_XA_Alg3_S16_Kernel(half * __restrict__ Y,
    const half * __restrict__ X, const half * __restrict__ A,
    int M, int N)
{
    const int y_stride = 4;
    const int x_stride = 16;
    const int warp_num_per_block = 4;
    int tx = threadIdx.x;
    int ty = threadIdx.y; //here: blockDim.y == 1 and blockIdx.y == 0
    int thread_id = threadIdx.x * WARP_SIZE + threadIdx.y;
    int col_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    //int warp_id = col_idx / WARP_SIZE;
    const Float8 *A8 = reinterpret_cast<const Float8*>(A);

    __shared__ float sum_cache[warp_num_per_block][16];

    if (col_idx >= N) {
        return;
    }

    float sum[16] = {};
    int iterations = max(1, M / WARP_SIZE / y_stride);
    #pragma unroll
    for(int iter_idx = 0; iter_idx < iterations; iter_idx++)
    {
        int row_idx = iter_idx * WARP_SIZE * y_stride + ty * y_stride + tx;
        if (row_idx < M)
        {
            const Float8 *row_data = A8 + row_idx * (N / x_stride);
            Float8 float8_a = row_data[col_idx / x_stride];

            half x_h = X[row_idx];
            float x_f = (float)x_h;
            const half *data_array = (const half*)&float8_a;
            #pragma unroll
            for (int idx = 0; idx < 16; idx++)
            {
                sum[idx] += x_f * (float)data_array[idx];
            }
        }
    }

    #pragma unroll
    for (int idx = 0; idx < 16; idx++)
    {
        sum[idx] = WarpReduceSum(sum[idx], WARP_SIZE);
    }

    if (thread_id % WARP_SIZE == 0)
    {
        #pragma unroll
        for (int idx = 0; idx < 16; idx++)
        {
            sum_cache[thread_id / WARP_SIZE][idx] = sum[idx];
        }
    }

    __syncthreads();
    if (ty == 0 && tx == 0)
    {
        Float8 res;
        #pragma unroll 
        for (int idx = 0; idx < 16; idx++)
        {
            float sum = 0;
            sum += sum_cache[0][idx];
            sum += sum_cache[1][idx];
            sum += sum_cache[2][idx];
            sum += sum_cache[3][idx];
            ((half*)&res)[idx] = (half)sum;
        }

        Float8 *Y8 = reinterpret_cast<Float8*>(Y);
        Y8[col_idx / x_stride] = res;
    }
}

// Y = X*A (X: 1*M, A: M*N, Y: 1*N)
__global__ void GemvHalf_XA_Alg3_Kernel(half * __restrict__ Y,
    const half * __restrict__ X, const half * __restrict__ A,
    int M, int N)
{
    const int stride = 4;
    const int warp_num_per_block = 4;
    int tx = threadIdx.x;
    int ty = threadIdx.y; //here: blockDim.y == 1 and blockIdx.y == 0
    int thread_id = threadIdx.x * WARP_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int lane_id = threadIdx.x % WARP_SIZE;
    //int warp_id = col_idx / WARP_SIZE;
    const float2 *A2 = reinterpret_cast<const float2*>(A);
    //const float2 *X2 = reinterpret_cast<const float2*>(X);

    //__shared__ float2 cache_x[WARP_SIZE];
    //__shared__ float2 cache_a[WARP_SIZE][stride];
    __shared__ float sum1_cache[warp_num_per_block];
    __shared__ float sum2_cache[warp_num_per_block];
    __shared__ float sum3_cache[warp_num_per_block];
    __shared__ float sum4_cache[warp_num_per_block];
    /*if (ty == 0)
    {
        printf("M: %d, N: %d, col_idx: %d, blockDim: (%d, %d), blockIdx: (%d, %d), threadIdx: (%d, %d)\n",
            M, N, col_idx, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    }*/

    if (col_idx >= N) {
        return;
    }

    float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    int iterations = max(1, M / WARP_SIZE / stride);
    #pragma unroll
    for(int iter_idx = 0; iter_idx < iterations; iter_idx++)
    {
        int row_idx = iter_idx * WARP_SIZE * stride + ty * stride + tx;
        if (row_idx < M)
        {
            //cache_x[ty].x = 0;
            //cache_x[ty].y = 0;
            //cache_a[ty][tx].x = 0;
            //cache_a[ty][tx].y = 0;
            //load some elements of X and A to the cache
            /*if (threadIdx.x == 0) {
                cache_x[ty] = X2[row_idx / stride];
            }

            __syncthreads();*/

            const float2 *row_data = A2 + row_idx * (N / stride);
            float2 float2_a = row_data[col_idx / stride];
            //cache_a[ty][tx] = row_data[col_idx / stride];

            //half x_h = ((const half*)&cache_x[ty])[tx];
            half x_h = X[row_idx];
            float x_f = (float)x_h;
            const half2 *a_h1 = (const half2*)&float2_a.x;
            const half2 *a_h2 = (const half2*)&float2_a.y;
            sum1 += x_f * (float)(a_h1->x);
            sum2 += x_f * (float)(a_h1->y);
            sum3 += x_f * (float)(a_h2->x);
            sum4 += x_f * (float)(a_h2->y);

            /*if (ty == 0 || ty == 31)
            {
                printf("thread: (%d, %d), x_h: %f, a_h1: (%f, %f), a_h2: (%f, %f), sum: (%f, %f, %f, %f)\n",
                    tx, ty, (float)x_h, (float)a_h1->x, (float)a_h1->y, (float)a_h2->x, (float)a_h2->y,
                    sum1, sum2, sum3, sum4);
            }*/
        }
    }

    sum1 = WarpReduceSum(sum1, WARP_SIZE);
    sum2 = WarpReduceSum(sum2, WARP_SIZE);
    sum3 = WarpReduceSum(sum3, WARP_SIZE);
    sum4 = WarpReduceSum(sum4, WARP_SIZE);
    /*if (ty == 0 || ty == 31)
    {
        printf("thread: (%d, %d), partial_sum: (%f, %f, %f, %f)\n",
            tx, ty, sum1, sum2, sum3, sum4);
    }*/

    if (thread_id % WARP_SIZE == 0)
    {
        sum1_cache[thread_id / WARP_SIZE] = sum1;
        sum2_cache[thread_id / WARP_SIZE] = sum2;
        sum3_cache[thread_id / WARP_SIZE] = sum3;
        sum4_cache[thread_id / WARP_SIZE] = sum4;
    }

    __syncthreads();
    if (ty == 0 && tx == 0)
    {
        sum1 = 0; sum2 = 0; sum3 = 0; sum4 = 0;
        #pragma unroll
        for (int idx = 0; idx < warp_num_per_block; idx++)
        {
            sum1 += sum1_cache[idx];
            sum2 += sum2_cache[idx];
            sum3 += sum3_cache[idx];
            sum4 += sum4_cache[idx];
        }

        //if (col_idx <= 32) {
        //    printf("col_idx: %d, sum: (%f, %f, %f, %f)\n", col_idx, sum1, sum2, sum3, sum4);
        //}

        float2 res;
        ((half2*)&res.x)->x = (half)sum1;
        ((half2*)&res.x)->y = (half)sum2;
        ((half2*)&res.y)->x = (half)sum3;
        ((half2*)&res.y)->y = (half)sum4;
        float2 *Y2 = reinterpret_cast<float2*>(Y);
        Y2[col_idx / stride] = res;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Alg3_Kernel(half * __restrict__ Y,
    const half * __restrict__ A, const half * __restrict__ X,
    const half * __restrict__ bias_data, int M, int N)
{
    const int stride = 8;
    const int zi = threadIdx.z + blockIdx.z * blockDim.z;
    const int row_idx = threadIdx.y + blockIdx.y * blockDim.y + zi * gridDim.y;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    const float4 *A4 = reinterpret_cast<const float4*>(A);
    const float4 *X4 = reinterpret_cast<const float4*>(X);

    if (row_idx >= M) {
        return;
    }

    half sum = 0;
    if (threadIdx.x < N / stride)
    {
        int iterations = max(1, N / blockDim.x / stride);
        const float4 *row_data = A4 + row_idx * (N / stride);

        #pragma unroll
        for(int idx = 0; idx < iterations; idx++)
        {
            int float4_offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            float4 a_val = row_data[float4_offset];
            float4 x_val = X4[float4_offset];
            const half2 *x_h1 = (half2*)&x_val.x;
            const half2 *x_h2 = (half2*)&x_val.y;
            const half2 *x_h3 = (half2*)&x_val.z;
            const half2 *x_h4 = (half2*)&x_val.w;
            const half2 *a_h1 = (half2*)&a_val.x;
            const half2 *a_h2 = (half2*)&a_val.y;
            const half2 *a_h3 = (half2*)&a_val.z;
            const half2 *a_h4 = (half2*)&a_val.w;
            sum = __hfma(x_h1->x, a_h1->x, sum);
            sum = __hfma(x_h1->y, a_h1->y, sum);
            sum = __hfma(x_h2->x, a_h2->x, sum);
            sum = __hfma(x_h2->y, a_h2->y, sum);
            sum = __hfma(x_h3->x, a_h3->x, sum);
            sum = __hfma(x_h3->y, a_h3->y, sum);
            sum = __hfma(x_h4->x, a_h4->x, sum);
            sum = __hfma(x_h4->y, a_h4->y, sum);
            //sum = __hadd(sum, __hmul(x_h1->x, a_h1->x));
            //sum = __hadd(sum, __hmul(x_h1->y, a_h1->y));
            //sum = __hadd(sum, __hmul(x_h2->x, a_h2->x));
            //sum = __hadd(sum, __hmul(x_h2->y, a_h2->y));
            //sum = __hadd(sum, __hmul(x_h3->x, a_h3->x));
            //sum = __hadd(sum, __hmul(x_h3->y, a_h3->y));
            //sum = __hadd(sum, __hmul(x_h4->x, a_h4->x));
            //sum = __hadd(sum, __hmul(x_h4->y, a_h4->y));
        }
    }

    sum = WarpReduceSumAll(sum);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = bias_data == nullptr ? sum : (sum + bias_data[row_idx]);
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ half warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? (half)0
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = bias_data == nullptr ? sum : (sum + bias_data[row_idx]);
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*K, where K = N / stride / WARP_SIZE)
__global__ void GemvHalf_AX_Alg4_Kernel(half * __restrict__ Y,
    const half * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 8;
    //const int K = N / stride / WARP_SIZE;
    const int K = N / stride / blockDim.x;
    //int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    //int warp_id = col_idx / WARP_SIZE;
    int cache_idx = threadIdx.x / WARP_SIZE;
    const float4 *A4 = reinterpret_cast<const float4*>(A);
    const float4 *X4 = reinterpret_cast<const float4*>(X);

    __shared__ float sum_cache[16];

    float sum = 0;
    if (row_idx < M && col_idx < N / stride)
    {
        const float4 *row_data = A4 + row_idx * (N / stride);
        int float4_offset = col_idx / stride;
        float4 a_val = row_data[float4_offset];
        float4 x_val = X4[float4_offset];

        const half2 *x_h1 = (half2*)&x_val.x;
        const half2 *x_h2 = (half2*)&x_val.y;
        const half2 *x_h3 = (half2*)&x_val.z;
        const half2 *x_h4 = (half2*)&x_val.w;
        const half2 *a_h1 = (half2*)&a_val.x;
        const half2 *a_h2 = (half2*)&a_val.y;
        const half2 *a_h3 = (half2*)&a_val.z;
        const half2 *a_h4 = (half2*)&a_val.w;
        sum += (float)(x_h1->x) * (float)(a_h1->x);
        sum += (float)(x_h1->y) * (float)(a_h1->y);
        sum += (float)(x_h2->x) * (float)(a_h2->x);
        sum += (float)(x_h2->y) * (float)(a_h2->y);
        sum += (float)(x_h3->x) * (float)(a_h3->x);
        sum += (float)(x_h3->y) * (float)(a_h3->y);
        sum += (float)(x_h4->x) * (float)(a_h4->x);
        sum += (float)(x_h4->y) * (float)(a_h4->y);
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (lane_id == 0)
    {
        /*if (blockDim.x <= WARP_SIZE)
        {
            //if (row_idx <= 4) {
                printf("blockDim.x: %d", blockDim.x);
            //}
            Y[row_idx * K + warp_id] = (half)sum;
            return;
        }*/

        sum_cache[cache_idx] = sum;
        //Y[row_idx * K + warp_id] = (half)sum;
    }

    __syncthreads();

    if (cache_idx == 0 && lane_id == 0)
    {
        sum = 0;
        #pragma unroll
        for (int idx = 0; idx < blockDim.x / WARP_SIZE; idx++) {
            sum += sum_cache[idx];
        }
        int offset = row_idx * K + col_idx / blockDim.x;
        Y[offset] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q8_B32T1_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int quant_block_num_per_row = N / stride;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ8_B32T1 *block_array = reinterpret_cast<const BlockQ8_B32T1*>(A);
    const Float16 *X32 = reinterpret_cast<const Float16*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[32];
    float sum = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ8_B32T1 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / stride));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ8_B32T1 a_block = row_data[offset];
                Float16 x_val = X32[offset];
                Quantization::DequantizeQ8_B32T1(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < stride; idx++) {
                    sum += (float)arr_a[idx] * (float)arr_x[idx];
                }
            }
        }
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = (half)sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ float warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? 0.0f
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q8_B32T2_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int quant_block_num_per_row = N / stride;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ8_B32T2 *block_array = reinterpret_cast<const BlockQ8_B32T2*>(A);
    const Float16 *X32 = reinterpret_cast<const Float16*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[32];
    half sum = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ8_B32T2 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / stride));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ8_B32T2 a_block = row_data[offset];
                Float16 x_val = X32[offset];
                Quantization::DequantizeQ8_B32T2(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < stride; idx++) {
                    sum = __hadd(sum, __hmul(arr_a[idx], arr_x[idx]));
                }
            }
        }
    }

    sum = WarpReduceSumAll(sum);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ half warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? (half)0
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q6_B64T1_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int block_capacity = Q6_B64_CAPACITY;
    const int quant_block_num_per_row = N / block_capacity;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ6_B64T1 *block_array = reinterpret_cast<const BlockQ6_B64T1*>(A);
    const Float32 *X64 = reinterpret_cast<const Float32*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[block_capacity];
    float sum = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ6_B64T1 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / Q6_B64_CAPACITY));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ6_B64T1 a_block = row_data[offset];
                Float32 x_val = X64[offset];
                Quantization::DequantizeQ6_B64T1(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < block_capacity; idx++) {
                    sum += (float)arr_a[idx] * (float)arr_x[idx];
                }
            }
        }
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = (half)sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ float warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? 0.0f
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q5_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int quant_block_num_per_row = N / stride;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ5_B32T1 *block_array = reinterpret_cast<const BlockQ5_B32T1*>(A);
    const Float16 *X32 = reinterpret_cast<const Float16*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[32];
    float sum = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ5_B32T1 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / stride));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ5_B32T1 a_block = row_data[offset];
                Float16 x_val = X32[offset];
                Quantization::DequantizeQ5Block(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < stride; idx++) {
                    sum += (float)arr_a[idx] * (float)arr_x[idx];
                }
            }
        }
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = (half)sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ float warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? 0.0f
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q4_B32T1_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int quant_block_num_per_row = N / stride;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ4_B32T1 *block_array = reinterpret_cast<const BlockQ4_B32T1*>(A);
    const Float16 *X32 = reinterpret_cast<const Float16*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[32];
    half sum_f16 = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ4_B32T1 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / stride));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ4_B32T1 a_block = row_data[offset];
                Float16 x_val = X32[offset];
                Quantization::DequantizeQ4_B32T1(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < stride; idx++) {
                    sum_f16 = __hadd(sum_f16, __hmul(arr_a[idx], arr_x[idx]));
                }
            }
            else
            {
                //printf("threadIdx.x: %d, iterations: %d/%d; offset: %d, quant_block_num_per_row: %d\n",
                //    threadIdx.x, idx, iterations, offset, quant_block_num_per_row);
            }
        }
    }

    sum_f16 = WarpReduceSumAll(sum_f16);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = sum_f16;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ half warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum_f16;
    }

    __syncthreads();

    sum_f16 = (threadIdx.x >= blockDim.x / WARP_SIZE) ? (half)0
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum_f16 = WarpReduceSum(sum_f16, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum_f16;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q4B16_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 16;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ4_B16 *block_array = reinterpret_cast<const BlockQ4_B16*>(A);
    const Float8 *X32 = reinterpret_cast<const Float8*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[32];
    float sum = 0;
    if (threadIdx.x < N / stride)
    {
        int iterations = max(1, N / blockDim.x / stride);
        const BlockQ4_B16 *row_data = block_array + row_idx * (N / stride);

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / stride));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            BlockQ4_B16 a_block = row_data[offset];
            Float8 x_val = X32[offset];
            Quantization::DequantizeQ4_B16(arr_a, &a_block);
            const half *arr_x = (const half*)&x_val;

#           pragma unroll
            for (int idx = 0; idx < stride; idx++) {
                sum += (float)arr_a[idx] * (float)arr_x[idx];
            }
        }
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = (half)sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ float warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? 0.0f
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q3H_B64T1_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int block_capacity = Q3H_B64_CAPACITY;
    const int quant_block_num_per_row = N / block_capacity;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ3H_B64T1 *block_array = reinterpret_cast<const BlockQ3H_B64T1*>(A);
    const Float32 *X64 = reinterpret_cast<const Float32*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[block_capacity];
    float sum = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ3H_B64T1 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / Q3H_B64_CAPACITY));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ3H_B64T1 a_block = row_data[offset];
                Float32 x_val = X64[offset];
                Quantization::DequantizeQ3H_B64T1(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < block_capacity; idx++) {
                    sum += (float)arr_a[idx] * (float)arr_x[idx];
                }
            }
        }
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = (half)sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ float warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? 0.0f
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q3_B32T1_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int quant_block_num_per_row = N / stride;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ3_B32T1 *block_array = reinterpret_cast<const BlockQ3_B32T1*>(A);
    const Float16 *X32 = reinterpret_cast<const Float16*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[32];
    float sum = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ3_B32T1 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / stride));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ3_B32T1 a_block = row_data[offset];
                Float16 x_val = X32[offset];
                Quantization::DequantizeQ3_B32T1(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < stride; idx++) {
                    sum += (float)arr_a[idx] * (float)arr_x[idx];
                }
            }
        }
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = (half)sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ float warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? 0.0f
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void GemvHalf_AX_Q2_B32T1_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const half * __restrict__ X,
    int M, int N)
{
    const int stride = 32;
    const int quant_block_num_per_row = N / stride;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x % WARP_SIZE;
    const BlockQ2_B32T1 *block_array = reinterpret_cast<const BlockQ2_B32T1*>(A);
    const Float16 *X32 = reinterpret_cast<const Float16*>(X);

    if (row_idx >= M) {
        return;
    }

    half arr_a[32];
    float sum = 0;
    if (threadIdx.x < quant_block_num_per_row)
    {
        int iterations = max(1, (quant_block_num_per_row + blockDim.x - 1) / blockDim.x);
        const BlockQ2_B32T1 *row_data = block_array + row_idx * quant_block_num_per_row;

        //if (threadIdx.x == 0)
        //{
        //    printf("row_idx: %d, tx: %d, row_data offset: %d\n", row_idx,
        //        threadIdx.x, row_idx * (N / stride));
        //}

#       pragma unroll
        for (int idx = 0; idx < iterations; idx++)
        {
            int offset = (idx * blockDim.x + threadIdx.x % blockDim.x);
            if (offset < quant_block_num_per_row)
            {
                BlockQ2_B32T1 a_block = row_data[offset];
                Float16 x_val = X32[offset];
                Quantization::DequantizeQ2_B32T1(arr_a, &a_block);
                const half *arr_x = (const half*)&x_val;

#               pragma unroll
                for (int idx = 0; idx < stride; idx++) {
                    sum += (float)arr_a[idx] * (float)arr_x[idx];
                }
            }
            else
            {
                //printf("threadIdx.x: %d, iterations: %d/%d; offset: %d, quant_block_num_per_row: %d\n",
                //    threadIdx.x, idx, iterations, offset, quant_block_num_per_row);
            }
        }
    }

    sum = WarpReduceSum(sum, WARP_SIZE);
    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0) {
            Y[row_idx] = (half)sum;
        }
        return;
    }

    const int MAX_WARPS_PER_ROW = 32;
    const int MAX_BLOCK_DIM_Y = 16;
    static __shared__ float warp_level_sums[MAX_BLOCK_DIM_Y][MAX_WARPS_PER_ROW];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        warp_level_sums[threadIdx.y][warp_id] = sum;
    }

    __syncthreads();

    sum = (threadIdx.x >= blockDim.x / WARP_SIZE) ? 0.0f
        : warp_level_sums[threadIdx.y][lane_id];

    if (warp_id == 0) {
        sum = WarpReduceSum(sum, blockDim.x / WARP_SIZE);
    }
    if (tid == 0) {
        Y[row_idx] = (half)sum;
    }
}

// Y = A*X (A: M*N, X: N*1, Y: M*1)
__global__ void Gemv_AX8_Q8_B32T2_Kernel(half * __restrict__ Y,
    const uint8_t * __restrict__ A, const uint8_t * __restrict__ X,
    int cx, int cy)
{
    const int stride = 32;
    const int quant_blocks_per_row = cx / stride;
    const int groups_per_quant_block = Q8B32_CAPACITY / 4; //4 is related to dp4a
    const int groups_per_thread = 2;
    const int quant_blocks_per_warp = groups_per_thread * WARP_SIZE / groups_per_quant_block;
    int tid = threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    //int lane_id = threadIdx.x % WARP_SIZE;

    if (row_idx >= cy) {
        return;
    }

    const auto *blocks_x = reinterpret_cast<const BlockQ8_B32T2*>(X);
    const auto *blocks_a = reinterpret_cast<const BlockQ8_B32T2*>(A);

    float sum = 0.0f;
    int n1 = 0, n2 = 0;
    for (int block_idx = 0; block_idx < quant_blocks_per_row; block_idx += quant_blocks_per_warp)
    {
        int start_group_idx = (tid * groups_per_thread) % groups_per_quant_block;
        int end_group_idx = start_group_idx + groups_per_thread;
        int delta_idx = (tid * groups_per_thread) / groups_per_quant_block;
        const auto *block_x = blocks_x + block_idx + delta_idx;
        const auto *block_a = blocks_a + row_idx * quant_blocks_per_row + block_idx + delta_idx;

        int group_sum = 0;
#       pragma unroll
        for (int group_idx = start_group_idx; group_idx < end_group_idx; group_idx++)
        {
            n1 = Quantization::GetInt4(*block_a, 4 * group_idx);
            n2 = Quantization::GetInt4(*block_x, 4 * group_idx);
            group_sum = __dp4a(n1, n2, group_sum);
            //if (row_idx == 0 && tid == 0 && group_idx == 0)
            //{
            //    printf("[block_a] scale: %.7f, n1: %d, data: %d, %d, %d, %d, %d, %d, %d, %d\n",
            //        (float)block_a->scale, n1,
            //        (int)block_a->data[0], (int)block_a->data[1],
            //        (int)block_a->data[2], (int)block_a->data[3],
            //        (int)block_a->data[4], (int)block_a->data[5],
            //        (int)block_a->data[6], (int)block_a->data[7]);
            //    printf("[block_x] scale: %.7f, n2: %d, data: %d, %d, %d, %d, %d, %d, %d, %d\n",
            //        (float)block_x->scale, n2,
            //        (int)block_x->data[0], (int)block_x->data[1],
            //        (int)block_x->data[2], (int)block_x->data[3],
            //        (int)block_x->data[4], (int)block_x->data[5],
            //        (int)block_x->data[6], (int)block_x->data[7]);
            //}
        }

        //sum += (float)__hmul(block_x->scale, block_a->scale) * group_sum;
        sum += (float)block_a->scale * group_sum * (float)block_x->scale;
        //if (row_idx == 0)
        //{
        //    printf("row: %d, tid: %d, block_idx: %d, group range: [%d, %d), group_sum: %d, sum: %.3f\n",
        //        row_idx, tid, block_idx + delta_idx, start_group_idx, end_group_idx, group_sum, sum);
        //}
    }

    sum = WarpReduceSumAll(sum);

    if (tid == 0) {
        Y[row_idx] = sum;
    }
}

} //end of namespace
