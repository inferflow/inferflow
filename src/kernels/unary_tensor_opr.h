#pragma once

#include "kernels_common.h"
#include "lock.h"
#include "tensor/tensor_common.h"

namespace inferflow
{

template <typename SourceType, typename TargetType>
__global__ void Tensor_Transpose_Kernel(int size, int ne0, int ne1, int ne2,
    SourceType const *S, TargetType *T, int cell_len)
{
    //printf("thread: (%d, %d); blockIdx: (%d, %d), blockDim: (%d, %d), cell: (%d, %d)\n",
    //    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y,
    //    cell_len, cell_len);

    int ri_start = (threadIdx.x + blockIdx.x * blockDim.x) * cell_len;
    int cj_start = (threadIdx.y + blockIdx.y * blockDim.y) * cell_len;
    int ri_end = ri_start + cell_len;
    int cj_end = cj_start + cell_len;
    for (int ri = ri_start; ri < ne1 * ne2 && ri < ri_end; ri++)
    {
        for (int cj = cj_start; cj < ne0 && cj < cj_end; cj++)
        {
            int t_offset = ne0 * ne1 * (ri % ne2) + ne1 * cj + (ri / ne2);
            if (t_offset >= size)
            {
                printf("Invalid target offset: %d (>= %d)\n", t_offset, size);
                return;
            }

            T[t_offset] = S[cj + ri * ne0];
        }
    }
}

// S(cx, cy) --> T(cy, cx)
template <typename DataType>
__global__ void Tensor_Transpose_Alg2_Kernel(const DataType *S, DataType *T,
    int cx, int cy)
{
    const int tile_dim = 32, block_rows = 8;
    __shared__ float tile[tile_dim][tile_dim + 1];

    int x = blockIdx.x * tile_dim + threadIdx.x;
    int y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows)
    {
        int offset = (y + j) * cx + x;
        tile[threadIdx.y + j][threadIdx.x] = y + j < cy && x < cx ? (float)S[offset] : 0;
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x; // transpose block offset
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows)
    {
        if (y + j < cx && x < cy) {
            T[(y + j) * cy + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <typename SourceType, typename TargetType>
__global__ void Tensor_StdNorm_Kernel(int rows, int cols, SourceType const *S, TargetType *T,
    float eps, int rows_mul = 0, SourceType const *M = nullptr,
    int rows_add = 0, SourceType const *A = nullptr)
{
    int tid = threadIdx.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    //printf("thread: (%d, %d); blockIdx: (%d, %d), blockDim: (%d, %d)\n",
    //    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);

    const int block_size = blockDim.x;
    int x_len = (cols + blockDim.x - 1) / blockDim.x;
    //int x_start = tid * x_len;
    int x_end = min((tid + 1) * x_len, cols);

    const int cache_len = Inferflow_MaxThreadPerBlock;
    __shared__ float sum_cache[cache_len];
    __shared__ float sum2_cache[cache_len];

    if (row < rows)
    {
        float sum = 0.0f, sum2 = 0.0f;
        const SourceType *src_row = S + row * cols;

#       pragma unroll
        for (int xi = tid; xi < cols; xi += block_size)
        {
            const double v = (double)src_row[xi];
            sum += v;
            sum2 += v * v;
        }

        sum_cache[tid] = sum;
        sum2_cache[tid] = sum2;

        //printf("row: %d; tid: %d; x_len: %d; x_start: %d; x_end: %d; sum: %.3f\n",
        //    row, tid, x_len, x_start, x_end, (float)sum);

        __syncthreads();

        sum = 0.0;
        sum2 = 0.0;
        if (tid == 0)
        {
#           pragma unroll
            for (int idx = 0; idx < blockDim.x; idx++)
            {
                sum += sum_cache[idx];
                sum2 += sum2_cache[idx];
            }

            float mean = (float)(sum / cols);
            float var_value = (float)(sum2 / cols - mean * mean);
            float scale = rsqrtf(var_value + eps);
            sum_cache[0] = mean;
            sum2_cache[0] = scale;
        }

        __syncthreads();

        float mean = sum_cache[0];
        float scale = sum2_cache[0];
        TargetType *target_row = T + row * cols;

#       pragma unroll
        for (int xi = tid; xi < cols; xi += block_size)
        {
            float v = ((float)src_row[xi] - mean) * scale;
            if (M != nullptr && A != nullptr)
            {
                v = v * (float)M[(row % rows_mul) * cols + xi]
                    + (float)A[(row % rows_add) * cols + xi];
            }
            else if (M != nullptr)
            {
                v = v * (float)M[(row % rows_mul) * cols + xi];
            }

            target_row[xi] = (TargetType)v;
        }
    }
}

__global__ void Tensor_StdNorm_Half_Alg2S_Kernel(int rows, int cols, half const *S, half *T,
    float eps, int rows_mul = 0, half const *M = nullptr,
    int rows_add = 0, half const *A = nullptr)
{
    const int block_size = blockDim.x;
    int tid = threadIdx.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= rows) {
        return;
    }

    float mean = 0.0f;
    float var_value = 0.0f;
#   pragma unroll
    for (int xi = tid; xi < cols / 4; xi += block_size)
    {
        const float2 v = *(const float2*)(S + row * cols + 4 * xi);
        const half *vh = (const half*)&v;
#       pragma unroll
        for (int idx = 0; idx < 4; idx++)
        {
            mean += (float)vh[idx];
            var_value += ((float)vh[idx] * (float)vh[idx]);
        }
    }

    // sum up partial sums
    mean = WarpReduceSum(mean);
    var_value = WarpReduceSum(var_value);
 
    mean = mean / cols;
    var_value = var_value / cols - mean * mean;
    const float scale = rsqrtf(var_value + eps);

    const half *src_row = S + row * cols;
    half *target_row = T + row * cols;

#   pragma unroll
    for (int xi = tid; xi < cols / 4; xi += block_size)
    {
        const float2 src_f2 = *(const float2*)(src_row + 4 * xi);
        const half *src_vh = (const half*)&src_f2;
        float2 target_f2;
        half *target_vh = (half*)&target_f2;
#       pragma unroll
        for (int idx = 0; idx < 4; idx++)
        {
            float v = ((float)src_vh[idx] - mean) * scale;
            int delta_idx = 4 * xi + idx;
            if (M != nullptr && A != nullptr)
            {
                v = v * (float)M[(row % rows_mul) * cols + delta_idx]
                    + (float)A[(row % rows_add) * cols + delta_idx];
            }
            else if (M != nullptr)
            {
                v = v * (float)M[(row % rows_mul) * cols + delta_idx];
            }
            target_vh[idx] = (half)v;
        }

        *(float2*)(target_row + 4 * xi) = target_f2;
    }
}

template <typename SourceType, typename TargetType>
__global__ void Tensor_RmsNorm_Kernel(int rows, int cols, SourceType const *S, TargetType *T,
    float eps, int rows_mul = 0, SourceType const *M = nullptr,
    int rows_add = 0, SourceType const *A = nullptr)
{
    int tid = threadIdx.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    //printf("thread: (%d, %d); blockIdx: (%d, %d), blockDim: (%d, %d)\n",
    //    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);

    int x_len = (cols + blockDim.x - 1) / blockDim.x;
    int x_start = tid * x_len;
    int x_end = min((tid + 1) * x_len, cols);

    const int cache_len = Inferflow_MaxThreadPerBlock;
    __shared__ float sum_cache[cache_len];

    if (row < rows)
    {
        float sum = 0.0f;
        const SourceType *src_row = S + row * cols;

#       pragma unroll
        for (int xi = x_start; xi < x_end; xi++)
        {
            sum += ((double)src_row[xi] * (double)src_row[xi]);
        }

        sum_cache[tid] = sum;

        //printf("row: %d; tid: %d; x_len: %d; x_start: %d; x_end: %d; sum: %.3f\n",
        //    row, tid, x_len, x_start, x_end, (float)sum);

        __syncthreads();

        if (tid == 0)
        {
            sum = 0.0f;
#           pragma unroll
            for (int idx = 0; idx < blockDim.x; idx++)
            {
                sum += sum_cache[idx];
            }

            float mean = (float)(sum / cols);
            float scale = 1.0f / sqrtf(mean + eps);
            sum_cache[0] = scale;

            //printf("row: %d, sum: %g, mean: %g, scale: %f\n", row, sum, mean, scale);
        }

        __syncthreads();

        float scale = sum_cache[0];
        TargetType *target_row = T + row * cols;

#       pragma unroll
        for (int xi = x_start; xi < x_end; xi++)
        {
            float v = (float)src_row[xi] * scale;
            if (M != nullptr && A != nullptr)
            {
                v = v * (float)M[(row % rows_mul) * cols + xi]
                    + (float)A[(row % rows_add) * cols + xi];
            }
            else if (M != nullptr)
            {
                v = v * (float)M[(row % rows_mul) * cols + xi];
            }

            target_row[xi] = (TargetType)v;
        }
    }
}

template <typename SourceType, typename TargetType>
__global__ void Tensor_RmsNorm_Naive_Kernel(int rows, int cols,
    SourceType const *S, TargetType *T, float eps)
{
    //ri: row index
    int ri = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("thread: (%d, %d); blockIdx: (%d, %d), blockDim: (%d, %d)\n",
    //    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);

    if (ri < rows)
    {
        const SourceType *src_row = S + ri * cols;
        TargetType *target_row = T + ri * cols;

        double sum = 0.0f;
        for (int cj = 0; cj < cols; cj++) {
            sum += ((double)src_row[cj] * (double)src_row[cj]);
        }

        float mean = (float)(sum / cols);
        float scale = 1.0f / sqrtf(mean + eps);

        //printf("row: %d; cols: %d; sum: %f, mean: %f, scale: %f, v0: %f, v0_norm: %f\n",
        //    ri, cols, sum, mean, scale, (float)src_row[0], (float)src_row[0] * scale);

        for (int cj = 0; cj < cols; cj++) {
            target_row[cj] = (TargetType)((float)src_row[cj] * scale);
        }
    }
}

template <typename SourceType, typename TargetType>
__global__ void Tensor_AggregateByRow_Kernel(int M, int N, SourceType const *S,
    TargetType *T, RowAggregationType agg_type)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int batch_size = (N + blockDim.x - 1) / blockDim.x;
    int col_start = threadIdx.x * batch_size;
    int col_end = col_start + batch_size;

    const int cache_len = Inferflow_MaxThreadPerBlock;
    __shared__ float cache[cache_len];

    if (row < M && col_start < N)
    {
        float agg_value = (float)S[row * N + col_start];
        float value = agg_value;
        for (int col = col_start + 1; col < col_end && col < N; col++)
        {
            value = (float)S[row * N + col];
            switch (agg_type)
            {
            case RowAggregationType::Max:
                if (agg_value < value) {
                    agg_value = value;
                }
                break;
            case RowAggregationType::Min:
                if (agg_value > value) {
                    agg_value = value;
                }
                break;
            case RowAggregationType::Sum:
            default:
                agg_value += value;
                break;
            }
        }

        cache[threadIdx.y * blockDim.x + threadIdx.x] = agg_value;
        //printf("row: %d, col_start: %d, offset: %d, value: %f\n", row, col_start,
        //    threadIdx.y * blockDim.x + threadIdx.x, value);
    }

    //Wait until the temporary maximal values of all rows have been set
    __syncthreads();

    if (row < M && col_start < N && threadIdx.x == 0)
    {
        float agg_value = cache[threadIdx.y * blockDim.x];
        for (int idx = 1; idx < blockDim.x; idx++)
        {
            if (idx * batch_size >= N) {
                break;
            }

            float value = cache[threadIdx.y * blockDim.x + idx];
            switch (agg_type)
            {
                case RowAggregationType::Max:
                    if (agg_value < value) {
                        agg_value = value;
                    }
                    break;
                case RowAggregationType::Min:
                    if (agg_value > value) {
                        agg_value = value;
                    }
                    break;
                case RowAggregationType::Sum:
                default:
                    agg_value += value;
                    break;
            }

            //printf("row: %d, idx: %d, offset: %d, value: %f\n", row, idx,
            //    threadIdx.y * blockDim.x + idx, value);
        }

        T[row] = agg_value;
    }
}

template <typename SourceType, typename TargetType>
__global__ void Tensor_MaxByRow_NotWorking_Kernel(int M, int N, SourceType const *S,
    TargetType *T, int batch_size, unsigned int *mutex, bool is_min)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col_start = (threadIdx.x + blockIdx.x * blockDim.x) * batch_size;
    int col_end = col_start + batch_size;

    if (row == 0 && col_start == 0) {
        *mutex = 1;
    }

    if (row < M && col_start == 0)
    {
        //Set the column 0 of the row as the temporary maximal value
        T[row] = S[row * N];
    }

    //Wait until the temporary maximal values of all rows have been set
    __syncthreads();

    if (row < M && col_start < N)
    {
        //printf("row: %d, col_start: %d, col_end: %d, base_offset: %d\n",
        //    row, col_start, col_end,  row * N + col_start);
        float chosen_value = (float)S[row * N + col_start];
        float value = chosen_value;
        for (int col = col_start + 1; col < col_end && col < N; col++)
        {
            value = (float)S[row * N + col];
            if ((is_min && chosen_value > value) || (!is_min && chosen_value < value))
            {
                chosen_value = value;
            }
        }

        //if (row <= 1) {
        //    printf("=== row: %d, col_start: %d, chosen_value: %f\n", row, col_start, chosen_value);
        //}

        bool next = true;
        while (next)
        {
            MutexLock(mutex);
            float t_value = (float)T[row];
            if ((is_min && t_value > chosen_value) || (!is_min && t_value < chosen_value))
            {
                T[row] = chosen_value;
            }
            MutexUnlock(mutex);
            next = false;
        }
    }
}

template <typename SourceType, typename TargetType>
__global__ void Tensor_SoftMaxPre_Kernel(int M, int N, SourceType const *S,
    TargetType *T, SourceType const *max_data, float neg_infinity)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float x = (float)S[row * N + col];
        float row_max = max_data[row];
        float new_x = 0;
        if (x != neg_infinity)
        {
            new_x = exp(x - row_max);
        }

        T[row * N + col] = new_x;
    }
}

template <typename SourceType, typename TargetType>
static __global__ void Tensor_SoftMax_Alg2_Kernel(int cx, int cy, int cz,
    const SourceType *S, TargetType *T, float neg_infinity,
    int diag_mask_prefix_len, float mask_value, float s_scale)
{
    const int zi = threadIdx.z + blockIdx.z * blockDim.z;
    const int row = threadIdx.y + blockIdx.y * blockDim.y + zi * gridDim.y;
    const int block_size = blockDim.x;
    const int tid = threadIdx.x;

    float max_value = neg_infinity;
    for (int xi = tid; xi < cx; xi += block_size)
    {
        const int offset = row * cx + xi;
        float v = s_scale * (float)S[offset];
        if (diag_mask_prefix_len >= 0)
        {
            int r1 = row % cy;
            if (xi > diag_mask_prefix_len + r1) {
                v = mask_value;
            }
        }
        max_value = max(max_value, v);
    }

    max_value = WarpReduceMax(max_value);
    //printf("tid: %d, max_value: %.3f\n", tid, max_value);

    float sum = 0.0f;
    for (int xi = tid; xi < cx; xi += block_size)
    {
        const int offset = row * cx + xi;
        float v = s_scale * (float)S[offset];
        if (diag_mask_prefix_len >= 0)
        {
            int r1 = row % cy;
            if (xi > diag_mask_prefix_len + r1) {
                v = mask_value;
            }
        }
        const float val = expf(v - max_value);
        sum += val;
        T[offset] = val;
    }

    //printf("tid: %d, sum: %.3f\n", tid, sum);
    sum = WarpReduceSumAll(sum);
    const float scale = 1.0f / sum;
    //printf("tid: %d, sum: %.3f, scale: %.3f\n", tid, sum, scale);

    for (int xi = tid; xi < cx; xi += block_size)
    {
        const int offset = row * cx + xi;
        T[offset] *= scale;
    }
}

template <typename SourceType, typename TargetType>
__global__ void ReluActivation_Kernel(int M, int N,
    SourceType const *S, TargetType *T)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float x = (float)S[row * N + col];
        float fx = x > 0 ? x : 0;
        T[row * N + col] = (TargetType)fx;
    }
}

template <typename SourceType, typename TargetType>
__global__ void SiluActivation_Kernel(int M, int N,
    SourceType const *S, TargetType *T)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float x = (float)S[row * N + col];
        float fx = x/(1.0f + expf(-x));
        T[row * N + col] = fx;
    }
}

template <typename SourceType, typename TargetType>
__global__ void GeluActivation_Kernel(int M, int N,
    SourceType const *S, TargetType *T)
{
    static const float GELU_COEF_A    = 0.044715f;
    static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float x = (float)S[row * N + col];
        float fx = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
        T[row * N + col] = fx;
    }
}

template <typename SourceType, typename TargetType>
__global__ void PosEmbedding_Sinusoidal1_Order2_Kernel(SourceType const *S, TargetType *T,
    const int *token_id_array, int ne0, int ne1, int ne2, int context_len, int dims)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int half_dim = dims / 2;

    int idx2 = row / ne1;
    int idx1 = row % ne1;
    if (idx2 < ne2 && idx1 < ne1 && 2 * col < ne0)
    {
        float theta = (context_len + idx1) * pow(10000.0f, -1.0f * col / (half_dim - 1));
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        T[row * ne0 + col] = (float)S[row * ne0 + col] + sin_theta;
        T[row * ne0 + col + ne0 / 2] = (float)S[row * ne0 + col + ne0 / 2] + cos_theta;
    }
}

template <typename SourceType, typename TargetType>
__global__ void PosEmbedding_Sinusoidal2_Std_Kernel(SourceType const *S, TargetType *T,
    const int *token_id_array, int ne0, int ne1, int ne2, int context_len, int dims)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

    int idx2 = row / ne1;
    int idx1 = row % ne1;
    if (idx2 < ne2 && idx1 < ne1 && col + 1 < ne0)
    {
        float theta = (context_len + idx1) * pow(10000.0f, -2.0f * col / dims);
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        T[row * ne0 + col] = (float)S[row * ne0 + col] + sin_theta;
        T[row * ne0 + col + 1] = (float)S[row * ne0 + col + 1] + cos_theta;
        //T[row * ne0 + col] = sin_theta;
        //T[row * ne0 + col + 1] = cos_theta;
    }
}

template <typename SourceType, typename TargetType>
__global__ void PosEmbedding_Sinusoidal2_Order2_Kernel(SourceType const *S, TargetType *T,
    const int *token_id_array, int ne0, int ne1, int ne2, int context_len, int dims)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    int idx2 = row / ne1;
    int idx1 = row % ne1;
    if (idx2 < ne2 && idx1 < ne1 && 2 * col < ne0)
    {
        float theta = (context_len + idx1) * pow(10000.0f, -2.0f * col / dims);
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        T[row * ne0 + col] = (float)S[row * ne0 + col] + sin_theta;
        T[row * ne0 + col + ne0 / 2] = (float)S[row * ne0 + col + ne0 / 2] + cos_theta;
        //T[row * ne0 + col] = sin_theta;
        //T[row * ne0 + col + ne0 / 2] = cos_theta;
    }
}

template <typename SourceType, typename TargetType>
__global__ void PosEmbedding_Rope_Std_Kernel(SourceType const *S, TargetType *T,
    int ne0, int ne1, int ne2, int context_len, int dims, int mode, float theta)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

    const float theta_scale = powf(theta, -2.0f / dims);
    int start_idx2 = (mode & 1) == 0 ? 0 : context_len;

    int idx2 = row / ne1;
    int idx1 = row % ne1;
    if (idx2 >= start_idx2 && idx2 < ne2 && idx1 < ne1 && col + 1 < ne0)
    {
        const int p = (mode & 1) == 0 ? context_len + idx2 : idx2;

        float theta = (float)p;
        if (col > 0) {
            theta *= powf(theta_scale, col / 2);
        }

        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        const float x0 = S[row * ne0 + col];
        const float x1 = S[row * ne0 + col + 1];
        T[row * ne0 + col] = (TargetType)(x0 * cos_theta - x1 * sin_theta);
        T[row * ne0 + col + 1] = (TargetType)(x0 * sin_theta + x1 * cos_theta);
        //if (row <= 1 && col <= 4)
        //{
        //    printf("T[%d]: %f, T[%d]: %f, theta: %f, cos_theta: %f, sin_theta: %f\n",
        //        row * ne0 + col, x0 * cos_theta - x1 * sin_theta,
        //        row * ne0 + col + 1, x0 * sin_theta + x1 * cos_theta,
        //        theta, cos_theta, sin_theta);
        //}
    }
}

template <typename SourceType, typename TargetType>
__global__ void PosEmbedding_Rope_Order2_Kernel(SourceType const *S, TargetType *T,
    int ne0, int ne1, int ne2, int context_len, int dims, int mode, float theta,
    int rope_cols)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    const float theta_scale = powf(theta, -2.0f / dims);
    int start_idx2 = (mode & 1) == 0 ? 0 : context_len;

    int idx2 = row / ne1;
    int idx1 = row % ne1;
    if (idx2 >= start_idx2 && idx2 < ne2 && idx1 < ne1 && 2 * col < ne0)
    {
        const int p = (mode & 1) == 0 ? context_len + idx2 : idx2;

        float theta = (float)p;
        if (col > 0) {
            theta *= powf(theta_scale, col);
        }

        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        if (2 * col < rope_cols)
        {
            const float x0 = S[row * ne0 + col];
            const float x1 = S[row * ne0 + col + rope_cols / 2];
            T[row * ne0 + col] = (TargetType)(x0 * cos_theta - x1 * sin_theta);
            T[row * ne0 + col + rope_cols / 2] = (TargetType)(x0 * sin_theta + x1 * cos_theta);
        }
        else
        {
            int base_col = row * ne0 + 2 * col;
            const float x0 = S[base_col];
            const float x1 = S[base_col + 1];
            T[base_col] = x0;
            T[base_col + 1] = x1;
        }
    }
}

template <typename SourceType, typename TargetType>
__global__ void PosEmbedding_Alibi_Std_Kernel(SourceType const *S, TargetType *T,
    int ne0, int ne1, int ne2, int base_ne2, int context_len, int dims, int heads)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    const int heads_log2_floor = 1 << (int)floor(log2((float)heads));
    const float m0 = powf(2.0f, -8.0f / heads_log2_floor);
    const float m1 = powf(2.0f, -4.0f / heads_log2_floor);

    int idx2 = row / ne1;
    int idx1 = row % ne1;
    if (idx2 < ne2 && idx1 < ne1 && col < ne0)
    {
        idx2 += base_ne2;
        float m_k = idx2 < heads_log2_floor ? powf(m0, idx2 + 1)
            : powf(m1, 2 * (idx2 - heads_log2_floor) + 1);
        T[row * ne0 + col] = (TargetType)(col * m_k + (float)S[row * ne0 + col]);
    }
}

template <typename SourceType, typename TargetType>
__global__ void Transformer_DiagMask_Kernel(SourceType const *S, TargetType *T,
    int ne0, int ne1, int ne2, int context_len, float value)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < ne1 * ne2 && col < ne0)
    {
        int r1 = row % ne1;
        if (col > context_len + r1)
        {
            T[row * ne0 + col] = value;
        }
    }
}

} //end of namespace
