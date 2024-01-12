#pragma once

#include "common/quant_cuda.h"

namespace inferflow
{

__global__ void Tensor_Assign_Kernel(float *A, int len, float value, int batch_size)
{
    int start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_size == 1 && start_idx < len)
    {
        A[start_idx] = value;
        return;
    }

    int end_idx = min(len, start_idx + batch_size);
    for (int idx = start_idx; idx < end_idx; idx++)
    {
        A[idx] = value;
    }
}

template <typename AType, typename BType>
__global__ void Tensor_Assign2_Kernel(const AType *A, int cx, int cy, int cz,
    bool a_is_zy_data, BType *B, int b_size)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix < cx && iy < cy && iz < cz)
    {
        int row_a = a_is_zy_data ? (iy * cz + iz) : (iz * cy + iy);
        int offset_a = row_a * cx + ix;
        int row_b = iz * cy + iy;
        int offset_b = row_b * cx + ix;
        B[offset_b] = A[offset_a];
    }
}

template <typename AType, typename BType>
__global__ void RepeatKV_Kernel(const AType *A, int cx, int cy, int cz,
    bool a_is_zy_data, BType *B, int heads_per_group)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix < cx && iy < cy && iz < cz)
    {
        int row_a = a_is_zy_data ? (iy * cz + iz) : (iz * cy + iy);
        int offset_a = row_a * cx + ix;
        AType v = A[offset_a];
        for (int head_idx = 0; head_idx < heads_per_group; head_idx++)
        {
            int row_b = (iz * heads_per_group + head_idx) * cy + iy;
            int offset_b = row_b * cx + ix;
            B[offset_b] = v;
        }
    }
}

template <typename AType, typename BType>
__global__ void Tensor_AssignColumns_Kernel(const AType *A, BType *B,
    int cx_a, int cx_b, int rows, int step)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < cx_b && iy < rows)
    {
        int offset_a = iy * cx_a + ix * step;
        int offset_b = iy * cx_b + ix;
        B[offset_b] = A[offset_a];
    }
}

template <typename AType>
__global__ void Tensor_AssignColumns2_Kernel(AType *A, float value,
    int cx, int cy, int col_num)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < col_num && iy < cy)
    {
        int offset = iy * cx + ix;
        A[offset] = value;
    }
}

template <typename EType>
__global__ void Tensor_CheckElements_Kernel(int M, int N,
    const EType *data, int *invalid_count, int *invalid_pos)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float value = (float)data[row * N + col];
        if (isnan(value) || isinf(value))
        {
            atomicAdd(invalid_count, 1);
            atomicMin(invalid_pos, row * N + col);
        }
    }
}

} //end of namespace
