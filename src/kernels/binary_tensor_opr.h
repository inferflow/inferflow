#pragma once

namespace inferflow
{

template <typename AType, typename BType, typename CType>
__global__ void ElementwiseAdd_Alg1_Kernel(int M1, int M2, int N,
    AType const *A, BType const *B, CType *C)
{
    int c_row = threadIdx.y + blockIdx.y * blockDim.y;
    int c_col = threadIdx.x + blockIdx.x * blockDim.x;

    if (c_row < M1 && c_col < N)
    {
        int offset1 = c_row * N + c_col;
        int offset2 = (c_row % M2) * N + c_col;
        C[offset1] = (float)A[offset1] + (float)B[offset2];
    }
}

template <typename AType, typename BType, typename CType>
__global__ void ElementwiseAdd_Alg2_Kernel(CType *C, AType const *A, BType const *B,
    int M1, int M2, int N, int tile_dim, int block_rows)
{
    int c_row = threadIdx.y + blockIdx.y * blockDim.y;
    int c_col = threadIdx.x + blockIdx.x * blockDim.x;

#   pragma unroll
    for (int j = 0; j < tile_dim; j += block_rows)
    {
        if (c_row + j < M1 && c_col < N)
        {
            int offset1 = (c_row + j) * N + c_col;
            int offset2 = ((c_row + j) % M2) * N + c_col;
            C[offset1] = (float)B[offset1] + (float)A[offset2];
        }
    }
}

template <typename DataType>
__global__ void ElementwiseAdd_Alg3_Half_Kernel(DataType *C, DataType const *A, DataType const *B,
    int M1, int M2, int N, int tile_len)
{
    int c_row = threadIdx.y + blockIdx.y * blockDim.y;
    int start_col = (threadIdx.x + blockIdx.x * blockDim.x) * tile_len;
    int end_col = min(start_col + tile_len, N);

    if (c_row >= M1 || start_col >= end_col) {
        return;
    }

    if (tile_len == 4 && start_col + tile_len == end_col)
    {
        int offset1 = c_row * N + start_col;
        int offset2 = c_row % M2 * N + start_col;
        float2 v1 = *(float2*)(A + offset1);
        float2 v2 = *(float2*)(B + offset2);

        float2 target_v;
        const half *ha = (const half*)&v1;
        const half *hb = (const half*)&v2;
        half *hc = (half*)&target_v;
        hc[0] = ha[0] + hb[0];
        hc[1] = ha[1] + hb[1];
        hc[2] = ha[2] + hb[2];
        hc[3] = ha[3] + hb[3];
        *(float2*)(C + offset1) = target_v;
        return;
    }

#   pragma unroll
    for (int col = start_col; col < end_col; col++)
    {
        int offset1 = c_row * N + col;
        int offset2 = c_row % M2 * N + col;
        C[offset1] = B[offset1] + A[offset2];
    }
}

template <typename AType, typename BType, typename CType>
__global__ void ElementwiseMul_Alg1_Kernel(int M1, int M2, int N,
    AType const *A, BType const *B, CType *C)
{
    int c_row = threadIdx.y + blockIdx.y * blockDim.y;
    int c_col = threadIdx.x + blockIdx.x * blockDim.x;

    if (c_row < M1 && c_col < N)
    {
        int offset1 = c_row * N + c_col;
        int offset2 = (c_row % M2) * N + c_col;
        C[offset1] = (float)A[offset1] * (float)B[offset2];
    }
}

template <typename AType, typename CType>
__global__ void Tensor_Scale_Kernel(int M, int N, const AType *A, float scale, CType *C)
{
    int c_row = threadIdx.y + blockIdx.y * blockDim.y;
    int c_col = threadIdx.x + blockIdx.x * blockDim.x;

    if (c_row < M && c_col < N)
    {
        int offset = c_row * N + c_col;
        C[offset] = (float)A[offset] * scale;
    }
}

template <typename AType, typename ScaleType, typename CType>
__global__ void Tensor_ScaleV_Kernel(int M, int N, const AType *A,
    const ScaleType *scale, CType *C, bool is_reverse)
{
    int c_row = threadIdx.y + blockIdx.y * blockDim.y;
    int c_col = threadIdx.x + blockIdx.x * blockDim.x;

    if (c_row < M && c_col < N)
    {
        int offset = c_row * N + c_col;
        if (is_reverse) {
            C[offset] = (float)A[offset] / (float)scale[c_row];
        }
        else {
            C[offset] = (float)A[offset] * (float)scale[c_row];
        }
    }
}

} //end of namespace

