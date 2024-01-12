#pragma once

#include "common/cuda_util.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 8   // BLOCK_COLS / WMMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / WMMA_M

#define WARP_ROW_TILES 4  // WARP_COLS / WMMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / WMMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / WMMA_K

#define CHUNK_LINE_BYTES 64          // CHUNK_K * WMMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * sizeof(int4) / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * WMMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

#define Inferflow_MaxThreadPerBlock 256

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ float WarpReduceSum(float sum, int warp_size = 32)
{
//#   pragma unroll
    //for (int mask = 16; mask > 0; mask >>= 1) {
    //    sum += __shfl_xor_sync(0xffffffff, sum, mask, warp_size);
    //}

    if (warp_size >= 32) {
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    }
    if (warp_size >= 16) {
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    }
    if (warp_size >= 8) {
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    }
    if (warp_size >= 4) {
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    }
    if (warp_size >= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    return sum;
}

__device__ __forceinline__ half WarpReduceSum(half sum, int warp_size = 32)
{
    if (warp_size >= 32) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 16));
    }
    if (warp_size >= 16) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 8));
    }
    if (warp_size >= 8) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 4));
    }
    if (warp_size >= 4) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 2));
    }
    if (warp_size >= 2) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 1));
    }

    return sum;
}

__device__ __forceinline__ float WarpReduceSumAll(float sum)
{
#   pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    return sum;
}

__device__ __forceinline__ half WarpReduceSumAll(half sum)
{
#   pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum = __hadd(sum, __shfl_xor_sync(0xffffffff, sum, mask));
    }

    return sum;
}

__device__ __forceinline__ float WarpReduceMax(float v, int warp_size = 32)
{
#   pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v = max(v, __shfl_xor_sync(0xffffffff, v, mask, warp_size));
    }

    return v;
}
