#pragma once

#include "kernels_common.h"

namespace inferflow
{

__global__ void VectorDotProduct_Kernel(float *V3, const float *V1, const float *V2,
    int vec_size, int cell_len)
{
    __shared__ float block_cache[Inferflow_MaxThreadPerBlock];

    const int start_idx = (blockDim.x * blockIdx.x + threadIdx.x) * cell_len;
    const int end_idx = start_idx + cell_len;
    const int cache_index = threadIdx.x;

    //printf("threadIdx: (%d, %d), blockIdx: (%d, %d), blockDim: (%d, %d), gridDim: (%d, %d), cell_len: %d\n",
    //    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
    //    blockDim.x, blockDim.y, gridDim.x, gridDim.y, cell_len);

    float acc = 0;
    for (int idx = start_idx; idx < end_idx && idx < vec_size; idx++)
    {
        acc += V1[idx] * V2[idx];
        //idx += blockDim.x * gridDim.x;
    }

    block_cache[cache_index] = acc;

    __syncthreads();

    if (cache_index == 0)
    {
        acc = 0;
        for (int idx = 0; idx < blockDim.x; idx++)
        {
            acc += block_cache[idx];
        }
        V3[blockIdx.x] = acc;
    }

    /*int idx  = blockDim.x / 2;
    while (idx != 0)
    {
        if (cache_index < idx) {
            block_cache[cache_index] += block_cache[cache_index + idx];
        }

        __syncthreads();
        idx /= 2;
    }

    if (cache_index == 0) {
        V3[blockIdx.x] = block_cache[0];
    }*/
}

} //end of namespace

