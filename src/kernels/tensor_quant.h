#pragma once

#include "common/quant_cuda.h"

namespace inferflow
{

template <typename SourceType>
__global__ void Tensor_QuantizeQ8_B32T1_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ8_B32T1);
    const int block_capacity = Q8B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, block_id: %d\n", row, block_id);
        BlockQ8_B32T1 *block = (BlockQ8_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q8_B32T1<SourceType>(block, 1, source, block_capacity);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ8_B32T2_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ8_B32T2);
    const int block_capacity = Q8B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, block_id: %d\n", row, block_id);
        BlockQ8_B32T2 *block = (BlockQ8_B32T2*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q8_B32T2<SourceType>(block, 1, source, block_capacity);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ8_B32T2_Alg2_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    const int zi = threadIdx.z + blockIdx.z * blockDim.z;
    const int row = threadIdx.y + blockIdx.y * blockDim.y + zi * gridDim.y;
    const int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row >= row_num) {
        return;
    }

    const int block_idx = col / Q8B32_CAPACITY;
    const int idx_in_block = col % Q8B32_CAPACITY;
    //const int block_size = sizeof(BlockQ8_B32T2);
    BlockQ8_B32T2 *target_block = ((BlockQ8_B32T2*)(B + row * b_bytes_per_row)) + block_idx;

    const float value = col < col_num ? (float)A[row * col_num + col] : 0.0f;
    float max_value = fabs(value);
    //float sum = value;

    max_value = WarpReduceMax(max_value);
    //sum = WarpReduceSumAll(sum);

    const float scale = max_value / (128 - 1);
    int q = scale <= 0.000001f ? 0 : roundf(value / scale);
    //printf("row: %d, col: %d, value: %.3f, max: %.3f, sum: %.3f, scale: %.3f, q: %d\n",
    //    row, col, value, max_value, sum, scale, (int)q);

    if (row < row_num && col < col_num)
    {
        target_block->data[idx_in_block] = (int8_t)min(max(q, -128), 127);
        if (idx_in_block == 0)
        {
            target_block->scale = (inferflow_fp16)scale;
            //printf("row: %d, col: %d, scale: %d\n", row, col, target_block->scale);
        }
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ6_B64T1_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ6_B64T1);
    const int block_capacity = Q6_B64_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        auto *block = (BlockQ6_B64T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q6_B64T1<SourceType>(block, 1, source, block_capacity, false);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ5_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ5_B32T1);
    const int block_capacity = Q5B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, block_id: %d\n", row, block_id);
        auto *block = (BlockQ5_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeQ5Row<SourceType>(block, source, block_capacity);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ4B16_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ4_B16);
    const int block_capacity = Q4B16_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, block_id: %d, block_size: %d, block_capacity: %d, blocks_per_row: %d\n",
        //    row, block_id, block_size, block_capacity, blocks_per_row);
        BlockQ4_B16 *block = (BlockQ4_B16*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q4B16<SourceType>(block, 1, source, block_capacity, false);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ4_B32T1A_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ4_B32T1);
    const int block_capacity = Q4B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, block_id: %d, block_size: %d, block_capacity: %d, blocks_per_row: %d\n",
        //    row, block_id, block_size, block_capacity, blocks_per_row);
        BlockQ4_B32T1 *block = (BlockQ4_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q4_B32T1A<SourceType>(block, 1, source, block_capacity, false);

        //printf("block base: 0x%X, scale: 0x%X, data[0-3]: 0x%X\n",
        //    block->base, block->scale, *(const uint32_t*)block->data);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ4_B32T1B_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ4_B32T1);
    const int block_capacity = Q4B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, block_id: %d, block_size: %d, block_capacity: %d, blocks_per_row: %d\n",
        //    row, block_id, block_size, block_capacity, blocks_per_row);
        BlockQ4_B32T1 *block = (BlockQ4_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q4_B32T1B<SourceType>(block, 1, source, block_capacity, false);

        //printf("block base: 0x%X, scale: 0x%X, data[0-3]: 0x%X\n",
        //    block->base, block->scale, *(const uint32_t*)block->data);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ3H_B64T1_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ3H_B64T1);
    const int block_capacity = Q3H_B64_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        auto *block = (BlockQ3H_B64T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q3H_B64T1<SourceType>(block, 1, source, block_capacity, false);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ3_B32T1A_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ3_B32T1);
    const int block_capacity = Q3B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        BlockQ3_B32T1 *block = (BlockQ3_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q3_B32T1A<SourceType>(block, 1, source, block_capacity, false);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ3_B32T1B_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ3_B32T1);
    const int block_capacity = Q3B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        BlockQ3_B32T1 *block = (BlockQ3_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q3_B32T1B<SourceType>(block, 1, source, block_capacity, false);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ2_B32T1A_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ2_B32T1);
    const int block_capacity = Q2B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        BlockQ2_B32T1 *block = (BlockQ2_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q2_B32T1A<SourceType>(block, 1, source, block_capacity, false);
    }
}

template <typename SourceType>
__global__ void Tensor_QuantizeQ2_B32T1B_Kernel(const SourceType *A, uint8_t *B,
    int row_num, int col_num, int b_bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ2_B32T1);
    const int block_capacity = Q2B32_CAPACITY;

    if (row < row_num && block_id < blocks_per_row)
    {
        BlockQ2_B32T1 *block = (BlockQ2_B32T1*)(B + row * b_bytes_per_row + block_id * block_size);
        const SourceType *source = A + (row * blocks_per_row + block_id) * block_capacity;
        DeviceQuantization::QuantizeRow_Q2_B32T1B<SourceType>(block, 1, source, block_capacity, false);
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ8_B32T1_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ8_B32T1);
    const int block_capacity = Q8B32_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, r: %d, bytes_per_row: %d, blocks_per_row: %d\n",
        //    row, r, bytes_per_row, blocks_per_row);
        const auto *block = (const BlockQ8_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v = target[r];
        DeviceQuantization::DequantizeQ8_B32T1(v, block, r);
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ8_B32T2_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ8_B32T2);
    const int block_capacity = Q8B32_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, r: %d, bytes_per_row: %d, blocks_per_row: %d\n",
        //    row, r, bytes_per_row, blocks_per_row);
        const auto *block = (const BlockQ8_B32T2*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v = target[r];
        DeviceQuantization::DequantizeQ8_B32T2(v, block, r);
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ6_B64T1_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ6_B64T1);
    const int block_capacity = Q6_B64_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        const auto *block = (const BlockQ6_B64T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v1 = target[4 * r];
        TargetType &v2 = target[4 * r + 1];
        TargetType &v3 = target[4 * r + 2];
        TargetType &v4 = target[4 * r + 3];
        DeviceQuantization::DequantizeQ6_B64T1(v1, v2, v3, v4, block, r);
    }
}

__global__ void Tensor_DequantizeQ5_Half_Kernel(const uint8_t *A, half *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ5_B32T1);
    const int block_capacity = Q5B32_CAPACITY;

    //if (row == 0)
    //{
    //    printf("blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d\n", blockDim.x, blockIdx.x, threadIdx.x);
    //}

    Float16 target_f16;
    if (row < row_num && block_id < blocks_per_row)
    {
        BlockQ5_B32T1 block = *(const BlockQ5_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        //Quantization::DequantizeQ5Block<TargetType>(target, block, blockDim.x, threadIdx.x);
        Quantization::DequantizeQ5Block<half>((half*)&target_f16, &block);
        *(Float16*)(B + (row * blocks_per_row + block_id) * block_capacity) = target_f16;
        //float delta = (float)block->delta;
        //float base = (float)block->base;
        //target[threadIdx.x] = base;
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ5_Alg2_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ5_B32T1);
    const int block_capacity = Q5B32_CAPACITY;

    int r = threadIdx.x;
    const int delta = block_capacity / 2;

    if (row < row_num && block_id < blocks_per_row)
    {
        const BlockQ5_B32T1 *block = (const BlockQ5_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v1 = target[r];
        TargetType &v2 = target[r + delta];
        DeviceQuantization::DequantizeQ5(v1, v2, block, r);
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ5_Alg2_Transpose_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ5_B32T1);
    const int block_capacity = Q5B32_CAPACITY;

    int r = threadIdx.x;
    const int delta = block_capacity / 2;

    if (row < row_num && block_id < blocks_per_row)
    {
        const BlockQ5_B32T1 *block = (const BlockQ5_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType &v1 = B[(block_id * block_capacity + r) * row_num + row];
        TargetType &v2 = B[(block_id * block_capacity + r + delta) * row_num + row];
        DeviceQuantization::DequantizeQ5(v1, v2, block, r);
    }
}

__global__ void Tensor_DequantizeQ4_Half_Kernel(const uint8_t *A, half *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_size = sizeof(BlockQ4_B32T1);
    const int block_capacity = Q4B32_CAPACITY;

    Float16 target_f16;
    if (row < row_num && block_id < blocks_per_row)
    {
        BlockQ4_B32T1 block = *(const BlockQ4_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        Quantization::DequantizeQ4_B32T1<half>((half*)&target_f16, &block);
        *(Float16*)(B + (row * blocks_per_row + block_id) * block_capacity) = target_f16;
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ4B16_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ4_B16);
    const int block_capacity = Q4B16_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, r: %d, bytes_per_row: %d, blocks_per_row: %d\n",
        //    row, r, bytes_per_row, blocks_per_row);
        const BlockQ4_B16 *block = (const BlockQ4_B16*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v1 = target[2 * r];
        TargetType &v2 = target[2 * r + 1];
        DeviceQuantization::DequantizeQ4_B16(v1, v2, block, r);
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ4_B32T1_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ4_B32T1);
    const int block_capacity = Q4B32_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        //printf("row: %d, r: %d, bytes_per_row: %d, blocks_per_row: %d\n",
        //    row, r, bytes_per_row, blocks_per_row);
        const BlockQ4_B32T1 *block = (const BlockQ4_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v1 = target[2 * r];
        TargetType &v2 = target[2 * r + 1];
        DeviceQuantization::DequantizeQ4_B32T1(v1, v2, block, r);
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ3H_B64T1_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ3H_B64T1);
    const int block_capacity = Q3H_B64_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        const auto *block = (const BlockQ3H_B64T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v1 = target[4 * r];
        TargetType &v2 = target[4 * r + 1];
        TargetType &v3 = target[4 * r + 2];
        TargetType &v4 = target[4 * r + 3];
        DeviceQuantization::DequantizeQ3H_B64T1(v1, v2, v3, v4, block, r);
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ3_B32T1_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ3_B32T1);
    const int block_capacity = Q3B32_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        const BlockQ3_B32T1 *block = (const BlockQ3_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v1 = target[4 * r];
        TargetType &v2 = target[4 * r + 1];
        TargetType &v3 = target[4 * r + 2];
        TargetType &v4 = target[4 * r + 3];
        DeviceQuantization::DequantizeQ3_B32T1(v1, v2, v3, v4, block, r);
        //if (row == 0)
        //{
        //    printf("row: %d, r: %d, bytes_per_row: %d, blocks_per_row: %d, values: [%f, %f, %f, %f]\n",
        //        row, r, bytes_per_row, blocks_per_row, (float)v1, (float)v2, (float)v3, (float)v4);
        //}
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ2_B32T1_Kernel(const uint8_t *A, TargetType *B,
    int row_num, int col_num, int bytes_per_row, int blocks_per_row)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int block_id = blockIdx.x;
    const int block_size = sizeof(BlockQ2_B32T1);
    const int block_capacity = Q2B32_CAPACITY;

    int r = threadIdx.x;
    if (row < row_num && block_id < blocks_per_row)
    {
        const BlockQ2_B32T1 *block = (const BlockQ2_B32T1*)(A + row * bytes_per_row + block_id * block_size);
        TargetType *target = B + (row * blocks_per_row + block_id) * block_capacity;
        TargetType &v1 = target[4 * r];
        TargetType &v2 = target[4 * r + 1];
        TargetType &v3 = target[4 * r + 2];
        TargetType &v4 = target[4 * r + 3];
        DeviceQuantization::DequantizeQ2_B32T1(v1, v2, v3, v4, block, r);
        //if (row == 0)
        //{
        //    printf("row: %d, r: %d, bytes_per_row: %d, blocks_per_row: %d, values: [%f, %f, %f, %f]\n",
        //        row, r, bytes_per_row, blocks_per_row, (float)v1, (float)v2, (float)v3, (float)v4);
        //}
    }
}

__global__ void Tensor_DequantizeQ8_S4_Kernel(const uint32_t *A, uint64_t *B,
    int cx, int cy, float base, float delta1, float delta2, const half *quant_map)
{
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int stride = 4;
    uint32_t source = 0;
    uint64_t target = 0;
    const uint8_t *src_array = (const uint8_t*)&source;
    half *dst_array = (half*)&target;

    __shared__ half shared_quant_map[256];

    bool has_quant_map = quant_map != nullptr;
    if (has_quant_map)
    {
        int block_size = blockDim.x * blockDim.y;
        int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
        shared_quant_map[thread_idx] = quant_map[thread_idx];
        if (thread_idx + block_size < 256) {
            shared_quant_map[thread_idx + block_size] = quant_map[thread_idx + block_size];
        }

        __syncthreads();
    }

    if (x_idx < cx && y_idx < cy)
    {
        int idx = y_idx * cx + x_idx;
        source = A[idx];

        if (has_quant_map)
        {
#           pragma unroll
            for (int idx = 0; idx < stride; idx++)
            {
                uint8_t q = src_array[idx];
                dst_array[idx] = shared_quant_map[q];
            }
        }
        else
        {
#           pragma unroll
            for (int idx = 0; idx < stride; idx++)
            {
                uint8_t q = src_array[idx];
                if (q >= 128) {
                    dst_array[idx] = (half)(base - (int)(q - 128) * delta2);
                }
                else {
                    dst_array[idx] = (half)(base + (int)q * delta1);
                }
            }
        }

        B[idx] = target;
    }
}

__global__ void Tensor_DequantizeQ8_S8_Kernel(const uint64_t *A, uint64_t *B,
    int cx, int cy, float base, float delta1, float delta2, const half *quant_map)
{
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int stride = 8;
    uint64_t source = 0;
    uint64_t target[2];
    const uint8_t *src_array = (const uint8_t*)&source;
    half *dst_array = (half*)target;

    __shared__ half shared_quant_map[256];

    bool has_quant_map = quant_map != nullptr;
    if (has_quant_map)
    {
        int block_size = blockDim.x * blockDim.y;
        int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
        shared_quant_map[thread_idx] = quant_map[thread_idx];
        if (thread_idx + block_size < 256) {
            shared_quant_map[thread_idx + block_size] = quant_map[thread_idx + block_size];
        }

        __syncthreads();
    }

    if (x_idx < cx && y_idx < cy)
    {
        int idx = y_idx * cx + x_idx;
        source = A[idx];

        if (has_quant_map)
        {
#           pragma unroll
            for (int idx = 0; idx < stride; idx++)
            {
                uint8_t q = src_array[idx];
                dst_array[idx] = shared_quant_map[q];
            }
        }
        else
        {
#           pragma unroll
            for (int idx = 0; idx < stride; idx++)
            {
                uint8_t q = src_array[idx];
                if (q >= 128) {
                    dst_array[idx] = (half)(base - (int)(q - 128) * delta2);
                }
                else {
                    dst_array[idx] = (half)(base + (int)q * delta1);
                }
            }
        }

        B[2 * idx] = target[0];
        B[2 * idx + 1] = target[1];
    }
}

__global__ void Tensor_DequantizeQ8_S16_Kernel(const uint64_t *A, uint64_t *B,
    int cx, int cy, float base, float delta1, float delta2)
{
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int stride = 16;
    uint64_t source[2];
    uint64_t target[4];
    const uint8_t *src_array = (const uint8_t*)source;
    half *dst_array = (half*)target;

    if (x_idx < cx && y_idx < cy)
    {
        int idx = y_idx * cx + x_idx;
        source[0] = A[2 * idx];
        source[1] = A[2 * idx + 1];

#       pragma unroll
        for (int idx = 0; idx < stride; idx++)
        {
            uint8_t q = src_array[idx];
            if (q >= 128) {
                dst_array[idx] = (half)(base - (int)(q - 128) * delta2);
            }
            else {
                dst_array[idx] = (half)(base + (int)q * delta1);
            }
        }

        B[4 * idx] = target[0];
        B[4 * idx + 1] = target[1];
        B[4 * idx + 2] = target[2];
        B[4 * idx + 3] = target[3];
    }
}

template <typename TargetType>
__global__ void Tensor_DequantizeQ8_Kernel(const uint8_t *A, TargetType *B,
    int cx, int cy, int stride_x, float base, float delta1, float delta2)
{
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int x_idx = (threadIdx.x + blockIdx.x * blockDim.x) * stride_x;

    //uint32_t source = 0;
    //uint8_t source_array[8];
    //half target_array[8];

    if (x_idx < cx && y_idx < cy)
    {
        //int start_idx = y_idx * cx + x_idx;
/*
#       pragma unroll
        for (int idx = 0; idx < stride_x; idx++)
        {
            uint8_t q = souce_array[idx];
            if (q >= 128) {
                target_array[idx] = (TargetType)(base - (int)(q - 128) * delta2);
            }
            else {
                target_array[idx] = (TargetType)(base + (int)q * delta1);
            }
        }*/
    }
}

__global__ void Tensor_DequantizeQ8_Log_S8_Kernel(const uint64_t *A, uint64_t *B,
    int cx, int cy, float param_base, int param_scale, int param_start,
    const half *quant_map)
{
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int stride = 8;
    uint64_t source = 0;
    uint64_t target[2];
    const uint8_t *src_array = (const uint8_t*)&source;
    half *dst_array = (half*)target;

    __shared__ half shared_quant_map[256];

    bool has_quant_map = quant_map != nullptr;
    if (has_quant_map)
    {
        int block_size = blockDim.x * blockDim.y;
        int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
        shared_quant_map[thread_idx] = quant_map[thread_idx];
        if (thread_idx + block_size < 256) {
            shared_quant_map[thread_idx + block_size] = quant_map[thread_idx + block_size];
        }

        __syncthreads();
    }

    if (x_idx < cx && y_idx < cy)
    {
        int idx = y_idx * cx + x_idx;
        source = A[idx];

        if (has_quant_map)
        {
#           pragma unroll
            for (int idx = 0; idx < stride; idx++)
            {
                uint8_t q = src_array[idx];
                dst_array[idx] = shared_quant_map[q];
            }
        }
        else
        {
            float v = 0;
#           pragma unroll
            for (int idx = 0; idx < stride; idx++)
            {
                uint8_t q = src_array[idx];
                int sign = q >= 128 ? 1 : -1;
                int num = q >= 128 ? (q - 128) : (128 - q);
                if (num >= param_start) {
                    v = pow(param_base, num - param_start) / param_scale;
                }
                else {
                    v = num / param_scale;
                }
                dst_array[idx] = (half)(v * sign);
            }
        }

        B[2 * idx] = target[0];
        B[2 * idx + 1] = target[1];
    }
}

} //end of namespace
