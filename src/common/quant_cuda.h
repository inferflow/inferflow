#pragma once

#include "common/cuda_util.h"
#include "quantization.h"

INFER_FLOW_BEGIN

class DeviceQuantization : public Quantization
{
public:
    template <typename TargetType>
    static __host__ __device__ void DequantizeQ8_B32T1(TargetType &v,
        const BlockQ8_B32T1 *block, int q_idx = 0)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        const int q = block->data[q_idx];
        v = (TargetType)(q * scale + base);
    }

    template <typename TargetType>
    static __host__ __device__ void DequantizeQ8_B32T2(TargetType &v,
        const BlockQ8_B32T2 *block, int q_idx = 0)
    {
        const float scale = (float)block->scale;

        const int q = (int)block->data[q_idx];
        v = (TargetType)(q * scale);
    }

    template <typename TargetType>
    static __host__ __device__ void DequantizeQ5(TargetType &v1, TargetType &v2,
        const BlockQ5_B32T1 *block, int r = 0)
    {
        const float scale = (float)*(half*)block->scale;
        const float base = (float)*(half*)block->base;
        const uint32_t qh = *(const uint32_t*)block->h_data;

        const uint8_t xh_1 = (qh >> (r +  0)) & 0x01;
        const uint8_t xh_2 = (qh >> (r + 16)) & 0x01;

        const int x1 = (block->data[r] & 0x0F) | (xh_1 << 4);
        const int x2 = (block->data[r] >>   4) | (xh_2 << 4);

        v1 = (TargetType)(x1 * scale + base);
        v2 = (TargetType)(x2 * scale + base);
    }

    template <typename TargetType>
    static __host__ __device__ void DequantizeQ4_B16(TargetType &v1, TargetType &v2,
        const BlockQ4_B16 *block, int r = 0)
    {
        const float scale = DecodeScale(block->scale);
        const float base = DecodeBase(block->base);

        const int x1 = (block->data[r] & 0x0F);
        const int x2 = (block->data[r] >> 4);

        v1 = (TargetType)(x1 * scale + base);
        v2 = (TargetType)(x2 * scale + base);
    }

    template <typename TargetType>
    static __host__ __device__ void DequantizeQ4_B32T1(TargetType &v1, TargetType &v2,
        const BlockQ4_B32T1 *block, int r = 0)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        const int x1 = (block->data[r] & 0x0F);
        const int x2 = (block->data[r] >> 4);

        v1 = (TargetType)(x1 * scale + base);
        v2 = (TargetType)(x2 * scale + base);
    }

    // r: [0, 15]
    template <typename TargetType>
    static __host__ __device__ void DequantizeQ3H_B64T1(TargetType &v1, TargetType &v2,
        TargetType &v3, TargetType &v4, const BlockQ3H_B64T1 *block, int r)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        uint8_t d8 = block->data[r];
        uint8_t m8 = r % 2 == 0 ? block->data_m[r / 2] : (block->data_m[r / 2] >> 4);
        uint8_t h8 = (block->data_h[r / 4] >> (r % 4 * 2));
        const int q1 = (d8 & 0x0F) | ((m8 & 0x03) << 4) | ((h8 & 0x01) << 6);
        const int q2 = ((d8 & 0xF0) >> 4) | ((m8 & 0x0C) << 2) | ((h8 & 0x02) << 5);

        v1 = (TargetType)((q1 % 11) * scale + base);
        v2 = (TargetType)((q1 / 11) * scale + base);
        v3 = (TargetType)((q2 % 11) * scale + base);
        v4 = (TargetType)((q2 / 11) * scale + base);
    }

    template <typename TargetType>
    static __host__ __device__ void DequantizeQ3_B32T1(TargetType &v1, TargetType &v2,
        TargetType &v3, TargetType &v4, const BlockQ3_B32T1 *block, int r = 0)
    { // range of r: [0, 7]
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        uint8_t h8 = r % 2 == 0 ? block->h_data[r / 2] : (block->h_data[r / 2] >> 4);
        const int q1 = ((block->data[r] & 0x03)     ) | ((h8 & 0x01) << 2);
        const int q2 = ((block->data[r] & 0x0C) >> 2) | ((h8 & 0x02) << 1);
        const int q3 = ((block->data[r] & 0x30) >> 4) | ((h8 & 0x04)     );
        const int q4 = ((block->data[r] & 0xC0) >> 6) | ((h8 & 0x08) >> 1);

        v1 = (TargetType)(q1 * scale + base);
        v2 = (TargetType)(q2 * scale + base);
        v3 = (TargetType)(q3 * scale + base);
        v4 = (TargetType)(q4 * scale + base);
    }

    template <typename TargetType>
    static __host__ __device__ void DequantizeQ2_B32T1(TargetType &v1, TargetType &v2,
        TargetType &v3, TargetType &v4, const BlockQ2_B32T1 *block, int r = 0)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        const int q1 = (block->data[r] & 0x03);
        const int q2 = ((block->data[r] & 0x0C) >> 2);
        const int q3 = ((block->data[r] & 0x30) >> 4);
        const int q4 = (block->data[r] >> 6);

        v1 = (TargetType)(q1 * scale + base);
        v2 = (TargetType)(q2 * scale + base);
        v3 = (TargetType)(q3 * scale + base);
        v4 = (TargetType)(q4 * scale + base);
    }

    static bool DequantizeQ5Row(half *target, const BlockQ5_B32T1 *blocks, uint64_t block_count);
};

INFER_FLOW_END
