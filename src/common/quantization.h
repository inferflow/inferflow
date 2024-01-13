#pragma once

#include <vector>
#include "data_types.h"
#include "quant_types.h"

INFER_FLOW_BEGIN

using std::vector;

#ifdef __GNUC__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wcast-qual"
#endif

#define QUANT_MIN(x, y) (x <= y ? x : y)

class Quantization
{
public:
    static bool Quantize_Q8_Linear(vector<uint8_t> &quant_array,
        const vector<inferflow_fp16> &source, const LinearQuantParams &params);
    static bool Dequantize_Q8_Linear(vector<inferflow_fp16> &target,
        const vector<uint8_t> &quant_array, const LinearQuantParams &params);

    static bool Quantize_Q8_Log(vector<uint8_t> &quant_array,
        const vector<inferflow_fp16> &source, const LogQuantParams &params);
    static bool Dequantize_Q8_Log(vector<inferflow_fp16> &target,
        const vector<uint8_t> &quant_array, const LogQuantParams &params);

    static bool DequantizeRow_Q8_B32T1(float *target, const BlockQ8_B32T1 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q8_B32T2(float *target, const BlockQ8_B32T2 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q6_B64T1(float *target, const BlockQ6_B64T1 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q5(float *target, const BlockQ5_B32T1 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q4_B16(float *target, const BlockQ4_B16 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q4_B32T1(float *target, const BlockQ4_B32T1 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q3H_B64T1(float *target, const BlockQ3H_B64T1 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q3_B32T1(float *target, const BlockQ3_B32T1 *blocks, uint64_t block_count);
    static bool DequantizeRow_Q2_B32T1(float *target, const BlockQ2_B32T1 *blocks, uint64_t block_count);

    static HOST_AND_DEVICE uint8_t EncodeScale(float scale)
    {
        return uint8_t(scale * 1000 + 0.5f);
    }

    static HOST_AND_DEVICE float DecodeScale(uint8_t u8_scale)
    {
        return (float)u8_scale / 1000;
    }

    //range of base : [-1.5, 1.5]
    static HOST_AND_DEVICE uint8_t EncodeBase(float base)
    {
        return uint8_t(base * 100 + 100.5f);
    }

    static HOST_AND_DEVICE float DecodeBase(uint8_t u8_base)
    {
        return (int)(uint32_t)u8_base / 100.0f - 1.0f;
    }

    static HOST_AND_DEVICE float AdjustBase(float base)
    {
        uint8_t u8 = uint8_t(base * 100 + 100.01);
        return DecodeBase(u8);
    }

public:
    template <typename SourceType>
    static HOST_AND_DEVICE void GetValueRange(float *min_value, float *max_value,
        const SourceType *arr, int arr_len)
    {
        *min_value = (float)arr[0];
        *max_value = *min_value;
        for (int offset = 1; offset < arr_len; offset++)
        {
            float value = (float)arr[offset];
            if (*min_value > value) {
                *min_value = value;
            }
            if (*max_value < value) {
                *max_value = value;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q8_B32T1 (8-bit quantization, 32 data items per block, configuration T1)
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ8_B32T1(TargetType *target,
        const BlockQ8_B32T1 *block, int m = 1, int r = 0)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        for (int idx = 0; idx < Q4B32_CAPACITY; idx++)
        {
            if (idx % m == r)
            {
                const int q = block->data[idx];
                target[idx] = (TargetType)(q * scale + base);
            }
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q8_B32T1(BlockQ8_B32T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q8B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / ((1 << 8) - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            BlockQ8_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 min_value_f16 = (inferflow_fp16)min_value;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            target_block.base = *(const uint16_t*)&min_value_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity; ++r)
            {
                const float qf = ((float)source[start_offset + r] - min_value) * inv_scale;
                uint32_t q = (uint32_t)(qf + 0.5f);
                q = q > 255 ? 255 : q;
                target_block.data[r] = (uint8_t)q;
            }

            //if (is_debug_mode)
            //{
            //    printf("min: %f, max: %f, scale: %f, inv_scale: %f, base: (0x%X, %f), scale: (0x%X, %f)\n",
            //        min_value, max_value, scale, inv_scale, target_block.base,
            //        (float)min_value_f16, target_block.scale, (float)scale_f16);
            //}
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q8_B32T2 (8-bit quantization, 32 data items per block, configuration T2)
    ////////////////////////////////////////////////////////////////////////////

    static HOST_AND_DEVICE FORCE_INLINE int GetInt4(const BlockQ8_B32T2 &block, int start_offset)
    {
        int n = 0;
        n = *(uint16_t*)(block.data + start_offset);
        n <<= 16;
        n |= *(uint16_t*)(block.data + start_offset + 2);
        return n;
    }

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ8_B32T2(TargetType *target,
        const BlockQ8_B32T2 *block, int m = 1, int r = 0)
    {
        const float scale = (float)block->scale;

        for (int idx = 0; idx < Q4B32_CAPACITY; idx++)
        {
            if (idx % m == r)
            {
                const int q = (int)block->data[idx];
                target[idx] = (TargetType)(q * scale);
            }
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q8_B32T2(BlockQ8_B32T2 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q8B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float m1 = fabs(max_value), m2 = fabs(min_value);
            float m0 = m1 > m2 ? m1 : m2;
            float scale = m0 / ((1 << 7) - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            BlockQ8_B32T2 &target_block = blocks[block_idx];
            target_block.scale = (inferflow_fp16)scale;

            for (int r = 0; r < block_capacity; ++r)
            {
                const float qf = (float)source[start_offset + r] * inv_scale;
                int q = (int)round(qf);
                q = q > 127 ? 127 : q;
                q = q < -128 ? -128 : q;
                target_block.data[r] = (int8_t)q;
            }
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q6_B64T1 (6-bit quantization with block size 64)
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ6_B64T1(TargetType *target,
        const BlockQ6_B64T1 *block)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        // 16 parts, 4 values per part
        for (int idx = 0; idx < Q6_B64_CAPACITY / 4; idx++)
        {
            uint8_t qh = block->data_h[idx];
            const uint8_t qh0 = qh & 0x03;
            const uint8_t qh1 = (qh >> 2) & 0x03;
            const uint8_t qh2 = (qh >> 4) & 0x03;
            const uint8_t qh3 = (qh >> 6) & 0x03;

            const uint16_t qd = *(const uint16_t*)(block->data + 2 * idx);
            const int q0 = ((qd      ) & 0x0F) | (qh0 << 4);
            const int q1 = ((qd >>  4) & 0x0F) | (qh1 << 4);
            const int q2 = ((qd >>  8) & 0x0F) | (qh2 << 4);
            const int q3 = ((qd >> 12) & 0x0F) | (qh3 << 4);

            target[4 * idx    ] = (TargetType)(q0 * scale + base);
            target[4 * idx + 1] = (TargetType)(q1 * scale + base);
            target[4 * idx + 2] = (TargetType)(q2 * scale + base);
            target[4 * idx + 3] = (TargetType)(q3 * scale + base);
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q6_B64T1(BlockQ6_B64T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q6_B64_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        const int quant_num = (1 << 6) - 1; //63
        const uint32_t max_q = (1 << 6) - 1;
        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / (quant_num - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            BlockQ6_B64T1 &target_block = blocks[block_idx];
            inferflow_fp16 min_value_f16 = (inferflow_fp16)min_value;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            target_block.base = *(const uint16_t*)&min_value_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity / 4; ++r)
            {
                const float v0 = ((float)source[start_offset + 4 * r    ] - min_value) * inv_scale;
                const float v1 = ((float)source[start_offset + 4 * r + 1] - min_value) * inv_scale;
                const float v2 = ((float)source[start_offset + 4 * r + 2] - min_value) * inv_scale;
                const float v3 = ((float)source[start_offset + 4 * r + 3] - min_value) * inv_scale;
                const uint32_t q0 = QUANT_MIN((uint32_t)(v0 + 0.5f), max_q);
                const uint32_t q1 = QUANT_MIN((uint32_t)(v1 + 0.5f), max_q);
                const uint32_t q2 = QUANT_MIN((uint32_t)(v2 + 0.5f), max_q);
                const uint32_t q3 = QUANT_MIN((uint32_t)(v3 + 0.5f), max_q);

                const uint32_t qh = (q0 >> 4) | ((q1 & 0x30) >> 2)
                    | (q2 & 0x30) | ((q3 & 0x30) << 2);
                target_block.data_h[r] = (uint8_t)qh;

                target_block.data[2 * r    ] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                target_block.data[2 * r + 1] = (q2 & 0x0F) | ((q3 & 0x0F) << 4);
            }
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q5_B32T1
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ5Block(TargetType *target,
        const BlockQ5_B32T1 *block, int m = 1, int r = 0)
    {
        const float scale = (float)*(const inferflow_fp16*)block->scale;
        const float base = (float)*(const inferflow_fp16*)block->base;
        const uint32_t qh = *(const uint32_t*)block->h_data;

        for (int idx = 0; idx < Q5B32_CAPACITY / 2; idx++)
        {
            if (idx % m == r)
            {
                const uint8_t xh_0 = (qh >> (idx + 0)) & 0x01;
                const uint8_t xh_1 = (qh >> (idx + 16)) & 0x01;

                const int x0 = (block->data[idx] & 0x0F) | (xh_0 << 4);
                const int x1 = (block->data[idx] >> 4) | (xh_1 << 4);

                target[idx] = (TargetType)(x0 * scale + base);
                target[idx + Q5B32_CAPACITY / 2] = (TargetType)(x1 * scale + base);
            }
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeQ5Row(BlockQ5_B32T1 *blocks,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q5B32_CAPACITY;
        int block_num = source_len / block_capacity;
        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float delta = (max_value - min_value) / ((1 << 5) - 1);
            float inv_delta = delta >= 0.00001f ? (1.0f / delta) : 0.0f;
            //if (block_idx == 6 && is_debug_mode) {
            //    printf("min: %f, max: %f, delta: %f, inv_delta: %f\n", min_value, max_value, delta, inv_delta);
            //}

            BlockQ5_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 base_f16 = (inferflow_fp16)min_value;
            inferflow_fp16 scale_f16 = (inferflow_fp16)delta;

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            //memcpy(target_block.base, &base_f16, sizeof(inferflow_fp16));
            //memcpy(target_block.delta, &scale_f16, sizeof(inferflow_fp16));
            *(uint16_t*)target_block.base = *(const uint16_t*)&base_f16;
            *(uint16_t*)target_block.scale = *(const uint16_t*)&scale_f16;

            uint32_t &qh = *(uint32_t*)target_block.h_data;
            qh = 0;

            for (int r = 0; r < block_capacity / 2; ++r)
            {
                const float v1 = ((float)source[start_offset + r] - min_value) * inv_delta;
                const float v2 = ((float)source[start_offset + r + block_capacity / 2] - min_value) * inv_delta;

                const uint32_t q1 = (uint32_t)(v1 + 0.5f);
                const uint32_t q2 = (uint32_t)(v2 + 0.5f);
                target_block.data[r] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);

                qh |= (((q1 & 0x10) >> 4) << r);
                qh |= (((q2 & 0x10) >> 4) << (r + block_capacity / 2));
            }
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q4_B32T1
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ4_B32T1(TargetType *target,
        const BlockQ4_B32T1 *block, int m = 1, int r = 0)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        for (int idx = 0; idx < Q4B32_CAPACITY / 2; idx++)
        {
            if (idx % m == r)
            {
                const int x0 = (block->data[idx] & 0x0F);
                const int x1 = (block->data[idx] >> 4);

                target[2 * idx] = (TargetType)(x0 * scale + base);
                target[2 * idx + 1] = (TargetType)(x1 * scale + base);
            }
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q4_B32T1A(BlockQ4_B32T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q4B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            //printf("min_value: %f\n", min_value);
            float scale = (max_value - min_value) / ((1 << 4) - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            BlockQ4_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 min_value_f16 = (inferflow_fp16)min_value;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            //memcpy(target_block.base, &min_value_f16, sizeof(uint16_t));
            //memcpy(target_block.scale, &scale_f16, sizeof(uint16_t));
            target_block.base = *(const uint16_t*)&min_value_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity / 2; ++r)
            {
                const float v1 = ((float)source[start_offset + 2 * r + 0] - min_value) * inv_scale;
                const float v2 = ((float)source[start_offset + 2 * r + 1] - min_value) * inv_scale;

                uint32_t q1 = (uint32_t)(v1 + 0.5f);
                uint32_t q2 = (uint32_t)(v2 + 0.5f);
                q1 = q1 > 15 ? 15 : q1;
                q2 = q2 > 15 ? 15 : q2;
                target_block.data[r] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
            }

            //if (is_debug_mode)
            //{
            //    printf("min: %f, max: %f, scale: %f, inv_scale: %f, base: (0x%X, %f), scale: (0x%X, %f), data[0-3]: 0x%X\n",
            //        min_value, max_value, scale, inv_scale, target_block.base,
            //        (float)min_value_f16, target_block.scale, (float)scale_f16,
            //        *(const uint32_t*)target_block.data);
            //}
        }

        return true;
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q4_B32T1B(BlockQ4_B32T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q4B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / (1 << 4);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            float base = min_value + 0.5f * scale;
            BlockQ4_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 base_f16 = (inferflow_fp16)base;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            target_block.base = *(const uint16_t*)&base_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity / 2; ++r)
            {
                const float v1 = ((float)source[start_offset + 2 * r + 0] - min_value) * inv_scale;
                const float v2 = ((float)source[start_offset + 2 * r + 1] - min_value) * inv_scale;

                uint32_t q1 = (uint32_t)(v1 + 0.0001f);
                uint32_t q2 = (uint32_t)(v2 + 0.0001f);
                q1 = q1 > 15 ? 15 : q1;
                q2 = q2 > 15 ? 15 : q2;
                target_block.data[r] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
            }
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q4_B16T1
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ4_B16(TargetType *target,
        const BlockQ4_B16 *block, int m = 1, int r = 0)
    {
        const float scale = DecodeScale(block->scale);
        const float base = DecodeBase(block->base);

        for (int idx = 0; idx < Q4B16_CAPACITY / 2; idx++)
        {
            if (idx % m == r)
            {
                const int x0 = (block->data[idx] & 0x0F);
                const int x1 = (block->data[idx] >> 4); 

                target[2 * idx] = (TargetType)(x0 * scale + base);
                target[2 * idx + 1] = (TargetType)(x1 * scale + base);
            }
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q4B16(BlockQ4_B16 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q4B16_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            //printf("min_value: %f\n", min_value);
            min_value = AdjustBase(min_value);
            float scale = (max_value - min_value) / ((1 << 4) - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            //if (block_idx == 0 && is_debug_mode)
            //{
            //    printf("min: %f, max: %f, scale: %f, inv_scale: %f\n",
            //        min_value, max_value, scale, inv_scale);
            //}

            BlockQ4_B16 &target_block = blocks[block_idx];
            uint8_t u8_min_value = EncodeBase(min_value);
            uint8_t u8_scale = EncodeScale(scale);

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            //memcpy(&target_block.base, &u8_min_value, sizeof(uint8_t));
            //memcpy(&target_block.scale, &u8_scale, sizeof(uint8_t));
            target_block.base = u8_min_value;
            target_block.scale = u8_scale;

            for (int r = 0; r < block_capacity / 2; ++r)
            {
                const float v1 = ((float)source[start_offset + 2 * r + 0] - min_value) * inv_scale;
                const float v2 = ((float)source[start_offset + 2 * r + 1] - min_value) * inv_scale;

                uint32_t q1 = (uint32_t)(v1 + 0.5f);
                uint32_t q2 = (uint32_t)(v2 + 0.5f);
                q1 = q1 > 15 ? 15 : q1;
                q2 = q2 > 15 ? 15 : q2;
                target_block.data[r] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
            }
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q3H_B64T1 (3.5-bit quantization with block size 64)
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ3H_B64T1(TargetType *target,
        const BlockQ3H_B64T1 *block)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        // 8 parts, 4 data-pair items per part
        // Each data-pair corresponds to 7 bits in the block
        for (int idx = 0; idx < Q3H_B64_CAPACITY / 8; idx++)
        {
            uint16_t u16 = *(const uint16_t*)(block->data + 2 * idx);
            uint8_t m8 = block->data_m[idx];
            uint8_t h8 = idx % 2 == 0 ? (block->data_h[idx / 2] & 0x0F)
                : ((block->data_h[idx / 2] & 0xF0) >> 4);

            const int q0 = ((u16 & 0x000F)      ) | ((m8 & 0x03) << 4) | ((h8 & 0x01) << 6);
            const int q1 = ((u16 & 0x00F0) >>  4) | ((m8 & 0x0C) << 2) | ((h8 & 0x02) << 5);
            const int q2 = ((u16 & 0x0F00) >>  8) | ((m8 & 0x30)     ) | ((h8 & 0x04) << 4);
            const int q3 = ((u16 & 0xF000) >> 12) | ((m8 & 0xC0) >> 2) | ((h8 & 0x08) << 3);

            target[8 * idx + 0] = (TargetType)(q0 % 11 * scale + base);
            target[8 * idx + 1] = (TargetType)(q0 / 11 * scale + base);
            target[8 * idx + 2] = (TargetType)(q1 % 11 * scale + base);
            target[8 * idx + 3] = (TargetType)(q1 / 11 * scale + base);
            target[8 * idx + 4] = (TargetType)(q2 % 11 * scale + base);
            target[8 * idx + 5] = (TargetType)(q2 / 11 * scale + base);
            target[8 * idx + 6] = (TargetType)(q3 % 11 * scale + base);
            target[8 * idx + 7] = (TargetType)(q3 / 11 * scale + base);
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q3H_B64T1(BlockQ3H_B64T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q3H_B64_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        const int quant_num = 11;
        float v_arr[8];
        int q_arr[8];
        int q1 = 0, q2 = 0, q3 = 0, q4 = 0;
        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / (quant_num - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            BlockQ3H_B64T1 &target_block = blocks[block_idx];
            inferflow_fp16 min_value_f16 = (inferflow_fp16)min_value;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            target_block.base = *(const uint16_t*)&min_value_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            int data_h = 0;
            for (int r = 0; r < block_capacity / 8; ++r)
            {
                const int base_offset = start_offset + 8 * r;
                for (int idx = 0; idx < 8; idx++)
                {
                    v_arr[idx] = ((float)source[base_offset + idx] - min_value) * inv_scale;
                    q_arr[idx] = (int)(v_arr[idx] + 0.5f);
                    if (q_arr[idx] < 0) {
                        q_arr[idx] = 0;
                    }
                    if (q_arr[idx] > quant_num - 1) {
                        q_arr[idx] = quant_num - 1;
                    }
                }

                q1 = q_arr[0] + q_arr[1] * 11;
                q2 = q_arr[2] + q_arr[3] * 11;
                q3 = q_arr[4] + q_arr[5] * 11;
                q4 = q_arr[6] + q_arr[7] * 11;
                target_block.data[2 * r + 0] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
                target_block.data[2 * r + 1] = (q3 & 0x0F) | ((q4 & 0x0F) << 4);
                target_block.data_m[r] = ((q1 & 0x30) >> 4) | ((q2 & 0x30) >> 2)
                    | (q3 & 0x30) | ((q4 & 0x30) << 2);

                if (r % 2 == 0)
                {
                    data_h = ((q1 & 0x40) >> 6) | ((q2 & 0x40) >> 5)
                        | ((q3 & 0x40) >> 4) | ((q4 & 0x40) >> 3);
                }
                if (r % 2 == 1)
                {
                    data_h = data_h | ((q1 & 0x40) >> 2) | ((q2 & 0x40) >> 1)
                        | (q3 & 0x40) | ((q4 & 0x40) << 1);
                    target_block.data_h[r / 2] = (uint8_t)data_h;
                    data_h = 0;
                }
            }
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q3_B32T1 (3-bit quantization with block size 32)
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ3_B32T1(TargetType *target,
        const BlockQ3_B32T1 *block)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        // 4 parts, 8 data items (corresponding to 3 bytes in the block) per part
        for (int idx = 0; idx < Q3B32_CAPACITY / 8; idx++)
        {
            uint16_t u16 = *(const uint16_t*)(block->data + 2 * idx);
            uint8_t h8 = block->h_data[idx];
            const int q0 = ((u16 & 0x0003)      ) | ((h8 & 0x01) << 2);
            const int q1 = ((u16 & 0x000C) >>  2) | ((h8 & 0x02) << 1);
            const int q2 = ((u16 & 0x0030) >>  4) | ((h8 & 0x04)     );
            const int q3 = ((u16 & 0x00C0) >>  6) | ((h8 & 0x08) >> 1);
            const int q4 = ((u16 & 0x0300) >>  8) | ((h8 & 0x10) >> 2);
            const int q5 = ((u16 & 0x0C00) >> 10) | ((h8 & 0x20) >> 3);
            const int q6 = ((u16 & 0x3000) >> 12) | ((h8 & 0x40) >> 4);
            const int q7 = ((u16 & 0xC000) >> 14) | ((h8 & 0x80) >> 5);

            target[8 * idx + 0] = (TargetType)(q0 * scale + base);
            target[8 * idx + 1] = (TargetType)(q1 * scale + base);
            target[8 * idx + 2] = (TargetType)(q2 * scale + base);
            target[8 * idx + 3] = (TargetType)(q3 * scale + base);
            target[8 * idx + 4] = (TargetType)(q4 * scale + base);
            target[8 * idx + 5] = (TargetType)(q5 * scale + base);
            target[8 * idx + 6] = (TargetType)(q6 * scale + base);
            target[8 * idx + 7] = (TargetType)(q7 * scale + base);
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q3_B32T1A(BlockQ3_B32T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q3B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        float v_arr[8];
        uint32_t q_arr[8];
        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / ((1 << 3) - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            BlockQ3_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 min_value_f16 = (inferflow_fp16)min_value;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;
 
            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            target_block.base = *(const uint16_t*)&min_value_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity / 8; ++r)
            {
                for (int idx = 0; idx < 8; idx++)
                {
                    v_arr[idx] = ((float)source[start_offset + 8 * r + idx] - min_value) * inv_scale;
                    q_arr[idx] = (uint32_t)(v_arr[idx] + 0.5f);
                    if (q_arr[idx] > 7) {
                        q_arr[idx] = 7;
                    }
                }

                target_block.data[2 * r + 0] = (q_arr[0] & 0x03) | ((q_arr[1] & 0x03) << 2)
                    | ((q_arr[2] & 0x03) << 4) | ((q_arr[3] & 0x03) << 6);
                target_block.data[2 * r + 1] = (q_arr[4] & 0x03) | ((q_arr[5] & 0x03) << 2)
                    | ((q_arr[6] & 0x03) << 4) | ((q_arr[7] & 0x03) << 6);
                target_block.h_data[r] = ((q_arr[0] & 0x04) >> 2) | ((q_arr[1] & 0x04) >> 1)
                    | ((q_arr[2] & 0x04)     ) | ((q_arr[3] & 0x04) << 1) | ((q_arr[4] & 0x04) << 2)
                    | ((q_arr[5] & 0x04) << 3) | ((q_arr[6] & 0x04) << 4) | ((q_arr[7] & 0x04) << 5);
            }
        }

        return true;
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q3_B32T1B(BlockQ3_B32T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q3B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        float v_arr[8];
        uint32_t q_arr[8];
        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / (1 << 3);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            float base = min_value + 0.5f * scale;
            BlockQ3_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 base_f16 = (inferflow_fp16)base;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            target_block.base = *(const uint16_t*)&base_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity / 8; ++r)
            {
                for (int idx = 0; idx < 8; idx++)
                {
                    v_arr[idx] = ((float)source[start_offset + 8 * r + idx] - min_value) * inv_scale;
                    q_arr[idx] = (uint32_t)(v_arr[idx] + 0.0001f);
                    if (q_arr[idx] > 7) {
                        q_arr[idx] = 7;
                    }
                }

                target_block.data[2 * r + 0] = (q_arr[0] & 0x03) | ((q_arr[1] & 0x03) << 2)
                    | ((q_arr[2] & 0x03) << 4) | ((q_arr[3] & 0x03) << 6);
                target_block.data[2 * r + 1] = (q_arr[4] & 0x03) | ((q_arr[5] & 0x03) << 2)
                    | ((q_arr[6] & 0x03) << 4) | ((q_arr[7] & 0x03) << 6);
                target_block.h_data[r] = ((q_arr[0] & 0x04) >> 2) | ((q_arr[1] & 0x04) >> 1)
                    | ((q_arr[2] & 0x04)) | ((q_arr[3] & 0x04) << 1) | ((q_arr[4] & 0x04) << 2)
                    | ((q_arr[5] & 0x04) << 3) | ((q_arr[6] & 0x04) << 4) | ((q_arr[7] & 0x04) << 5);
            }
        }

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Q2_B32T1 (2-bit quantization with block size 32)
    ////////////////////////////////////////////////////////////////////////////

    template <typename TargetType>
    static HOST_AND_DEVICE void DequantizeQ2_B32T1(TargetType *target,
        const BlockQ2_B32T1 *block, int m = 1, int r = 0)
    {
        const float scale = (float)*(const inferflow_fp16*)&block->scale;
        const float base = (float)*(const inferflow_fp16*)&block->base;

        for (int idx = 0; idx < Q2B32_CAPACITY / 4; idx++)
        {
            if (idx % m == r)
            {
                const int q0 = (block->data[idx] & 0x03);
                const int q1 = ((block->data[idx] & 0x0C) >> 2);
                const int q2 = ((block->data[idx] & 0x30) >> 4);
                const int q3 = (block->data[idx] >> 6);

                target[4 * idx + 0] = (TargetType)(q0 * scale + base);
                target[4 * idx + 1] = (TargetType)(q1 * scale + base);
                target[4 * idx + 2] = (TargetType)(q2 * scale + base);
                target[4 * idx + 3] = (TargetType)(q3 * scale + base);
            }
        }
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q2_B32T1A(BlockQ2_B32T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q2B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / ((1 << 2) - 1);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            BlockQ2_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 min_value_f16 = (inferflow_fp16)min_value;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            //do NOT use memcpy here (using memcpy inside a CUDA kernel can cause unexpected behavior)
            target_block.base = *(const uint16_t*)&min_value_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity / 4; ++r)
            {
                const float v1 = ((float)source[start_offset + 4 * r + 0] - min_value) * inv_scale;
                const float v2 = ((float)source[start_offset + 4 * r + 1] - min_value) * inv_scale;
                const float v3 = ((float)source[start_offset + 4 * r + 2] - min_value) * inv_scale;
                const float v4 = ((float)source[start_offset + 4 * r + 3] - min_value) * inv_scale;

                uint32_t q1 = (uint32_t)(v1 + 0.5f);
                uint32_t q2 = (uint32_t)(v2 + 0.5f);
                uint32_t q3 = (uint32_t)(v3 + 0.5f);
                uint32_t q4 = (uint32_t)(v4 + 0.5f);
                q1 = q1 > 3 ? 3 : q1;
                q2 = q2 > 3 ? 3 : q2;
                q3 = q3 > 3 ? 3 : q3;
                q4 = q4 > 3 ? 3 : q4;
                target_block.data[r] = uint8_t(q1 | (q2 << 2) | (q3 << 4) | (q4 << 6));
            }
        }

        return true;
    }

    template <typename SourceType>
    static HOST_AND_DEVICE bool QuantizeRow_Q2_B32T1B(BlockQ2_B32T1 *blocks, int max_block_num,
        const SourceType *source, int source_len, bool is_debug_mode = false)
    {
        (void)is_debug_mode;
        const int block_capacity = Q2B32_CAPACITY;
        int block_num = source_len / block_capacity;
        if (block_num > max_block_num) {
            return false;
        }

        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int start_offset = block_idx * block_capacity;
            float min_value = 0, max_value = 0;
            GetValueRange(&min_value, &max_value, source + start_offset, block_capacity);

            float scale = (max_value - min_value) / (1 << 2);
            float inv_scale = scale >= 0.00001f ? (1.0f / scale) : 0.0f;
            float base = min_value + 0.5f * scale;
            BlockQ2_B32T1 &target_block = blocks[block_idx];
            inferflow_fp16 base_f16 = (inferflow_fp16)base;
            inferflow_fp16 scale_f16 = (inferflow_fp16)scale;

            target_block.base = *(const uint16_t*)&base_f16;
            target_block.scale = *(const uint16_t*)&scale_f16;

            for (int r = 0; r < block_capacity / 4; ++r)
            {
                const float v1 = ((float)source[start_offset + 4 * r + 0] - min_value) * inv_scale;
                const float v2 = ((float)source[start_offset + 4 * r + 1] - min_value) * inv_scale;
                const float v3 = ((float)source[start_offset + 4 * r + 2] - min_value) * inv_scale;
                const float v4 = ((float)source[start_offset + 4 * r + 3] - min_value) * inv_scale;

                uint32_t q1 = (uint32_t)(v1 + 0.0001f);
                uint32_t q2 = (uint32_t)(v2 + 0.0001f);
                uint32_t q3 = (uint32_t)(v3 + 0.0001f);
                uint32_t q4 = (uint32_t)(v4 + 0.0001f);
                q1 = q1 > 3 ? 3 : q1;
                q2 = q2 > 3 ? 3 : q2;
                q3 = q3 > 3 ? 3 : q3;
                q4 = q4 > 3 ? 3 : q4;
                target_block.data[r] = uint8_t(q1 | (q2 << 2) | (q3 << 4) | (q4 << 6));
            }
        }

        return true;
    }
};

#ifdef __GNUC__
#   pragma GCC diagnostic pop
#endif

INFER_FLOW_END
