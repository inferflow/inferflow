#include "quantization.h"
#include <cmath>
#include <algorithm>

INFER_FLOW_BEGIN

using namespace std;

//static
bool Quantization::Quantize_Q8_Linear(vector<uint8_t> &quant_array,
    const vector<inferflow_fp16> &source, const LinearQuantParams &params)
{
    int size = (int)source.size();
    quant_array.resize(size);

    float z = params.z;
    float scale1 = params.scale1;
    float scale2 = params.scale2;

    int q = 0;
    for (int idx = 0; idx < size; idx++)
    {
        float v = source[idx];
        if (v >= z)
        {
            q = (int)((v - z + scale1 / 2) / scale1);
            quant_array[idx] = (uint8_t)min(127, q);
        }
        else
        {
            q = (int)((z - v + scale2 / 2) / scale2);
            quant_array[idx] = 128 + (uint8_t)min(127, q);
        }
    }

    return true;
}

//static
bool Quantization::Dequantize_Q8_Linear(vector<inferflow_fp16> &target,
    const vector<uint8_t> &quant_array, const LinearQuantParams &params)
{
    int size = (int)quant_array.size();
    target.resize(size);

    float z = params.z;
    float scale1 = params.scale1;
    float scale2 = params.scale2;

    for (int idx = 0; idx < size; idx++)
    {
        uint8_t q = quant_array[idx];
        if (q >= 128) {
            target[idx] = (inferflow_fp16)(z - (int)(q - 128) * scale2);
        }
        else {
            target[idx] = (inferflow_fp16)(z + (int)q * scale1);
        }
    }
 
    return true;
}

//static
bool Quantization::Quantize_Q8_Log(vector<uint8_t> &quant_array,
    const vector<inferflow_fp16> &source, const LogQuantParams &params)
{
    int size = (int)source.size();
    quant_array.resize(size);

    float base = params.base;
    float scale = (float)params.scale;
    float start = (float)params.start;

    int q = 0;
    for (int idx = 0; idx < size; idx++)
    {
        int sign = (float)source[idx] >= 0 ? 1 : -1;
        float v = (float)source[idx] * scale * sign;

        if (v >= 0.99f) {
            q = (int)(0.5f + start + log(v) / log(base));
        }
        else {
            q = (int)(0.5f + v * start);
        }

        quant_array[idx] = (uint8_t)(128 + min(127, q) * sign);
    }

    return true;
}

//static
bool Quantization::Dequantize_Q8_Log(vector<inferflow_fp16> &target,
    const vector<uint8_t> &quant_array, const LogQuantParams &params)
{
    int size = (int)quant_array.size();
    target.resize(size);

    float base = params.base;
    float scale = (float)params.scale;
    float start = (float)params.start;

    float v = 0;
    for (int idx = 0; idx < size; idx++)
    {
        uint8_t q = quant_array[idx];
        int sign = q >= 128 ? 1 : -1;
        int num = q >= 128 ? (q - 128) : (128 - q);
        if (num >= start) {
            v = pow(base, num - start) / scale;
        }
        else {
            v = num / scale;
        }

        target[idx] = (inferflow_fp16)(v * sign);
    }

    return true;
}

//static
bool Quantization::DequantizeRow_Q8_B32T1(float *target,
    const BlockQ8_B32T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ8_B32T1(target + idx * Q8B32_CAPACITY, blocks + idx);
    }

    return true;
}

//static
bool Quantization::DequantizeRow_Q8_B32T2(float *target,
    const BlockQ8_B32T2 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ8_B32T2(target + idx * Q8B32_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q6_B64T1(float * target,
    const BlockQ6_B64T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ6_B64T1(target + idx * Q6_B64_CAPACITY, blocks + idx);
    }

    return true;
}

//static
bool Quantization::DequantizeRow_Q5(float *target, const BlockQ5_B32T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ5Block(target + idx * Q5B32_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q5_B64T1(float * target,
    const BlockQ5_B64T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ5_B64T1(target + idx * Q5_B64_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q4_B16(float *target,
    const BlockQ4_B16 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ4_B16(target + idx * Q4B16_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q4_B32T1(float *target,
    const BlockQ4_B32T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ4_B32T1(target + idx * Q4B32_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q4_B64T1(float * target,
    const BlockQ4_B64T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ4_B64T1(target + idx * Q4_B64_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q3H_B64T1(float * target,
    const BlockQ3H_B64T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ3H_B64T1(target + idx * Q3H_B64_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q3_B32T1(float *target,
    const BlockQ3_B32T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ3_B32T1(target + idx * Q3B32_CAPACITY, blocks + idx);
    }

    return true;
}

bool Quantization::DequantizeRow_Q2_B32T1(float *target,
    const BlockQ2_B32T1 *blocks, uint64_t block_count)
{
    if (target == nullptr || blocks == nullptr) {
        return false;
    }

    for (uint64_t idx = 0; idx < block_count; idx++)
    {
        DequantizeQ2_B32T1(target + idx * Q2B32_CAPACITY, blocks + idx);
    }

    return true;
}

INFER_FLOW_END
