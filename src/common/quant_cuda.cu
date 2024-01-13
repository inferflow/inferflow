#include "quant_cuda.h"

INFER_FLOW_BEGIN

//static
bool DeviceQuantization::DequantizeQ5Row(half *target, const BlockQ5_B32T1 *blocks, uint64_t block_count)
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

INFER_FLOW_END
