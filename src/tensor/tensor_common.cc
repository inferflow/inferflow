#include "tensor_common.h"

INFER_FLOW_BEGIN

//static
bool TensorCommon::IsQuantType(ElementType etype)
{
    return etype >= ElementType::Q8_GL && etype < ElementType::Invalid;
}

//static
int TensorCommon::ElementSize(ElementType etype)
{
    int element_size = 0;
    switch (etype)
    {
    case ElementType::F32:
        element_size = 4;
        break;
    case ElementType::F16:
    case ElementType::BF16:
        element_size = 2;
        break;
    case ElementType::Q8_GL:
    case ElementType::Q8_LOG:
        element_size = 1;
        break;
    default:
        break;
    }

    return element_size;
}

//static
int TensorCommon::BlockSize(ElementType etype)
{
    int block_size = 1;
    switch (etype)
    {
        case ElementType::F32:
            block_size = 4;
            break;
        case ElementType::F16:
        case ElementType::BF16:
            block_size = 2;
            break;
        case ElementType::Q8_GL:
        case ElementType::Q8_LOG:
            block_size = 1;
            break;
        case ElementType::Q8_B32T1:
            block_size = sizeof(BlockQ8_B32T1);
            break;
        case ElementType::Q8_B32T2:
            block_size = sizeof(BlockQ8_B32T2);
            break;
        case ElementType::Q5:
            block_size = sizeof(BlockQ5_B32T1);
            break;
        case ElementType::Q4_B16:
            block_size = sizeof(BlockQ4_B16);
            break;
        case ElementType::Q4_B32T1A:
        case ElementType::Q4_B32T1B:
            block_size = sizeof(BlockQ4_B32T1);
            break;
        case ElementType::Q4_B32T2:
            block_size = sizeof(BlockQ4_B32T2);
            break;
        case ElementType::Q3H_B64T1:
            block_size = sizeof(BlockQ3H_B64T1);
            break;
        case ElementType::Q3_B32T1A:
        case ElementType::Q3_B32T1B:
            block_size = sizeof(BlockQ3_B32T1);
            break;
        case ElementType::Q2_B32T1A:
        case ElementType::Q2_B32T1B:
            block_size = sizeof(BlockQ2_B32T1);
            break;
        default:
            break;
    }

    return block_size;
}

//static
int TensorCommon::BlockCapacity(ElementType etype)
{
    int block_capacity = 32;
    switch (etype)
    {
    case ElementType::F32:
    case ElementType::F16:
    case ElementType::BF16:
    case ElementType::Q8_GL:
    case ElementType::Q8_LOG:
        block_capacity = 1;
        break;
    case ElementType::Q8_B32T1:
    case ElementType::Q8_B32T2:
        block_capacity = Q8B32_CAPACITY;
        break;
    case ElementType::Q5:
        block_capacity = Q5B32_CAPACITY;
        break;
    case ElementType::Q4_B16:
        block_capacity = Q4B16_CAPACITY;
        break;
    case ElementType::Q4_B32T1A:
    case ElementType::Q4_B32T1B:
    case ElementType::Q4_B32T2:
        block_capacity = Q4B32_CAPACITY;
        break;
    case ElementType::Q3H_B64T1:
        block_capacity = Q3H_B64_CAPACITY;
        break;
    case ElementType::Q3_B32T1A:
    case ElementType::Q3_B32T1B:
        block_capacity = Q3B32_CAPACITY;
        break;
    case ElementType::Q2_B32T1A:
    case ElementType::Q2_B32T1B:
        block_capacity = Q2B32_CAPACITY;
        break;
    default:
        break;
    }

    return block_capacity;
}

//static
uint64_t TensorCommon::ByteCount(ElementType etype, uint64_t element_num)
{
    int c = BlockCapacity(etype);
    int b = BlockSize(etype);
    return (element_num + (c - 1)) / c * b;
}

//static
void TensorCommon::InitElementTypeMap(ElementTypeMap &type_map)
{
    type_map.clear();
    type_map["fp32"] = ElementType::F32;
    type_map["f32"] = ElementType::F32;
    type_map["fp16"] = ElementType::F16;
    type_map["f16"] = ElementType::F16;
    type_map["q8"] = ElementType::Q8_B32T1;
    type_map["q4"] = ElementType::Q4_B32T1A;
    type_map["q3h"] = ElementType::Q3H_B64T1;
    type_map["q3"] = ElementType::Q3_B32T1B;
    type_map["q2"] = ElementType::Q2_B32T1B;
    type_map["q8_gl"] = ElementType::Q8_GL;
    type_map["q8_log"] = ElementType::Q8_LOG;
    type_map["q8_b32t1"] = ElementType::Q8_B32T1;
    type_map["q8_b32t2"] = ElementType::Q8_B32T2;
    type_map["q5"] = ElementType::Q5;
    type_map["q4_b16"] = ElementType::Q4_B16;
    type_map["q4_b32t1a"] = ElementType::Q4_B32T1A;
    type_map["q4_b32t1b"] = ElementType::Q4_B32T1B;
    type_map["q4_b32t1"] = ElementType::Q4_B32T1A;
    type_map["q4_b32t2"] = ElementType::Q4_B32T2;
    type_map["q3h_b64t1"] = ElementType::Q3H_B64T1;
    type_map["q3_b32t1a"] = ElementType::Q3_B32T1A;
    type_map["q3_b32t1b"] = ElementType::Q3_B32T1B;
    type_map["q3_b32t1"] = ElementType::Q3_B32T1B;
    type_map["q2_b32t1a"] = ElementType::Q2_B32T1A;
    type_map["q2_b32t1b"] = ElementType::Q2_B32T1B;
    type_map["q2_b32t1"] = ElementType::Q2_B32T1B;
}

//static
void TensorCommon::InitElementTypeMap(ElementTypeNameMap &type_name_map)
{
    type_name_map[ElementType::F32] = "fp32";
    type_name_map[ElementType::F16] = "fp16";
    type_name_map[ElementType::BF16] = "bf16";

    type_name_map[ElementType::Q8_GL] = "q8_gl";
    type_name_map[ElementType::Q8_LOG] = "q8_log";
    type_name_map[ElementType::Q8_B32T1] = "q8_b32t1";
    type_name_map[ElementType::Q8_B32T2] = "q8_b32t2";
    type_name_map[ElementType::Q5] = "q5";
    type_name_map[ElementType::Q4_B16] = "q4_b16";
    type_name_map[ElementType::Q4_B32P8] = "q4_b32p8";
    type_name_map[ElementType::Q4_B32T1A] = "q4_b32t1a";
    type_name_map[ElementType::Q4_B32T1B] = "q4_b32t1b";
    type_name_map[ElementType::Q4_B32T2] = "q4_b32t2";
    type_name_map[ElementType::Q3H_B64T1] = "q3h_b64t1";
    type_name_map[ElementType::Q3_B32T1A] = "q3_b32t1a";
    type_name_map[ElementType::Q3_B32T1B] = "q3_b32t1b";
    type_name_map[ElementType::Q2_B32T1A] = "q2_b32t1a";
    type_name_map[ElementType::Q2_B32T1B] = "q2_b32t1b";
}

INFER_FLOW_END
