#pragma once

#include "sslib/blocked_allocator.h"
#include "ggml/ggml.h"
#include "tensor/host_tensor.h"
#if defined(USE_CUDA)
#   include "tensor/device_tensor.h"
#   include "tensor/tensor_mul.h"
#endif

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using sslib::BlockedAllocator;

enum class TokenType
{
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    SingleByte = 6,
    Invalid = 255
};

struct TransformerContext
{
    BlockedAllocator<uint8_t> byte_heap;
    BlockedAllocator<inferflow_fp16> fp16_heap;
    BlockedAllocator<float> float_heap;
    BlockedAllocator<HostTensor> host_tensor_heap;
    struct ggml_context *ggml_ctx = nullptr;

#if defined(USE_CUDA)
    BlockedAllocator<DeviceTensor> device_tensor_heap;
    BlockedAllocator<DeviceSparseMatrix> sparse_matrix_heap;
#endif   
};

TRANSFORMER_END
INFER_FLOW_END
