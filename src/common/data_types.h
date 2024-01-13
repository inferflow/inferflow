#pragma once

#include <stdint.h>
#include "namespace.inc"

//#pragma warning(push)
//#pragma warning(disable:4003)
#include "half/half.hpp"
//#pragma warning(pop)

#ifdef USE_CUDA
#   include "common/cuda_util.h"
#endif //USE_CUDA


INFER_FLOW_BEGIN

struct Float8
{
    float data[8];
};

struct Float16
{
    float data[16];
};

struct Float32
{
    float data[32];
};

typedef half_float::half host_fp16_t;
#if defined(USE_CUDA)
    typedef half inferflow_fp16;
#else
    typedef half_float::half inferflow_fp16;
#endif

#if defined(USE_CUDA)
#   define HOST_AND_DEVICE __host__ __device__
#   define FORCE_INLINE __forceinline__
#else
#   define HOST_AND_DEVICE
#   define FORCE_INLINE
#endif

/*
#if defined(USE_CUDA)
    typedef half inferflow_fp16_t;
#elif defined(__ARM_NEON)
    typedef __fp16 inferflow_fp16_t;
#else
    typedef uint16_t inferflow_fp16_t;
#endif

#if defined(USE_CUDA)
#   define FP16_TO_FP32
#else
#endif*/

INFER_FLOW_END
