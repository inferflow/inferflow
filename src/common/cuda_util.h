#pragma once

#ifndef __builtin_ia32_serialize
void __builtin_ia32_serialize(void);
#endif

#include <stdint.h>
#include "namespace.inc"

#ifdef __GNUC__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunknown-pragmas"
#else
#   pragma warning(push)
#   pragma warning(disable:4505)
#endif
#   include <cuda_runtime.h>
#   include <cuda_fp16.h>
//#   include <sm_61_intrinsics.hpp>
#ifdef __GNUC__
#   pragma GCC diagnostic pop
#else
#   pragma warning(pop)
#endif

INFER_FLOW_BEGIN

struct DeviceMemoryInfo
{
    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
    uint64_t used_bytes = 0;
    float free_gb = 0;
    float total_gb = 0;
    float used_gb = 0;
};

class CudaUtil
{
public:
    static bool CheckReturnCode(cudaError_t ret_code, const char *error_title = nullptr);

    static int DeviceCount();
    static int GetDevice();
    static void SetDevice(int device_id);

    static bool DeviceSynchronize(const char *error_title = nullptr);

    //reutrn: device id
    static int GetDeviceMemoryInfo(DeviceMemoryInfo &mem_info, int device_id = -1);
    static int LogDeviceMemoryInfo(int device_id = -1);

    static bool DeviceToDeviceMemcpy(void *dst, const void *src, size_t byte_count);
    static bool DeviceToHostMemcpy(void *dst, const void *src, size_t byte_count);
    static bool HostToDeviceMemcpy(void *dst, const void *src, size_t byte_count);

    static bool MemcpyPeer(void *dst, int dst_device, const void *src, int src_device,
        size_t byte_count);
};

INFER_FLOW_END
