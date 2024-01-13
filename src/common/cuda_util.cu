#include "cuda_util.h"
#include "sslib/string.h"
#include "sslib/log.h"

INFER_FLOW_BEGIN

using namespace sslib;

//static
bool CudaUtil::CheckReturnCode(cudaError_t ret_code, const char *error_title)
{
    if (ret_code != cudaSuccess)
    {
        if (String::IsNullOrEmpty(error_title))
        {
            LogError("Failed to call a cuda function: %d (%s)",
                ret_code, cudaGetErrorString(ret_code));
        }
        else
        {
            LogError("%s: %d (%s)", error_title, ret_code,
                cudaGetErrorString(ret_code));
        }
        return false;
    }

    return true;
}

//static
int CudaUtil::DeviceCount()
{
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

//static
int CudaUtil::GetDevice()
{
    int device_id = 0;
    cudaGetDevice(&device_id);
    return device_id;
}

//static
void CudaUtil::SetDevice(int device_id)
{
    cudaSetDevice(device_id);
}

bool CudaUtil::DeviceSynchronize(const char *error_title)
{
    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    bool ret = CheckReturnCode(status, error_title);
    return ret;
}

//static
int CudaUtil::GetDeviceMemoryInfo(DeviceMemoryInfo &mem_info, int device_id)
{
    int device_count = DeviceCount();
    int current_device_id = GetDevice();
    if (device_id < 0 && device_id >= device_count) {
        device_id = current_device_id;
    }

    if (device_id != current_device_id) {
        SetDevice(device_id);
    }

    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    mem_info.free_bytes = free_bytes;
    mem_info.total_bytes = total_bytes;
    mem_info.used_bytes = total_bytes - free_bytes;
    mem_info.free_gb = free_bytes / 1024.0f / 1024 / 1024;
    mem_info.total_gb = total_bytes / 1024.0f / 1024 / 1024;
    mem_info.used_gb = mem_info.total_gb - mem_info.free_gb;

    if (device_id != current_device_id) {
        SetDevice(current_device_id);
    }
    return device_id;
}

//static
int CudaUtil::LogDeviceMemoryInfo(int device_id)
{
    DeviceMemoryInfo mi;
    int ret_device_id = GetDeviceMemoryInfo(mi, device_id);
    LogKeyInfo("device %d: total %.3f GB, used %.3f GB, free %.3f GB",
        ret_device_id, mi.total_gb, mi.used_gb, mi.free_gb);

    return ret_device_id;
}

//static
bool CudaUtil::DeviceToDeviceMemcpy(void *dst, const void *src, size_t byte_count)
{
    auto ret_code = cudaMemcpy(dst, src, byte_count, cudaMemcpyDeviceToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy %d byte(s) from device to device: %d (%s)",
            byte_count, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

//static
bool CudaUtil::DeviceToHostMemcpy(void *dst, const void *src, size_t byte_count)
{
    auto ret_code = cudaMemcpy(dst, src, byte_count, cudaMemcpyDeviceToHost);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy %d byte(s) to host: %d (%s)",
            byte_count, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

//static
bool CudaUtil::HostToDeviceMemcpy(void *dst, const void *src, size_t byte_count)
{
    auto ret_code = cudaMemcpy(dst, src, byte_count, cudaMemcpyHostToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy %d byte(s) from host: %d (%s)",
            byte_count, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

//static
bool CudaUtil::MemcpyPeer(void *dst, int dst_device, const void *src,
    int src_device, size_t byte_count)
{
    if (dst_device == src_device) {
        return DeviceToDeviceMemcpy(dst, src, byte_count);
    }

    auto ret_code = cudaMemcpyPeer(dst, dst_device, src, src_device, byte_count);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to call cudaMemcpyPeer: %d (%s)",
            ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    bool ret = DeviceSynchronize();
    return ret;
}

INFER_FLOW_END

