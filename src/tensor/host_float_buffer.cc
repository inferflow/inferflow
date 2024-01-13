#include "host_float_buffer.h"
#include "sslib/log.h"

INFER_FLOW_BEGIN

using namespace std;
using namespace sslib;

////////////////////////////////////////////////////////////////////////////////
// HostFloatBuffer

HostFloatBuffer::HostFloatBuffer()
{
}

HostFloatBuffer::~HostFloatBuffer()
{
    Clear();
}

void HostFloatBuffer::Clear()
{
    if (data_ != nullptr)
    {
        if (is_pinned_memory_)
        {
#if defined(USE_CUDA)
            auto ret_code = cudaFreeHost(data_);
            if (ret_code != cudaSuccess)
            {
                LogError("Failed to call cudaFreeHost: %d (%s)",
                    ret_code, cudaGetErrorString(ret_code));
            }
#endif //USE_CUDA
        }
        else
        {
            delete[] data_;
        }

        data_ = nullptr;
    }

    capacity_ = 0;
    is_pinned_memory_ = false;
}

bool HostFloatBuffer::New(int capacity)
{
    Clear();

#if defined(USE_CUDA)
    int element_size = sizeof(float);
    int bytes = capacity * element_size;
    auto ret_code = cudaMallocHost((void**)&data_, bytes);
    if (ret_code == cudaSuccess)
    {
        is_pinned_memory_ = true;
    }
    else
    {
        LogWarning("Failed to call cudaMallocHost (%d bytes). Error: %d (%s)",
            bytes, ret_code, cudaGetErrorString(ret_code));

        is_pinned_memory_ = false;
        data_ = new float[capacity];
    }
#else
    is_pinned_memory_ = false;
    data_ = new float[capacity];
#endif //USE_CUDA

    if (data_ != nullptr)
    {
        capacity_ = capacity;
    }

    return data_ != nullptr;
}

void HostFloatBuffer::Set(int idx, float value)
{
    if (idx >= 0 && idx < capacity_) {
        data_[idx] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////
// HostHalfBuffer

HostHalfBuffer::HostHalfBuffer()
{
}

HostHalfBuffer::~HostHalfBuffer()
{
    Clear();
}

void HostHalfBuffer::Clear()
{
    if (data_ != nullptr)
    {
        if (is_pinned_memory_)
        {
#if defined(USE_CUDA)
            auto ret_code = cudaFreeHost(data_);
            if (ret_code != cudaSuccess)
            {
                LogError("Failed to call cudaFreeHost: %d (%s)",
                    ret_code, cudaGetErrorString(ret_code));
            }
#endif //USE_CUDA
        }
        else
        {
            delete[] data_;
        }

        data_ = nullptr;
    }

    capacity_ = 0;
    is_pinned_memory_ = false;
}

bool HostHalfBuffer::New(int capacity)
{
    Clear();

#if defined(USE_CUDA)
    int element_size = sizeof(inferflow_fp16);
    int bytes = capacity * element_size;
    auto ret_code = cudaMallocHost((void**)&data_, bytes);
    if (ret_code == cudaSuccess)
    {
        is_pinned_memory_ = true;
    }
    else
    {
        LogWarning("Failed to call cudaMallocHost (%d bytes). Error: %d (%s)",
            bytes, ret_code, cudaGetErrorString(ret_code));

        is_pinned_memory_ = false;
        data_ = new half[capacity];
    }
#else
    is_pinned_memory_ = false;
    data_ = new inferflow_fp16[capacity];
#endif //USE_CUDA

    if (data_ != nullptr)
    {
        capacity_ = capacity;
    }

    return data_ != nullptr;
}

void HostHalfBuffer::Set(int idx, inferflow_fp16 value)
{
    if (idx >= 0 && idx < capacity_) {
        data_[idx] = value;
    }
}

void HostHalfBuffer::Set(int start_idx, const inferflow_fp16 *buf, int buf_len)
{
    if (start_idx < 0 || buf_len < 0 || start_idx + buf_len > capacity_)
    {
        LogError("Invalid parameters: %d, %d (capacity = %d)", start_idx, buf_len, capacity_);
        return;
    }

    memcpy(data_ + start_idx, buf, buf_len * sizeof(inferflow_fp16));
}

INFER_FLOW_END
