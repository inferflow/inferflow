#include "device_memory_heap.h"
#include "sslib/log.h"

INFER_FLOW_BEGIN

using namespace sslib;

void DeviceMemoryHeap::Clear(bool free_memory)
{
    allocated_size_ = 0;

    if (data_ != nullptr && free_memory)
    {
        cudaFree(data_);
        data_ = nullptr;
    }
}

bool DeviceMemoryHeap::Init(int capacity)
{
    Clear(true);

    auto ret_code = cudaMalloc((void**)&this->data_, capacity);
    if (ret_code != cudaSuccess)
    {
        this->data_ = nullptr;
        return false;
    }

    this->capacity_ = capacity;
    this->allocated_size_ = 0;
    return true;
}

void* DeviceMemoryHeap::New(int size)
{
    if (size == 0) {
        LogError("Size 0 is not allowed in memory allocation.");
        return nullptr;
    }

    int mod = allocated_size_ % Alignment_;
    if (mod != 0)
    {
        //LogKeyInfo("Before alignment: %d", allocated_size_);
        allocated_size_ += (Alignment_ - mod);
        //LogKeyInfo("After alignment: %d",  allocated_size_);
    }

    if (size < 0 || allocated_size_ + size > capacity_)
    {
        LogError("No space is available: %d vs. %d", allocated_size_ + size, capacity_);
        return nullptr;
    }

    void *ret_data = ((uint8_t*)data_) + allocated_size_;
    allocated_size_ += size;
    return ret_data;
}

INFER_FLOW_END

