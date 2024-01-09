#pragma once

#include <iostream>
#include <vector>
#include "common/cuda_util.h"
#include "namespace.inc"

INFER_FLOW_BEGIN

class DeviceMemoryHeap
{
protected:
    void *data_ = nullptr;
    int capacity_ = 0;
    int allocated_size_ = 0;

    const static int Alignment_ = 256;

public:
    explicit DeviceMemoryHeap()
    {
        this->capacity_ = 0;
    }

    virtual ~DeviceMemoryHeap()
    {
        Clear(true);
    }

    void Clear(bool free_memory = false);

    bool Init(int capacity = 32 * 1000 * 1000);

    void* New(int size);

    float* NewFloatArray(int size)
    {
        void *ret_data = New(size * sizeof(float));
        return (float*)ret_data;
    }

    half* NewHalfArray(int size)
    {
        void *ret_data = New(size * sizeof(half));
        return (half*)ret_data;
    }
};

INFER_FLOW_END

