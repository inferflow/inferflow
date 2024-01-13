#pragma once

#include "common/data_types.h"
#include "tensor_common.h"

INFER_FLOW_BEGIN

//FP32 buffer
class HostFloatBuffer
{
public:
    HostFloatBuffer();
    virtual ~HostFloatBuffer();
    void Clear();

    int capacity() const {
        return capacity_;
    }

    const float* data() const {
        return data_;
    }

    float* data() {
        return data_;
    }

    bool New(int capacity);
    void Set(int idx, float value);

protected:
    float *data_ = nullptr;
    int capacity_ = 0;
    bool is_pinned_memory_ = false;

protected:
    //disable the copy constructor and the assignment function
    HostFloatBuffer(const HostFloatBuffer &rhs) = delete;
    HostFloatBuffer& operator = (const HostFloatBuffer &rhs) = delete;
};

//FP16 buffer
class HostHalfBuffer
{
public:
    HostHalfBuffer();
    virtual ~HostHalfBuffer();
    void Clear();

    int capacity() const {
        return capacity_;
    }

    const inferflow_fp16* data() const {
        return data_;
    }

    inferflow_fp16* data() {
        return data_;
    }

    bool New(int capacity);
    void Set(int idx, inferflow_fp16 value);
    void Set(int start_idx, const inferflow_fp16 *buf, int buf_len);

protected:
    inferflow_fp16 *data_ = nullptr;
    int capacity_ = 0;
    bool is_pinned_memory_ = false;

protected:
    //disable the copy constructor and the assignment function
    HostHalfBuffer(const HostHalfBuffer &rhs) = delete;
    HostHalfBuffer& operator = (const HostHalfBuffer &rhs) = delete;
};

INFER_FLOW_END
