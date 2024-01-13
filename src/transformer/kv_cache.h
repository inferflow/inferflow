#pragma once

#include <map>
#include "common/cuda_util.h"
#include "tensor/host_float_buffer.h"
#include "tensor/device_tensor.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::map;

class LayerKVCache
{
public:
    virtual ~LayerKVCache() {};

    virtual bool Init(ElementType etype, int max_context_len, int dim) = 0;

    virtual bool GetKRows(DeviceTensor &rows, int start_row, int row_count) = 0;
    virtual bool GetVRows(DeviceTensor &rows, int start_row, int row_count) = 0;

    virtual bool SetKRows(const DeviceTensor &rows, int start_row,
        int prefix_len, int token_num) = 0;
    virtual bool SetVRows(const DeviceTensor &rows, int start_row,
        int prefix_len, int token_num) = 0;

    bool IsCpu() const {
        return is_cpu_;
    }

protected:
    bool is_cpu_ = false;
};

class CpuLayerKVCache : public LayerKVCache
{
public:
    virtual ~CpuLayerKVCache() {};

    virtual bool Init(ElementType etype, int max_context_len, int dim) override;

    //row_count: token_num
    virtual bool GetKRows(DeviceTensor &rows, int start_row, int row_count) override;
    virtual bool GetVRows(DeviceTensor &rows, int start_row, int row_count) override;

    virtual bool SetKRows(const DeviceTensor &rows, int start_row,
        int prefix_len, int token_num) override;
    virtual bool SetVRows(const DeviceTensor &rows, int start_row,
        int prefix_len, int token_num) override;

protected:
    HostHalfBuffer k_data_;
    HostHalfBuffer v_data_;
    ElementType data_type_ = ElementType::F16;
    int dim_ = 0;
};

class GpuLayerKVCache : public LayerKVCache
{
public:
    virtual ~GpuLayerKVCache() {};

    virtual bool Init(ElementType etype, int max_context_len, int dim) override;

    //row_count: token_num
    virtual bool GetKRows(DeviceTensor &rows, int start_row, int row_count) override;
    virtual bool GetVRows(DeviceTensor &rows, int start_row, int row_count) override;

    virtual bool SetKRows(const DeviceTensor &rows, int start_row,
        int prefix_len, int token_num) override;
    virtual bool SetVRows(const DeviceTensor &rows, int start_row,
        int prefix_len, int token_num) override;

protected:
    DeviceTensor k_data_;
    DeviceTensor v_data_;
};

class KVCache
{
public:
    KVCache();
    virtual ~KVCache();
    void Clear();

    bool Init(ElementType etype, int max_context_len, int dim,
        int start_layer, int end_layer, int cpu_layer_percent = 0,
        int r = 0, int m = 1);
    LayerKVCache* Layer(int layer_id);
    const LayerKVCache* Layer(int layer_id) const;

    //memory cost in MB
    float MemoryCost() const;
    float HostMemoryCost() const;
    float DeviceMemoryCost() const;

protected:
    map<int, LayerKVCache*> layer_map_;
    float device_memory_cost_mb_ = 0;
    float host_memory_cost_mb_ = 0;
};

TRANSFORMER_END
INFER_FLOW_END
