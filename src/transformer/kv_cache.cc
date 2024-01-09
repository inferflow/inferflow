#include "kv_cache.h"
#include "sslib/log.h"
#include <algorithm>

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

////////////////////////////////////////////////////////////////////////////////
// class CpuLayerKVCache

bool CpuLayerKVCache::Init(ElementType etype, int max_context_len, int dim)
{
    is_cpu_ = true;
    data_type_ = etype;
    dim_ = dim;
    bool ret1 = k_data_.New(dim *  max_context_len);
    bool ret2 = v_data_.New(dim * max_context_len);
    return ret1 && ret2;
}

bool CpuLayerKVCache::GetKRows(DeviceTensor &rows, int prefix_len, int token_num)
{
    if (rows.data_type != data_type_) {
        LogError("Inconsistent data types: %d vs. %d", (int)rows.data_type, (int)data_type_);
        return false;
    }

    rows.SetStructure(dim_, token_num);
    const inferflow_fp16 *f16_data = k_data_.data() + dim_ * prefix_len;
    int bytes = dim_ * token_num * (int)sizeof(inferflow_fp16);
    bool ret = rows.CopyFromHost((const void*)f16_data, bytes);
    return ret;
}

bool CpuLayerKVCache::GetVRows(DeviceTensor &rows, int prefix_len, int token_num)
{
    if (rows.data_type != data_type_) {
        LogError("Inconsistent data types: %d vs. %d", (int)rows.data_type, (int)data_type_);
        return false;
    }

    rows.SetStructure(dim_, token_num);
    const inferflow_fp16 *f16_data = v_data_.data() + dim_ * prefix_len;
    int bytes = dim_ * token_num * (int)sizeof(inferflow_fp16);
    bool ret = rows.CopyFromHost((const void*)f16_data, bytes);
    return ret;
}

bool CpuLayerKVCache::SetKRows(const DeviceTensor &rows, int start_row,
    int prefix_len, int token_num)
{
    if (rows.data_type != data_type_) {
        LogError("Inconsistent data type: %d vs. %d", rows.data_type, data_type_);
        return false;
    }

    if (rows.ne[0] != dim_) {
        LogError("Inconsistent column count: %d vs. %d", rows.ne[0], dim_);
        return false;
    }

    uint64_t byte_num = TensorCommon::ByteCount(data_type_, dim_ * token_num);
    const void *source_data = rows.RowData(start_row);
    inferflow_fp16 *target_data = k_data_.data() + prefix_len * dim_;
    bool ret = CudaUtil::DeviceToHostMemcpy((void*)target_data, source_data, byte_num);
    return ret;
}

bool CpuLayerKVCache::SetVRows(const DeviceTensor &rows, int start_row,
    int prefix_len, int token_num)
{
    if (rows.data_type != data_type_) {
        LogError("Inconsistent data type: %d vs. %d", rows.data_type, data_type_);
        return false;
    }

    if (rows.ne[0] != dim_) {
        LogError("Inconsistent column count: %d vs. %d", rows.ne[0], dim_);
        return false;
    }

    uint64_t byte_num = TensorCommon::ByteCount(data_type_, dim_ * token_num);
    const void *source_data = rows.RowData(start_row);
    inferflow_fp16 *target_data = v_data_.data() + prefix_len * dim_;
    bool ret = CudaUtil::DeviceToHostMemcpy((void*)target_data, source_data, byte_num);
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// class GpuLayerKVCache

bool GpuLayerKVCache::Init(ElementType etype, int max_context_len, int dim)
{
    is_cpu_ = false;
    bool ret1 = k_data_.New(etype, dim, max_context_len);
    bool ret2 = v_data_.New(etype, dim, max_context_len);
    return ret1 && ret2;
}

bool GpuLayerKVCache::GetKRows(DeviceTensor &rows, int context_len, int token_num)
{
    rows.data_type = k_data_.data_type;
    rows.SetStructure(k_data_.ne[0], token_num);
    rows.data = k_data_.RowData(context_len);
    rows.SetAutoFree(false);
    return true;
}

bool GpuLayerKVCache::GetVRows(DeviceTensor &rows, int context_len, int token_num)
{
    rows.data_type = k_data_.data_type;
    rows.SetStructure(v_data_.ne[0], token_num);
    rows.data = v_data_.RowData(context_len);
    rows.SetAutoFree(false);
    return true;
}

bool GpuLayerKVCache::SetKRows(const DeviceTensor &rows, int start_row,
    int prefix_len, int token_num)
{
    if (rows.data_type != k_data_.data_type) {
        LogError("Inconsistent data type: %d vs. %d", rows.data_type, k_data_.data_type);
        return false;
    }

    if (rows.ne[0] != k_data_.ne[0]) {
        LogError("Inconsistent column count: %d vs. %d", rows.ne[0], k_data_.ne[0]);
        return false;
    }

    int element_size = TensorCommon::ElementSize(k_data_.data_type);
    int byte_num = element_size * k_data_.ne[0] * token_num;
    const void *source_data = rows.RowData(start_row);
    void *target_data = k_data_.RowData(prefix_len);
    auto ret_code = cudaMemcpy(target_data, source_data, byte_num, cudaMemcpyDeviceToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to call cudaMemcpy. Prefix_len: %d, token_num: %d, Ret-code: %d (%s)",
            prefix_len, token_num, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

bool GpuLayerKVCache::SetVRows(const DeviceTensor &rows, int start_row,
    int prefix_len, int token_num)
{
    if (rows.data_type != v_data_.data_type) {
        LogError("Inconsistent data type: %d vs. %d", rows.data_type, v_data_.data_type);
        return false;
    }

    if (rows.ne[0] != v_data_.ne[0]) {
        LogError("Inconsistent column count: %d vs. %d", rows.ne[0], v_data_.ne[0]);
        return false;
    }

    int element_size = TensorCommon::ElementSize(v_data_.data_type);
    int byte_num = element_size * v_data_.ne[0] * token_num;
    const void *source_data = rows.RowData(start_row);
    void *target_data = v_data_.RowData(prefix_len);
    auto ret_code = cudaMemcpy(target_data, source_data, byte_num, cudaMemcpyDeviceToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to call cudaMemcpy. Prefix_len: %d, token_num: %d, Ret-code: %d (%s)",
            prefix_len, token_num, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// class KVCache

KVCache::KVCache()
{
}

KVCache::~KVCache()
{
    Clear();
}

void KVCache::Clear()
{
    for (auto iter = layer_map_.begin(); iter != layer_map_.end(); iter++)
    {
        auto *layer_ptr = iter->second;
        if (layer_ptr != nullptr)
        {
            delete layer_ptr;
            layer_ptr = nullptr;
        }
    }

    layer_map_.clear();
}

bool KVCache::Init(ElementType etype, int max_context_len, int dim,
    int start_layer, int end_layer, int cpu_layer_percent, int r, int m)
{
    bool ret = true;
    if (end_layer <= start_layer || dim <= 0 || max_context_len <= 0) {
        return false;
    }

    int layer_num = end_layer - start_layer;
    int cpu_layer_num = layer_num * min(max(0, cpu_layer_percent), 100) / 100;
    int gpu_layer_num = layer_num - cpu_layer_num;

    int real_cpu_layer_num = 0, real_gpu_layer_num = 0;
    for (int layer_id = start_layer; ret && layer_id < end_layer; layer_id++)
    {
        LayerKVCache *layer_ptr = nullptr;
        if (layer_id % m == r)
        {
            if (layer_id - start_layer < gpu_layer_num) {
                layer_ptr = new GpuLayerKVCache;
                real_gpu_layer_num++;
            }
            else {
                layer_ptr = new CpuLayerKVCache;
                real_cpu_layer_num++;
            }
        }

        if (layer_ptr != nullptr) {
            ret = layer_ptr->Init(etype, max_context_len, dim);
        }
        layer_map_[layer_id] = layer_ptr;
    }

    int element_size = TensorCommon::ElementSize(etype);
    float mb_per_layer = 2.0f * max_context_len * dim * element_size / 1024 / 1024;
    host_memory_cost_mb_ = mb_per_layer * real_cpu_layer_num;
    device_memory_cost_mb_ = mb_per_layer * real_gpu_layer_num;

    return ret;
}

LayerKVCache* KVCache::Layer(int layer_id)
{
    auto iter = layer_map_.find(layer_id);
    return iter != layer_map_.end() ? iter->second : nullptr;
}

const LayerKVCache* KVCache::Layer(int layer_id) const
{
    auto iter = layer_map_.find(layer_id);
    return iter != layer_map_.end() ? iter->second : nullptr;
}

float KVCache::MemoryCost() const
{
    return device_memory_cost_mb_ + host_memory_cost_mb_;
}

float KVCache::DeviceMemoryCost() const
{
    return device_memory_cost_mb_;
}

float KVCache::HostMemoryCost() const
{
    return host_memory_cost_mb_;
}

TRANSFORMER_END
INFER_FLOW_END

