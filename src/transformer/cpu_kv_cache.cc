#include "cpu_kv_cache.h"
#include "sslib/log.h"
#include "ggml/ggml.h"
#include <iostream>

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace sslib;
using namespace std;

bool CpuKVCacheLayer::Init(ggml_context *ctx0, int max_context_len, int dim)
{
    k_cache_ = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, max_context_len);
    v_cache_ = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, max_context_len);
    return k_cache_ != nullptr && v_cache_ != nullptr;
}

ggml_tensor* CpuKVCacheLayer::GetKRows(ggml_context *ctx, int start_row, int row_count)
{

    ggml_tensor *result = ggml_view_2d(ctx, k_cache_, k_cache_->ne[0], row_count, k_cache_->nb[1],
        start_row * ggml_element_size(k_cache_) * k_cache_->ne[0]);
    return result;
}

ggml_tensor* CpuKVCacheLayer::GetVRows(ggml_context *ctx, int start_row, int row_count)
{

    ggml_tensor *result = ggml_view_2d(ctx, v_cache_, v_cache_->ne[0], row_count, v_cache_->nb[1],
        start_row * ggml_element_size(v_cache_) * v_cache_->ne[0]);

    return result;
}

bool CpuKVCacheLayer::SetKRows(ggml_cgraph *gf, ggml_context *ctx, ggml_tensor *input, int start_row, int prefix_len, int token_num)
{
    if (input->ne[0] != k_cache_->ne[0])
    {
        LogError("Inconsistent column count: %d vs. %d (k-cache)", input->ne[0], k_cache_->ne[0]);
        return false;
    }

    ggml_tensor *cal = ggml_view_2d(ctx, input, input->ne[0], token_num, input->nb[1],
        start_row * ggml_element_size(input) * input->ne[0]);
    ggml_tensor *cache = ggml_view_2d(ctx, k_cache_, k_cache_->ne[0], token_num, k_cache_->nb[1],
        prefix_len * ggml_element_size(k_cache_) * input->ne[0]);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, cal, cache));

    return true;
}

bool CpuKVCacheLayer::SetVRows(ggml_cgraph *gf, ggml_context *ctx, ggml_tensor *input, int start_row, int prefix_len, int token_num)
{
    if (input->ne[0] != v_cache_->ne[0])
    {
        LogError("Inconsistent column count: %d vs. %d (v-cache)", input->ne[0], v_cache_->ne[0]);
        return false;
    }

    ggml_tensor *cal = ggml_view_2d(ctx, input, input->ne[0], token_num, input->nb[1],
        start_row * ggml_element_size(input) * input->ne[0]);
    ggml_tensor *cache = ggml_view_2d(ctx, v_cache_, v_cache_->ne[0], token_num, v_cache_->nb[1],
        prefix_len * ggml_element_size(v_cache_) * v_cache_->ne[0]);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, cal, cache));

    return true;
}


CpuKVCache::CpuKVCache()
{
}

CpuKVCache::~CpuKVCache()
{
    Clear();
}

void CpuKVCache::Clear()
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

bool CpuKVCache::Init(ggml_context *ctx0, int max_context_len, int dim, int start_layer, int end_layer)
{
    bool ret = true;
    if (end_layer <= start_layer || dim <= 0 || max_context_len <= 0)
    {
        return false;
    }

    //int layer_num = end_layer - start_layer;
    for (int layer_id = start_layer; ret && layer_id < end_layer; layer_id++)
    {
        auto *layer_ptr = new CpuKVCacheLayer;
        ret = layer_ptr->Init(ctx0, max_context_len, dim);
        layer_map_[layer_id] = layer_ptr;
    }

    return ret;
}

CpuKVCacheLayer* CpuKVCache::Layer(int layer_id)
{
    auto iter = layer_map_.find(layer_id);
    if (iter == layer_map_.end())
    {
        return nullptr;
    }

    return iter->second;
}

const CpuKVCacheLayer* CpuKVCache::Layer(int layer_id) const
{
    auto iter = layer_map_.find(layer_id);
    if (iter == layer_map_.end())
    {
        return nullptr;
    }

    return iter->second;
}

TRANSFORMER_END
INFER_FLOW_END