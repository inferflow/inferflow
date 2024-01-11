#pragma once

#include <map>
#include "ggml/ggml.h"
#include "namespace.inc"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::map;

class CpuKVCacheLayer
{
public:
    bool Init(ggml_context *ctx, int max_context_len, int dim);

    ggml_tensor* GetKRows(ggml_context *ctx, int start_row, int row_count);
    ggml_tensor* GetVRows(ggml_context *ctx, int start_row, int row_count);

    bool SetKRows(ggml_cgraph *gf, ggml_context *ctx, ggml_tensor *input, int start_row, int prefix_len, int token_num);
    bool SetVRows(ggml_cgraph *gf, ggml_context *ctx, ggml_tensor *input, int start_row, int prefix_len, int token_num);

    struct ggml_tensor* Get_K_Cache() const {
        return k_cache_;
    }

    struct ggml_tensor* Get_V_Cache() const {
        return v_cache_;
    }

protected:
    struct ggml_tensor *k_cache_;
    struct ggml_tensor *v_cache_;
};

class CpuKVCache
{
public:
    CpuKVCache();
    virtual ~CpuKVCache();
    void Clear();

    bool Init(struct ggml_context *ctx, int max_context_len, int dim, int start_layer, int end_layer);
    CpuKVCacheLayer* Layer(int layer_id);
    const CpuKVCacheLayer* Layer(int layer_id) const;
    
protected:
    map<int, CpuKVCacheLayer*> layer_map_;
};

TRANSFORMER_END
INFER_FLOW_END
