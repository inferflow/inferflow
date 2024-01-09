#pragma once

#include <iostream>
#include "inference_types.h"
#include "query_state_table.h"
#include "cpu_kv_cache.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::ostream;

class CpuInferenceWorker
{
public:
    CpuInferenceWorker() {};
    virtual ~CpuInferenceWorker() {};

    bool Init(const InferenceConfig &cfg, const InferenceConfigEx &config_ex,
        TransformerModel &model, int start_layer, int end_layer,
        struct ggml_context *ctx);

    struct ggml_tensor* Inference(const vector<QueryProcInput> &query_list,
        struct ggml_tensor *input_tensor, ggml_cgraph &gf,
        struct ggml_context *ctx, ostream *tensor_writer);
    
    int start_layer() const {
        return start_layer_;
    }

    int end_layer() const {
        return end_layer_;
    }

    static ggml_context* CreateContext(size_t mem_size);

protected:
    struct CurQKV
    {
        ggml_tensor *q = nullptr;
        ggml_tensor *k = nullptr;
        ggml_tensor *v = nullptr;
    };

    struct QueryProcData
    {
        CpuKVCache kv_cache;
    };

    struct AttentionOutput
    {
        ggml_tensor *output = nullptr;
        ggml_tensor *pre_norm = nullptr;
    };

protected:
    void DebugTensor(ggml_tensor *tensor, string name, bool condition) const;
    void Calculate(ggml_tensor *tensor) const;
    void PrintInformation(ggml_tensor *tensor, string name) const;
    void PrintTensor(ggml_tensor *tensor, string name) const;
    void PrintTensor2d(ggml_tensor *tensor, string name) const;
    void PrintTensor3d(ggml_tensor *tensor, string name) const;
    bool Convert_F32(ggml_tensor *a, ggml_tensor *b);

    ggml_tensor* LayerNorm(ggml_tensor *input_tensor, ggml_tensor *norm, ggml_tensor *norm_b);
    ggml_tensor* ProcessPreLayer(ggml_tensor *input_tensor);
    ggml_tensor* ProcessPostLayer(ggml_tensor *input_tensor);
    ggml_tensor* ProcessCpuLayer(int layer_idx, ggml_tensor *input_tensor);
    CpuInferenceWorker::AttentionOutput ProcessCpuLayer_SelfAttention(int layer_idx, ggml_tensor *input_tensor);
    ggml_tensor* ProcessCpuLayer_FeedForward(int layer_idx, ggml_tensor *input_tensor);
    bool SelfAttention_CalculateCurQKV(CurQKV &cur_qkv, int layer_idx, const StdGgmlNetwork::DecoderLayer *layer, struct ggml_tensor *input_tensor);
    ggml_tensor* SelfAttention_CalculateQueryRope(ggml_tensor *input_tensor, ggml_tensor *output_tensor, int token_num, int start_row, int n_past, int rope_type);

protected:
    const InferenceConfig *config_ = nullptr;
    const InferenceConfigEx *config_ex_ = nullptr;
    TransformerModel *model_ptr_ = nullptr;
    int start_layer_ = 0;
    int end_layer_ = 0;
    vector<QueryProcInput> query_list_;
    PtrVector<QueryProcData> query_proc_data_list_;

    int cur_layer_idx_ = 0;
    ostream *tensor_writer_ = nullptr;  

    ggml_context *ctx_;
    ggml_cgraph *gf_;
};

TRANSFORMER_END
INFER_FLOW_END