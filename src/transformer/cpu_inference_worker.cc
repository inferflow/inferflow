#include "cpu_inference_worker.h"
#include "sslib/log.h"
#include "query_state_table.h"
#include "tensor/tensor_util.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

bool CpuInferenceWorker::Init(const InferenceConfig &cfg, const InferenceConfigEx &config_ex,
    TransformerModel &model, int start_layer, int end_layer, struct ggml_context *ctx)
{
    bool ret = true;

    config_ = &cfg;
    config_ex_ = &config_ex;
    model_ptr_ = &model;
    start_layer_ = start_layer;
    end_layer_ = end_layer;

    const auto &hparams = model_ptr_->spec.hyper_params;
    int max_context_len = model_ptr_->spec.max_context_len;

    int max_batch_size = min(1 + QueryState::MAX_PROC_ID, cfg.max_concurrent_queries);

    int kv_cache_dim = hparams.embd_dims;
    if (hparams.decoder_kv_heads != hparams.decoder_heads) { // for group attention
        kv_cache_dim = hparams.embd_dims / hparams.decoder_heads * hparams.decoder_kv_heads;
    }

    for (int proc_id = 0; proc_id < max_batch_size; proc_id++)
    {
        QueryProcData *proc_data = new QueryProcData;
        query_proc_data_list_.push_back(proc_data);

        ret = proc_data->kv_cache.Init(ctx, max_context_len, kv_cache_dim, start_layer, end_layer);
        if (!ret)
        { 
            LogError("Failed to initialize the KV cache");
            return false;
        }
    }

    return true;
}

ggml_tensor* CpuInferenceWorker::Inference(const vector<QueryProcInput> &query_list, struct ggml_tensor *layer_input, 
    ggml_cgraph &gf, struct ggml_context *ctx, ostream *tensor_writer)
{
    ggml_tensor *output_tensor = nullptr;

    query_list_ = query_list;
    ctx_ = ctx;
    gf_ = &gf;
    tensor_writer_ = tensor_writer;
    cur_layer_idx_ = start_layer_;

    if (start_layer_ == 0)
    {
        ggml_tensor *out = ProcessPreLayer(layer_input);
        if (out != nullptr) {
            layer_input = out;
        }
    }
    for (int layer_idx = start_layer_; layer_idx < end_layer_; layer_idx++) 
    {
        cur_layer_idx_ = layer_idx;
        output_tensor = ProcessCpuLayer(layer_idx, layer_input);
        layer_input = output_tensor;
    }

    if (end_layer_ == model_ptr_->spec.hyper_params.decoder_layers) {
        output_tensor = ProcessPostLayer(output_tensor);
    }

    Calculate(output_tensor);
    return output_tensor;
}

ggml_context * CpuInferenceWorker::CreateContext(size_t mem_size)
{
    struct ggml_init_params ggml_params;
    ggml_params.mem_size = mem_size;
    ggml_params.mem_buffer = nullptr;
    ggml_params.no_alloc = false;
    auto *ctx = ggml_init(ggml_params);
    return ctx;
}

void CpuInferenceWorker::DebugTensor(ggml_tensor *tensor, string name, bool condition) const
{
    if (condition && (cur_layer_idx_ != start_layer_ && cur_layer_idx_ != 1)) {
        return;
    }
    Calculate(tensor);
    PrintTensor(tensor, name);
    tensor_writer_->flush();
} 

void CpuInferenceWorker::Calculate(ggml_tensor *tensor) const
{
    ggml_build_forward_expand(gf_, tensor);
    ggml_graph_compute(ctx_, gf_);
}

void CpuInferenceWorker::PrintInformation(struct ggml_tensor *tensor, string name) const
{
    ostream &tensor_writer = tensor_writer_ == nullptr ? cout : *tensor_writer_;
    tensor_writer << "Tensor: " <<  name << "\tLayer:" << cur_layer_idx_ << " Information\n";
    tensor_writer << "Tensor type: " << tensor->type << "\n";
    tensor_writer << "Tensor n_dims: " << tensor->n_dims << "\n";
    tensor_writer << "Tensor ne: ";
    for (const auto &e : tensor->ne) {
        tensor_writer << e << " ";
    }
    tensor_writer << "\n";
}

void CpuInferenceWorker::PrintTensor(struct ggml_tensor *tensor, string name) const
{
    if (tensor->n_dims == 2) {
        PrintTensor2d(tensor, name);
    } else if (tensor->n_dims == 3) {
        PrintTensor3d(tensor, name);
    }
}

void CpuInferenceWorker::PrintTensor2d(struct ggml_tensor *tensor, string name) const
{
    ostream &tensor_writer = tensor_writer_ == nullptr ? cout : *tensor_writer_;
    PrintInformation(tensor, name);

    auto ne = tensor->ne;
    auto nb = tensor->nb;

    for (int i1 = 0; i1 < min((int)ne[1], 8); i1++)
    {
        for (int i0 = 0; i0 < min((int)ne[0], 8); i0++)
        {
            float *data = (float *)((char *)tensor->data + i1 * nb[1] + i0 * nb[0]);
            i0 > 0 ? tensor_writer << ", " : tensor_writer << "[";
            i0 == min((int)ne[0], 8) - 1 ? tensor_writer << *data << "]\n" : tensor_writer << *data;
        }
    }
}

void CpuInferenceWorker::PrintTensor3d(struct ggml_tensor *tensor, string name) const
{
    ostream &tensor_writer = tensor_writer_ == nullptr ? cout : *tensor_writer_;

    PrintInformation(tensor, name);

    auto ne = tensor->ne;
    auto nb = tensor->nb;

    for (int i2 = 0; i2 < min((int)ne[2], 8); i2++)
    {
        for (int i1 = 0; i1 < min((int)ne[1], 8); i1++)
        {
            if (i1 == 0) {
                tensor_writer << "[";
            }
            for (int i0 = 0; i0 < min((int)ne[0], 8); i0++)
            {
                float *data = (float *)((char *)tensor->data + i2 * nb[2] + i1 * nb[1] + i0 * nb[0]);
                i0 > 0 ? tensor_writer << ", " : tensor_writer << "[";
                i0 == min((int)ne[0], 8) - 1 ? tensor_writer << *data << "]" : tensor_writer << *data;
            }
            i1 == min((int)ne[1], 8) - 1 ? tensor_writer << "]\n" : tensor_writer << ",\n";
        }
    }
}

bool CpuInferenceWorker::Convert_F32(ggml_tensor *a, ggml_tensor *b)
{
    vector<float> result;
    TensorUtil::GetFloatList(result, (const ggml_tensor)*a);
    memcpy(b->data, result.data(), result.size() * sizeof(float));
    return true;
}

ggml_tensor* CpuInferenceWorker::LayerNorm(ggml_tensor *input_tensor, ggml_tensor *norm, ggml_tensor *norm_b)
{
    const auto &model_spec = model_ptr_->spec;

    if (input_tensor == nullptr) {
        return input_tensor;
    }

    ggml_tensor *cur = input_tensor;
    if (model_spec.norm_alg == TensorNormAlg::STD) {
        cur = ggml_norm(ctx_, input_tensor);
    } else {
        cur = ggml_rms_norm(ctx_, input_tensor);
    }

    if (norm != nullptr)
    {
        ggml_tensor *norm_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, norm->ne[0]);
        Convert_F32(norm, norm_f32);
        cur = ggml_mul(ctx_, ggml_repeat(ctx_, norm_f32, cur), cur);
    }
    if (norm_b != nullptr)
    {
        ggml_tensor *norm_b_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, norm_b->ne[0]);
        Convert_F32(norm_b, norm_b_f32);
        cur = ggml_add(ctx_, ggml_repeat(ctx_, norm_b_f32, cur), cur);
    }
    return cur;
}

ggml_tensor* CpuInferenceWorker::ProcessPreLayer(struct ggml_tensor *input_tensor)
{
    const auto &model_spec = model_ptr_->spec;
    bool is_bloom = model_spec.network_structure == NetworkType::BLOOM;
    if (!is_bloom) {
        return nullptr;
    }
    const auto *network = &model_ptr_->std_network.ggml_net;

    return LayerNorm(input_tensor, network->decoder_input_norm, network->decoder_input_norm_b);
}

ggml_tensor* CpuInferenceWorker::ProcessPostLayer(ggml_tensor *input_tensor)
{
    //const auto &model_spec = model_ptr_->spec;
    const auto *network = &model_ptr_->std_network.ggml_net;

    ggml_tensor *cur = nullptr;
    cur = LayerNorm(input_tensor, network->decoder_output_norm, network->decoder_output_norm_b);

    if (network->output == nullptr) {
        cur = ggml_mul_mat(ctx_, network->decoder_embeddings, cur);
    }
    else {
        cur = ggml_mul_mat(ctx_, network->output, cur);
    }
    return cur;
}

ggml_tensor* CpuInferenceWorker::ProcessCpuLayer(int layer_idx, ggml_tensor *input_tensor)
{
    const auto &model_spec = model_ptr_->spec;
    //const auto &hparams = model_spec.hyper_params;
    const auto *layer = model_ptr_->std_network.ggml_net.decoder_layers[layer_idx];
    const auto &self_attn = layer->self_attn;
    const auto &ffn = layer->ffn;

    ggml_tensor *cur = input_tensor;

    AttentionOutput self_attn_out = ProcessCpuLayer_SelfAttention(layer_idx, cur);
    if (!model_spec.is_parallel_attn) {
        self_attn_out.output = ggml_add(ctx_, self_attn_out.output, input_tensor);
    }

    if (self_attn.post_norm != nullptr) {
        self_attn_out.output = LayerNorm(self_attn_out.output, self_attn.post_norm, self_attn.post_norm_b);
    }

    ggml_tensor *input_ff = model_spec.is_parallel_attn ? self_attn_out.pre_norm
        : (model_spec.mlp_attn_share_input ? input_tensor : self_attn_out.output);
    cur = ProcessCpuLayer_FeedForward(layer_idx, input_ff);

    cur = ggml_add(ctx_, cur, self_attn_out.output);

    if (model_spec.is_parallel_attn || model_spec.mlp_attn_share_input) {
        cur = ggml_add(ctx_, cur, input_tensor);
    }

    if (ffn.post_norm != nullptr) {
        cur = LayerNorm(cur, ffn.post_norm, ffn.post_norm_b);
    }

    return cur;
}

CpuInferenceWorker::AttentionOutput CpuInferenceWorker::ProcessCpuLayer_SelfAttention(int layer_idx, ggml_tensor *input_tensor)
{
    AttentionOutput attn_out;
    const auto &model_spec = model_ptr_->spec;
    const auto &hparams = model_spec.hyper_params;

    int token_dim = hparams.embd_dims;
    int head_num = hparams.decoder_heads;
    int head_dim = token_dim / head_num;
    int token_num = (int)input_tensor->ne[1];

    int hparams_heads = hparams.decoder_heads;
    int kv_groups = hparams.decoder_kv_heads;
    int heads_per_kv_group = hparams_heads / kv_groups;
    int kv_head_num = hparams.decoder_kv_heads;
    //int kv_sub_dim = token_dim / heads_per_kv_group;
    //int query_num = (int)query_list_.size();

    const auto *layer = model_ptr_->std_network.ggml_net.decoder_layers[layer_idx];
    const auto &self_attn = layer->self_attn;

    bool is_alibi = model_spec.pos_embedding_alg == PositionEmbeddingAlg::ALIBI;

    ggml_tensor *input_norm = input_tensor;
    if (self_attn.pre_norm != nullptr)
    {
        ggml_tensor *pre_norm = model_spec.use_self_attn_pre_norm ? self_attn.pre_norm : nullptr;
        ggml_tensor *pre_norm_b = model_spec.use_self_attn_pre_norm ? self_attn.pre_norm_b : nullptr;

        input_norm = LayerNorm(input_norm, pre_norm, pre_norm_b);
        attn_out.pre_norm = input_norm;
    }

    CurQKV cur_qkv;
    SelfAttention_CalculateCurQKV(cur_qkv, layer_idx, layer, input_norm);

    // save cur kv cache
    for (auto &query: query_list_)
    {   
        QueryProcData *query_proc_data = query_proc_data_list_[query.proc_id];
        CpuKVCacheLayer *kv_cache_layer = query_proc_data->kv_cache.Layer(layer_idx);

        int q_token_num = query.token_num;
        int q_start_row = query.start_row;
        kv_cache_layer->SetKRows(gf_, ctx_, cur_qkv.k, q_start_row, query.prefix_len, q_token_num);
        kv_cache_layer->SetVRows(gf_, ctx_, cur_qkv.v, q_start_row, query.prefix_len, q_token_num);
    }

    ggml_tensor *cur = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, token_dim, token_num);
    for (int q_idx = 0; q_idx < (int)query_list_.size(); q_idx++)
    {
        const auto &query = query_list_[q_idx];
        QueryProcData *query_proc_data = query_proc_data_list_[query.proc_id];
        CpuKVCacheLayer *kv_cache_layer = query_proc_data->kv_cache.Layer(layer_idx);

        int q_prefix_len = query.prefix_len;
        int q_token_num = query.token_num;
        int q_start_row = query.start_row;

        ggml_tensor *cur_q = ggml_view_3d(ctx_, cur_qkv.q,
                                                 cur_qkv.q->ne[0], cur_qkv.q->ne[1], q_token_num,
                                                 cur_qkv.q->nb[1], cur_qkv.q->nb[2],
                                                 q_start_row * ggml_element_size(cur_qkv.q) * cur_qkv.q->ne[0] * cur_qkv.q->ne[1]);
        cur_q = ggml_permute(ctx_, cur_q, 0, 2, 1, 3);

        ggml_tensor *k_with_ctx = nullptr;
        k_with_ctx = kv_cache_layer->GetKRows(ctx_, 0, q_token_num + q_prefix_len);

        if (heads_per_kv_group > 1) 
        {
            k_with_ctx = ggml_reshape_3d(ctx_, k_with_ctx, head_dim, kv_head_num, q_token_num + q_prefix_len);
            ggml_tensor *tmp = ggml_new_tensor_3d(ctx_, k_with_ctx->type, head_dim * heads_per_kv_group, kv_head_num, q_token_num + q_prefix_len);
            k_with_ctx = ggml_repeat(ctx_, k_with_ctx, tmp);
            k_with_ctx = ggml_reshape_3d(ctx_, k_with_ctx, head_dim, head_num, q_token_num + q_prefix_len);
        }
        else {
            k_with_ctx = ggml_reshape_3d(ctx_, k_with_ctx, head_dim, head_num, q_token_num + q_prefix_len);
        }
        k_with_ctx = ggml_permute(ctx_, k_with_ctx, 0, 2, 1, 3);
        ggml_tensor *kq = ggml_mul_mat(ctx_, k_with_ctx, cur_q); // [n_past + n, n, n_head, 1]
        ggml_tensor *kq_scaled = ggml_scale(ctx_, kq, ggml_new_f32(ctx_, 1.0f/sqrt(float(head_dim))));

        if (is_alibi) { // bloom
            kq_scaled = ggml_alibi(ctx_, kq_scaled, q_prefix_len, head_num, 8); //todo: bias max
        }

        ggml_tensor *kq_masked = ggml_diag_mask_inf_inplace(ctx_, kq_scaled, q_prefix_len);
        ggml_tensor *kq_soft_max = ggml_soft_max_inplace(ctx_, kq_masked);

        ggml_tensor *v_with_ctx = kv_cache_layer->GetVRows(ctx_, 0, q_token_num + q_prefix_len); 
        if (heads_per_kv_group > 1)
        {
            v_with_ctx = ggml_reshape_3d(ctx_, v_with_ctx, head_dim, kv_head_num, q_token_num + q_prefix_len);
            ggml_tensor *tmp = ggml_new_tensor_3d(ctx_, v_with_ctx->type, head_dim * heads_per_kv_group, kv_head_num, q_token_num + q_prefix_len);
            v_with_ctx = ggml_repeat(ctx_, v_with_ctx, tmp);
            v_with_ctx = ggml_reshape_3d(ctx_, v_with_ctx, head_dim, head_num, q_token_num + q_prefix_len);

        }
        else {
            v_with_ctx = ggml_reshape_3d(ctx_, v_with_ctx, head_dim, head_num, q_token_num + q_prefix_len);
        }

        v_with_ctx = ggml_permute(ctx_, v_with_ctx, 1, 2, 0, 3); // [n_past + n, head_dim, head_num, 1]
        ggml_tensor *v_with_ctx_trans = ggml_cpy(ctx_, v_with_ctx, ggml_new_tensor_3d(ctx_, v_with_ctx->type, q_token_num + q_prefix_len, head_dim, head_num));

        ggml_tensor *kqv = ggml_mul_mat(ctx_, v_with_ctx_trans, kq_soft_max); // [n_past + n, head_dim, head_num, 1]
        ggml_tensor *kqv_merged = ggml_permute(ctx_, kqv, 0, 2, 1, 3); // [head_dim, n_head, n_past + n, 1]
        ggml_tensor *query_cur = ggml_cpy(ctx_, kqv_merged, ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, token_dim, q_token_num));

        ggml_tensor *cur_for_cpy = ggml_view_2d(ctx_, cur, token_dim, q_token_num, cur->nb[1], q_start_row * ggml_element_size(cur) * token_dim);
        ggml_build_forward_expand(gf_, ggml_cpy(ctx_, query_cur, cur_for_cpy));
    }
    ggml_tensor *out = ggml_mul_mat(ctx_, self_attn.wo, cur);


    if (out != nullptr && self_attn.wo_b != nullptr)
    {
        struct ggml_tensor *wo_b_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, self_attn.wo_b->ne[0]); // F32
        Convert_F32(self_attn.wo_b, wo_b_f32);
        out = ggml_add(ctx_, ggml_repeat(ctx_, wo_b_f32, out), out);
    }

    attn_out.output = out;
    return attn_out;
}

ggml_tensor* CpuInferenceWorker::ProcessCpuLayer_FeedForward(int layer_idx, ggml_tensor *input_tensor)
{
    const auto &model_spec = model_ptr_->spec;

    const auto *layer = model_ptr_->std_network.ggml_net.decoder_layers[layer_idx];
    const auto &ffn = layer->ffn;

    ggml_tensor *cur = input_tensor;

    if (ffn.pre_norm != nullptr) {
        cur = LayerNorm(input_tensor, ffn.pre_norm, ffn.pre_norm_b);
    }


    ggml_tensor *t1 = ggml_mul_mat(ctx_, ffn.w1, cur);
    if (ffn.w1_b != nullptr)
    {
        ggml_tensor *w1_b_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, ffn.w1_b->ne[0]); // F32
        Convert_F32(ffn.w1_b, w1_b_f32);
        t1 = ggml_add(ctx_, ggml_repeat(ctx_, w1_b_f32, t1), t1);
    }

    ggml_tensor *t2 = nullptr;
    if (ffn.w3 != nullptr)
    {
        t2 = ggml_mul_mat(ctx_, ffn.w3, cur);
        if (ffn.w3_b != nullptr)
        {
            ggml_tensor *w3_b_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, ffn.w3_b->ne[0]); // F32
            Convert_F32(ffn.w3_b, w3_b_f32);
            t2 = ggml_add(ctx_, ggml_repeat(ctx_, w3_b_f32, t2), t2);
        }
    }

   ggml_tensor *act = nullptr;
    if (model_spec.activation_fn == ActivationFn::GELU) {
        act = ggml_gelu(ctx_, t1);
    }
    else if (model_spec.activation_fn == ActivationFn::SILU) {
        act = ggml_silu(ctx_, t1);
    }
    else
    {
        LogError("Activation function is not implemented yet.");
        return nullptr;
    }

    ggml_tensor *t3 = act;
    if (t2 != nullptr) {
        t3 = ggml_mul(ctx_, act, t2);
    }

    ggml_tensor *out = ggml_mul_mat(ctx_, ffn.w2, t3);
    if (ffn.w2_b != nullptr)
    {
        ggml_tensor *w2_b_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, ffn.w2_b->ne[0]); // F32
        Convert_F32(ffn.w2_b, w2_b_f32);
        out = ggml_add(ctx_, ggml_repeat(ctx_, w2_b_f32, out), out);
    }

    return out;
}

bool CpuInferenceWorker::SelfAttention_CalculateCurQKV(CurQKV &cur_qkv, int layer_idx,
    const StdGgmlNetwork::DecoderLayer *layer, struct ggml_tensor *input_tensor)
{
    (void)layer_idx;
    const auto &model_spec = model_ptr_->spec;
    const auto &hparams = model_spec.hyper_params;

    int hparams_heads = hparams.decoder_heads;
    int kv_groups = hparams.decoder_kv_heads;
    int heads_per_kv_group = hparams_heads / kv_groups;
    int token_dim = hparams.embd_dims;
    int kv_sub_dim = token_dim / heads_per_kv_group;
    int head_num = hparams_heads; // hparams_heads
    int kv_head_num = head_num / heads_per_kv_group; // kv_groups
    int head_dim = token_dim / hparams_heads;
    int token_num = (int)input_tensor->ne[1];
    auto data_type = input_tensor->type;

    const auto &self_attn = layer->self_attn;

    ggml_tensor *cur = nullptr;
    bool is_rope = model_spec.pos_embedding_alg == PositionEmbeddingAlg::ROPE;
    int rope_type = model_spec.qk_column_order;
    if (self_attn.qkv)
    {
        cur = ggml_mul_mat(ctx_, self_attn.qkv, input_tensor);
        if (cur != nullptr && self_attn.qkv_b != nullptr)
        {
            ggml_tensor *qkv_b_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, self_attn.qkv_b->ne[0]); // F32
            Convert_F32(self_attn.qkv_b, qkv_b_f32);
            cur = ggml_add(ctx_, ggml_repeat(ctx_, qkv_b_f32, cur), cur);
        }
        if (model_spec.qkv_format == 1) { // for ggml
            cur_qkv.q = ggml_view_2d(ctx_, cur, token_dim, token_num, cur->nb[1], 0 * ggml_element_size(cur) * token_dim);
            cur_qkv.k = ggml_view_2d(ctx_, cur, kv_sub_dim, token_num, cur->nb[1], 1 * ggml_element_size(cur) * token_dim);
            cur_qkv.v = ggml_view_2d(ctx_, cur, kv_sub_dim, token_num, cur->nb[1], ggml_element_size(cur) * (token_dim + kv_sub_dim));
        } else {
            if (heads_per_kv_group > 1)
            {
                int h = heads_per_kv_group;
                ggml_tensor *tmp = ggml_reshape_2d(ctx_, cur, head_dim * (h + 2), kv_groups * token_num);
                cur_qkv.q = ggml_view_2d(ctx_, tmp, h * head_dim, kv_groups * token_num, tmp->nb[1], 0 * ggml_element_size(tmp) * h * head_dim);
                cur_qkv.k = ggml_view_2d(ctx_, tmp, head_dim, kv_groups * token_num, tmp->nb[1], 1 * ggml_element_size(tmp) * h * head_dim);
                cur_qkv.v = ggml_view_2d(ctx_, tmp, head_dim, kv_groups * token_num, tmp->nb[1], 0 * ggml_element_size(tmp) * (h + 1) * head_dim);
            }
            else
            {
                ggml_tensor *tmp = ggml_reshape_4d(ctx_, cur, head_dim, 3, head_num, token_num);
                tmp = ggml_permute(ctx_, tmp, 0, 2, 1, 3);
                ggml_tensor *tmp_cpy = ggml_new_tensor_4d(ctx_, tmp->type, head_dim, head_num, 3, token_num);
                ggml_build_forward_expand(gf_, ggml_cpy(ctx_, tmp, tmp_cpy));
                cur = ggml_reshape_3d(ctx_, tmp_cpy, head_dim * head_num, 3, token_num);

                cur_qkv.q = ggml_view_2d(ctx_, cur, head_dim * head_num, token_num, cur->nb[2], 0 * ggml_element_size(cur) * head_dim * head_num);
                cur_qkv.k = ggml_view_2d(ctx_, cur, head_dim * head_num, token_num, cur->nb[2], 1 * ggml_element_size(cur) * head_dim * head_num);
                cur_qkv.v = ggml_view_2d(ctx_, cur, head_dim * head_num, token_num, cur->nb[2], 2 * ggml_element_size(cur) * head_dim * head_num);
            }
        }
    } 
    else 
    {
        cur_qkv.q = ggml_mul_mat(ctx_, self_attn.wq, input_tensor);
        cur_qkv.k = ggml_mul_mat(ctx_, self_attn.wk, input_tensor);
        cur_qkv.v = ggml_mul_mat(ctx_, self_attn.wv, input_tensor);
    }
    
    cur_qkv.q = ggml_cpy(ctx_, cur_qkv.q, ggml_new_tensor_3d(ctx_, cur_qkv.q->type, head_dim, head_num, token_num));
    if (is_rope)
    {
        ggml_tensor *q_rope = ggml_new_tensor_3d(ctx_, data_type, head_dim, head_num, token_num);
        for (const auto &query : query_list_)
        {
            SelfAttention_CalculateQueryRope(cur_qkv.q, q_rope, query.token_num, query.start_row, query.prefix_len, rope_type);
        }
        cur_qkv.q = q_rope;
    }
    cur_qkv.k = ggml_cpy(ctx_, cur_qkv.k, ggml_new_tensor_3d(ctx_, cur_qkv.k->type, head_dim, kv_head_num, token_num));
    if (is_rope)
    {
        ggml_tensor *k_rope = ggml_new_tensor_3d(ctx_, data_type, head_dim, kv_head_num, token_num);
        for (const auto &query : query_list_)
        {
            SelfAttention_CalculateQueryRope(cur_qkv.k, k_rope, query.token_num, query.start_row, query.prefix_len, rope_type);
        }
        cur_qkv.k = k_rope;
    }
    cur_qkv.k = ggml_cpy(ctx_, cur_qkv.k, ggml_new_tensor_2d(ctx_, cur_qkv.k->type, head_dim * kv_head_num, token_num));
    cur_qkv.v = ggml_cpy(ctx_, cur_qkv.v, ggml_new_tensor_2d(ctx_, cur_qkv.v->type, head_dim * kv_head_num, token_num));

    return true;
}

ggml_tensor* CpuInferenceWorker::SelfAttention_CalculateQueryRope(ggml_tensor *input_tensor,
    struct ggml_tensor *output_tensor, int token_num, int start_row, int n_past, int rope_type)
{
    int head_dim = (int)input_tensor->ne[0];
    int head_num = (int)input_tensor->ne[1];
    ggml_tensor *cur = ggml_view_3d(ctx_, input_tensor, 
                                    input_tensor->ne[0], input_tensor->ne[1], token_num, 
                                    input_tensor->nb[1], input_tensor->nb[2], 
                                    start_row * ggml_element_size(input_tensor) * input_tensor->ne[0] * input_tensor->ne[1]);

    if (rope_type == 2) {
        ggml_tensor *cur_cpy = ggml_new_tensor_3d(ctx_, cur->type, head_dim, head_num, token_num);
        ggml_build_forward_expand(gf_, ggml_cpy(ctx_, cur, cur_cpy));
        ggml_tensor *tmp = ggml_reshape_4d(ctx_, cur_cpy, head_dim / 2, 2, head_num, token_num);
        tmp = ggml_permute(ctx_, tmp, 1, 0, 2, 3);
        ggml_tensor *tmp_cpy = ggml_new_tensor_4d(ctx_, tmp->type, 2, head_dim / 2, head_num, token_num);
        ggml_build_forward_expand(gf_, ggml_cpy(ctx_, tmp, tmp_cpy));
        cur = ggml_reshape_3d(ctx_, tmp_cpy, head_dim, head_num, token_num);
    }

    ggml_tensor *result = ggml_view_3d(ctx_, output_tensor, 
                                    output_tensor->ne[0], output_tensor->ne[1], token_num, 
                                    output_tensor->nb[1], output_tensor->nb[2], 
                                    start_row * ggml_element_size(output_tensor) * output_tensor->ne[0] * output_tensor->ne[1]); // todo

    cur = ggml_rope_inplace(ctx_, cur, n_past, (int)input_tensor->ne[0], 0); 
    ggml_build_forward_expand(gf_, ggml_cpy(ctx_, cur, result));
    return cur;
}

TRANSFORMER_END
INFER_FLOW_END