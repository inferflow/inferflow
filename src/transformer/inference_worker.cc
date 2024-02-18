#include "inference_worker.h"
#include <sstream>
#include "sslib/log.h"
#include "tensor/tensor_util.h"
#include "tensor/device_tensor_util.h"
#include "common/quantization.h"
#include "query_state_table.h"
#include "common/cuda_util.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

////////////////////////////////////////////////////////////////////////////////
// class GpuInferenceWorker

GpuInferenceWorker::~GpuInferenceWorker()
{
    if (device_token_id_array_ != nullptr)
    {
        cudaFree(device_token_id_array_);
        device_token_id_array_ = nullptr;
    }
}

bool GpuInferenceWorker::Init(int id, int worker_num, int group_id, int group_num,
    const InferenceConfig &cfg, const InferenceConfigEx &config_ex, TransformerModel &model,
    int device_id, const LayerRange encoder_layer_range, const LayerRange decoder_layer_range,
    bool is_by_layer)
{
    id_ = id;
    worker_num_ = worker_num;
    group_id_ = group_id;
    group_num_ = group_num;
    config_ = &cfg;
    config_ex_ = &config_ex;
    model_ptr_ = &model;
    device_id_ = device_id;
    encoder_layer_range_ = encoder_layer_range;
    decoder_layer_range_ = decoder_layer_range;
    global_encoder_end_layer_ = encoder_layer_range.layer_num;
    global_decoder_end_layer_ = decoder_layer_range.layer_num;
    is_by_layer_ = is_by_layer;
    is_quant_tensor_exchange_ = false;
    const auto &model_spec = model_ptr_->spec;

    layer_idx_for_study_ = 0;

    int device_id_bak = CudaUtil::GetDevice();
    CudaUtil::SetDevice(device_id_);

    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(model.spec.network_structure);
    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(model.spec.network_structure);

    const auto &hparams = model_ptr_->spec.hyper_params;
    int max_context_len = model_ptr_->spec.max_context_len;
    int max_head_num = max(hparams.decoder_heads, hparams.encoder_heads);
    soft_max_aux_tensor_.New(ElementType::F16, max_head_num * max_context_len);

    int fp16_size = sizeof(inferflow_fp16);
    bool is_last = group_id_ + 1 >= group_num_;

    aux_memory_size_ = 0;
    uint64_t mem_size = 160 * 1024 * 1024; //160MB
    if (is_last)
    {
        uint64_t mem_size2 = model_spec.max_context_len * hparams.vocab_size * fp16_size;
        //LogKeyInfo("mem_size: %d, mem_size2: %d", mem_size, mem_size2);
        if (mem_size2 > mem_size / 2) {
            mem_size = mem_size / 2 + mem_size2;
        }
    }
    for (int idx = 0; idx < LOCAL_DEVICE_HEAP_NUM; idx++) {
        local_device_heaps_[idx].Init(mem_size);
    }
    aux_memory_size_ += (mem_size * LOCAL_DEVICE_HEAP_NUM);

    mem_size = 16 * 1024 * 1024;
    local_device_heap_.Init(mem_size);
    aux_memory_size_ += mem_size;

    mem_size = 320 * 1024 * 1024; //320MB
    if (model_spec.max_input_len > 1024) {
        mem_size *= ((model_spec.max_context_len + 1023) / 1024);
    }
    if (is_last)
    {
        uint64_t mem_size2 = model_spec.max_context_len * hparams.vocab_size * fp16_size;
        //LogKeyInfo("mem_size: %d, mem_size2: %d", mem_size, mem_size2);
        if (mem_size2 > mem_size / 2) {
            mem_size = mem_size / 2 + mem_size2;
        }
    }
    layer_local_device_heap_.Init(mem_size);
    aux_memory_size_ += mem_size;

    int buffer_len = hparams.embd_dims * 256;
    aux_buffer_.New(buffer_len);
    aux_memory_size_ += (buffer_len * fp16_size);

    bool ret = true;
    if (device_token_id_array_ == nullptr)
    {
        auto ret_code = cudaMalloc((void**)&device_token_id_array_,
            model_ptr_->spec.max_context_len * sizeof(int));
        ret = CudaUtil::CheckReturnCode(ret_code, "cudaMalloc");
        Macro_RetFalseIf(!ret);
    }

    ret = cublas_engine_.Init();
    is_cublas_engine_initialized_ = true;
    Macro_RetxFalseIf(!ret, LogError("Failed to initialize the cublas_engine"));

    int kv_cache_dim = hparams.embd_dims;
    if (!is_encoder_only && hparams.decoder_kv_heads != hparams.decoder_heads) {
        kv_cache_dim = hparams.embd_dims / hparams.decoder_heads * hparams.decoder_kv_heads;
    }
    k_cache_item_.New(ElementType::F16, kv_cache_dim, max_context_len);
    v_cache_item_.New(ElementType::F16, kv_cache_dim, max_context_len);

    if (!is_by_layer) {
        kv_cache_dim /= worker_num;
    }

    float kv_cache_memory_cost[2] = {0, 0}; //0: device, 1: host
    int max_batch_size = min(1 + QueryState::MAX_PROC_ID, cfg.max_concurrent_queries);
    for (int proc_id = 0; proc_id < max_batch_size; proc_id++)
    {
        //if (!is_by_layer && proc_id % worker_num != id_)
        //{
        //    query_proc_data_list_.push_back(nullptr);
        //    continue;
        //}

        QueryProcData *proc_data = new QueryProcData;
        query_proc_data_list_.push_back(proc_data);

        int start_layer = is_encoder_only ? encoder_layer_range_.start
            : (is_decoder_only ? decoder_layer_range_.start
                : min(encoder_layer_range_.start, decoder_layer_range_.start));
        int end_layer = is_encoder_only ? encoder_layer_range_.end
            : (is_decoder_only ? decoder_layer_range_.end
                : max(encoder_layer_range_.end, decoder_layer_range_.end));

        int r = 0;//is_by_layer ? 0 : id_;
        int m = 1;//is_by_layer ? 1 : worker_num;
        ret = proc_data->kv_cache.Init(model_spec.device_kv_cache_data_type,
            max_context_len, kv_cache_dim, start_layer, end_layer,
            model_spec.host_kv_cache_percent, r, m);
        if (!ret)
        {
            LogError("Worker %d: Failed to initialize the KV cache", id_);
            return false;
        }

        if (!is_encoder_only && !is_decoder_only && model_spec.hyper_params.encoder_layers > 0
            && model_spec.has_cross_attn_kv_cache)
        {
            ret = proc_data->cross_attn_kv_cache.Init(model_spec.device_kv_cache_data_type,
                max_context_len, kv_cache_dim, decoder_layer_range_.start,
                decoder_layer_range_.end, model_spec.host_kv_cache_percent, r, m);
            if (!ret)
            {
                LogError("Worker %d: Failed to initialize the cross-attention KV cache", id_);
                return false;
            }
        }

        kv_cache_memory_cost[0] += proc_data->kv_cache.DeviceMemoryCost();
        kv_cache_memory_cost[1] += proc_data->kv_cache.HostMemoryCost();
        kv_cache_memory_cost[0] += proc_data->cross_attn_kv_cache.DeviceMemoryCost();
        kv_cache_memory_cost[1] += proc_data->cross_attn_kv_cache.HostMemoryCost();
    }

    //if (cfg.debug.is_study_mode)
    {
        LogKeyInfo("Device: %d; KV cache memory cost (GB): %.2f (device), %.2f (host)",
            device_id_, kv_cache_memory_cost[0] / 1024, kv_cache_memory_cost[1] / 1024);
    }

    //const auto &model_spec = config_->models[0];
    bool is_gpu_quant = TensorCommon::IsQuantType(model_spec.device_weight_data_type);
    if (is_gpu_quant)
    {
        int max_tensor_size = model_ptr_->std_network.device_net.MaxTensorSize();
        for (const auto *sub_net : model_ptr_->std_network.device_sub_nets)
        {
            int sub_max = sub_net->MaxTensorSize();
            max_tensor_size = max(max_tensor_size, sub_max);
        }

        //LogKeyInfo("max_tensor_size: %d", max_tensor_size);
        dequant_tensor_ex_.tensor = &dequant_tensor_;
        dequant_tensor_.New(ElementType::F16, max_tensor_size);
        //LogKeyInfo("Worker %d: Building the dequant layer...", id_);
        //BuildDequantLayer();
    }

    CudaUtil::SetDevice(device_id_bak);
    return ret;
}

bool GpuInferenceWorker::SetGlobalData(GpuInfGlobalData &global_data,
    const vector<int> &devices)
{
    devices_ = devices;
    global_data_ = &global_data;
    return true;
}

void GpuInferenceWorker::SetInput(const DeviceTensor *input, InferencePerfStat &perf_stat,
    ostream *tensor_writer, const vector<QueryProcInput> &query_list,
    const vector<int> &token_list, int global_end_layer, bool is_encoder)
{
    is_encoder_ = is_encoder;
    input_tensor_ = input;
    perf_stat_ = &perf_stat;
    tensor_writer_ = tensor_writer;
    query_list_ = query_list;
    token_id_list_ = token_list;
    if (global_end_layer > 0 && is_encoder) {
        global_encoder_end_layer_ = min(global_end_layer, encoder_layer_range_.layer_num);
    }
    if (global_end_layer > 0 && !is_encoder) {
        global_decoder_end_layer_ = min(global_end_layer, decoder_layer_range_.layer_num);
    }
}

void GpuInferenceWorker::Run()
{
    const auto &model_spec = model_ptr_->spec;
    auto net_type = model_spec.network_structure;
    bool is_encoder_decoder = NetworkStructure::IsEncoderDecoderTransformer(net_type);
    const StdDeviceNetwork *device_net = GetDeviceNet();

    output_tensor_ = nullptr;
    CudaUtil::SetDevice(device_id_);
    //LogKeyInfo("id: %d, device_id: %d, layer range: [%d, %d)",
    //    id_, device_id_, start_layer_, end_layer_);

    //if (!is_cublas_engine_initialized_)
    //{
    //    bool ret = cublas_engine_.Init();
    //    is_cublas_engine_initialized_ = true;
    //    Macro_RetxVoidIf(!ret, LogError("Failed to initialize the cublas_engine"));
    //}

    int heap_idx = 0;
    ClearLocalMemory();

    if (model_spec.pos_embedding_alg == PositionEmbeddingAlg::SINUSOIDAL
        || model_spec.pos_embedding_alg == PositionEmbeddingAlg::SINUSOIDAL2)
    {
        int bytes = (int)(token_id_list_.size() * sizeof(int));
        CudaUtil::HostToDeviceMemcpy(device_token_id_array_, token_id_list_.data(), bytes);
    }

    //DeviceTensor *layer_input = input_tensor_;
    ElementType data_type = input_tensor_->data_type;
    int cx = input_tensor_->ne[0], cy = input_tensor_->ne[1], cz = input_tensor_->ne[2];
    int input_byte_count = (int)(input_tensor_->bytes_per_row * cy * cz);
    DeviceTensor *layer_input = CreateLocalTensor(data_type,
        cx, cy, cz, false, heap_idx);
    layer_input->CopyFromDevice((const void*)input_tensor_->data, input_byte_count);

    if (config_->debug.is_study_mode && config_->debug.show_tensors) {
        PrintTensor(layer_input, 8, 8, 8, "input token embeddings:\n");
    }

    InputKV input_kv;
    if (is_encoder_decoder && !is_encoder_)
    {
        BuildInputKV(input_kv, data_type);
        if (config_->debug.is_study_mode && config_->debug.show_tensors) {
            PrintTensor(input_kv.tensor, 8, 8, 8, "input_kv:\n");
        }
    }

    const auto &layer_range = is_encoder_ ? encoder_layer_range_ : decoder_layer_range_;
    int global_end_layer = is_encoder_ ? global_encoder_end_layer_ : global_decoder_end_layer_;
    if (layer_range.start == 0)
    {
        heap_idx = (heap_idx + 1) % LOCAL_DEVICE_HEAP_NUM;
        local_device_heaps_[heap_idx].Clear();

        DeviceTensor *out = ProcessTransLayer(device_net->input_transform, layer_input, heap_idx);
        layer_input = out;

        out = ProcessPreLayer(layer_input, heap_idx);
        if (out != nullptr) {
            layer_input = out;
        }
    }

    //const QueryProcInput &query = query_list_[0];
    int end_layer = min(layer_range.end, global_end_layer);
    for (int layer_idx = layer_range.start; layer_idx < layer_range.end; layer_idx++)
    {
        if (config_->debug.is_study_mode) {
            //LogKeyInfo("worker %d, layer %d", id_, layer_idx);
        }
        TaskMonitor tm;
        heap_idx = (heap_idx + 1) % LOCAL_DEVICE_HEAP_NUM;
        local_device_heaps_[heap_idx].Clear();

        DeviceTensor *layer_output = ProcessGpuLayer(layer_idx, layer_input, input_kv, heap_idx);
        layer_input = layer_output;
        output_tensor_ = layer_output;
        if (layer_output == nullptr) {
            return;
        }

        if (config_->debug.enable_perf_stat && layer_idx <= 5)
        {
            int perf_base = (layer_idx + 1) * 10000;
            UpdatePerfStat(perf_base + 0, tm);
        }
    }

    bool is_last = end_layer == layer_range.layer_num || end_layer == global_end_layer;
    if (is_last)
    {
        heap_idx = (heap_idx + 1) % LOCAL_DEVICE_HEAP_NUM;
        local_device_heaps_[heap_idx].Clear();

        DeviceTensor *out = ProcessTransLayer(device_net->output_transform, layer_input, heap_idx);
        layer_input = out;

        out = ProcessPostLayer(layer_input, heap_idx);
        output_tensor_ = out;
    }

    //LogKeyInfo("End of worker %d. device: %d, is_last: %s",
    //    id_, device_id_, is_last ? "Y" : "N");
}

void GpuInferenceWorker::CancelThread()
{
}

const StdDeviceNetwork* GpuInferenceWorker::GetDeviceNet() const
{
    const StdDeviceNetwork *device_net = nullptr;
    if (is_by_layer_) {
        device_net = &model_ptr_->std_network.device_net;
    }
    else {
        device_net = model_ptr_->std_network.device_sub_nets[id_];
    }
    return device_net;
}

void GpuInferenceWorker::BuildInputKV(InputKV &input_kv, ElementType data_type)
{
    int kv_cx = 0, kv_cy = 0, kv_cz = 1;
    for (const auto &query : query_list_)
    {
        int rows = query.state->encoder_output.ne[1];
        PairUInt32 item(kv_cy, rows);
        input_kv.query_list.push_back(item);

        kv_cx = query.state->encoder_output.ne[0];
        kv_cy += rows;
    }

    input_kv.tensor = CreateLocalTensor(data_type, kv_cx, kv_cy, kv_cz, false, -1);

    int start_y = 0;
    for (const auto &query : query_list_)
    {
        int bytes = (int)query.state->encoder_output.ByteCount();
        const void *source_data = query.state->encoder_output.data;
        int offset = (int)TensorCommon::ByteCount(data_type, kv_cx * start_y);
        uint8_t *target_data = ((uint8_t*)input_kv.tensor->data) + offset;
        CudaUtil::DeviceToDeviceMemcpy(target_data, source_data, bytes);

        start_y += query.state->encoder_output.ne[1];
    }
}

//is_pos:
//  true: position_embedding
//  false: token_type_embedding
bool GpuInferenceWorker::GetEmbeddingTensor(DeviceTensor &embd_tensor,
    bool is_encoder, bool is_pos)
{
    bool ret = true;
    //const auto &hparams = model_ptr_->spec.hyper_params;
    int token_num = (int)token_id_list_.size();
    const auto &net = model_ptr_->std_network;

    DeviceTensor *src_tensor = nullptr;
    if (is_encoder)
    {
        src_tensor = is_pos ? net.device_net.encoder_pos_embeddings
            : net.device_net.encoder_token_type_embeddings;
    }
    else
    {
        src_tensor = is_pos ? net.device_net.decoder_pos_embeddings : nullptr;
    }

    if (src_tensor == nullptr) {
        return false;
    }

    embd_tensor.data_type = ElementType::F16;
    embd_tensor.data = local_device_heap_.NewHalfArray(token_num * src_tensor->ne[0]);
    embd_tensor.SetStructure(src_tensor->ne[0], token_num);

    int offset = model_ptr_->spec.pos_embedding_offset;
    int query_num = (int)query_list_.size();
    for (int query_idx = 0; ret && query_idx < query_num; query_idx++)
    {
        const auto &query = query_list_[query_idx];
        //LogKeyInfo("query.prefix_len: %d", query.prefix_len);
        int byte_num = sizeof(half) * src_tensor->ne[0];
        for (int token_idx = 0; ret && token_idx < query.token_num; token_idx++)
        {
            void *target_row = embd_tensor.RowData(query.start_row + token_idx);
            int src_row_idx = is_pos ? token_idx + query.prefix_len + offset : 0;
            const void *src_row = src_tensor->RowData(src_row_idx);
            ret = CudaUtil::DeviceToDeviceMemcpy(target_row, src_row, byte_num);
        }
    }

    return ret;
}

DeviceTensor* GpuInferenceWorker::ProcessPreLayer(const DeviceTensor *layer_input,
    int heap_idx)
{
    bool ret = true;
    //bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;
    const auto &hparams = model_ptr_->spec.hyper_params;
    if (config_->debug.is_study_mode) {
        PrintTensor(layer_input, 8, 8, 8, "input (0):\n");
    }

    const auto &model_spec = model_ptr_->spec;
    if (model_spec.pos_embedding_alg == PositionEmbeddingAlg::SINUSOIDAL
        || model_spec.pos_embedding_alg == PositionEmbeddingAlg::SINUSOIDAL2)
    {
        DeviceTensor *layer_norm = CreateLocalTensor(*layer_input, false, heap_idx);
        if (model_spec.has_linear_norm_before_sinusoidal)
        {
            TensorNormAlg norm_alg = TensorNormAlg::LINEAR;
            TensorOpr::LayerNormalization(*layer_norm, *layer_input, norm_alg);
        }
        else
        {
            int bytes = (int)layer_input->ByteCount();
            CudaUtil::DeviceToDeviceMemcpy(layer_norm->data, layer_input->data, bytes);
        }

        if (config_->debug.is_study_mode) {
            PrintTensor(layer_norm, 8, 8, 8, "layer_norm (100):\n");
        }

        PosEmbeddingParams pos_embd_params;
        pos_embd_params.dims = hparams.embd_dims;
        pos_embd_params.order_type = model_ptr_->spec.qk_column_order;
        pos_embd_params.alg = model_spec.pos_embedding_alg;
        pos_embd_params.rope_theta = model_spec.rope_theta;
        pos_embd_params.partial_rotary_factor = model_spec.partial_rotary_factor;

        pos_embd_params.device_token_id_array = device_token_id_array_;
        for (const auto &query : query_list_)
        {
            //pos_embd_params.context_len = query.prefix_len + 2;
            pos_embd_params.context_len = query.prefix_len;
            int start_z = query.start_row;
            int z_num = 1; //to do: check it
            TensorOpr::PositionEmbedding(*layer_norm, pos_embd_params, start_z, z_num);
        }

        if (config_->debug.is_study_mode) {
            PrintTensor(layer_norm, 8, 8, 8, "layer_norm (101):\n");
        }
        return layer_norm;
    }

    const StdDeviceNetwork *device_net = GetDeviceNet();
    DeviceTensor *new_input = nullptr;
    DeviceTensor *out_tensor = CreateLocalTensor(*layer_input, false, heap_idx);
    if (!is_encoder_ && device_net->decoder_pos_embeddings != nullptr)
    {
        DeviceTensor embd_tensor(false);
        GetEmbeddingTensor(embd_tensor, false, true);
        if (config_->debug.is_study_mode) {
            PrintTensor(&embd_tensor, 8, 8, 8, "embed_positions:\n");
        }

        TensorOpr::Add(*out_tensor, *layer_input, embd_tensor);
        new_input = out_tensor;
    }

    const auto *input_norm = is_encoder_ ? device_net->encoder_input_norm
        : device_net->decoder_input_norm;
    const auto *input_norm_b = is_encoder_ ? device_net->encoder_input_norm_b
        : device_net->decoder_input_norm_b;
    bool is_null_input_norm = input_norm == nullptr;
    if (is_null_input_norm) {
        return new_input;
    }

    const DeviceTensor *norm_input = new_input == nullptr ? layer_input : new_input;
    if (is_encoder_ && device_net->encoder_token_type_embeddings != nullptr
        && device_net->encoder_pos_embeddings != nullptr)
    {
        DeviceTensor embd_tensor(false);
        GetEmbeddingTensor(embd_tensor, true, false);
        TensorOpr::Add(*out_tensor, *layer_input, embd_tensor);

        PrintTensor(out_tensor, 8, 8, 8, "new_input-1:\n");

        GetEmbeddingTensor(embd_tensor, true, true);
        TensorOpr::Add(*out_tensor, *out_tensor, embd_tensor);
        new_input = out_tensor;
        norm_input = new_input;

        //PrintTensor(&embd_tensor, 8, 8, 8, "pos-embeddings:\n");
        PrintTensor(new_input, 8, 8, 8, "new_input-2:\n");
    }

    TensorOpr::LayerNormalization(*out_tensor, *norm_input, model_spec.norm_alg,
        input_norm, input_norm_b);
    new_input = out_tensor;

    if (new_input != nullptr && config_->debug.is_study_mode) {
        PrintTensor(new_input, 8, 8, 8, "pre_layer_output:\n");
    }
    return ret ? new_input : nullptr;
}

DeviceTensor* GpuInferenceWorker::ProcessPostLayer(DeviceTensor *layer_input,
    int heap_idx)
{
    bool ret = true;
    int cy = input_tensor_->ne[1], cz = input_tensor_->ne[2];
    const auto &model_spec = model_ptr_->spec;
    //const auto &hparams = model_spec.hyper_params;
    const StdDeviceNetwork *device_net = GetDeviceNet();
    bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;

    ClearLayerLocalMemory();

    const auto *output_norm = is_encoder_ ? device_net->encoder_output_norm
        : device_net->decoder_output_norm;

    DeviceTensor *norm = layer_input;
    if (output_norm != nullptr)
    {
        norm = CreateLocalTensor(*layer_input, false, heap_idx);
        TensorOpr::LayerNormalization(*norm, *layer_input, model_spec.norm_alg);

        if (config_->debug.is_study_mode) {
            PrintTensor(norm, 8, 8, 8, "norm (1000901):\n");
        }

        //DeviceTensor *norm2 = CreateLocalTensor(*layer_input, false);
        const auto *output_norm_b = is_encoder_ ? device_net->encoder_output_norm_b
            : device_net->decoder_output_norm_b;
        ret = TensorOpr::Mul(*norm, *norm, *output_norm);
        if (ret && output_norm_b != nullptr) {
            ret = TensorOpr::Add(*norm, *norm, *output_norm_b);
        }
    }

    DeviceTensor *norm2 = norm;
    if (config_->debug.is_study_mode) {
        PrintTensor(norm2, 8, 8, 8, "norm (1000905):\n");
    }

    DeviceTensor *out = is_encoder_ ? norm2 : nullptr;
    TaskMonitor tm;
    //int vocab_size = hparams.output_vocab_size > 0 ? hparams.output_vocab_size : hparams.vocab_size;
    if (ret && !is_encoder_ && (is_by_layer_ || id_ + 1 == worker_num_))
    {
        //MatrixMulAlg matrix_mul_alg = MatrixMulAlg::Alg2;
        int scenario_id = 90;
        //do not use vocab_size here (vocab_size may be smaller)
        int out_cx = device_net->output->ne[1];
        out = CreateLocalTensor(ElementType::F16, out_cx, cy * cz, 0,
            false, heap_idx, scenario_id);
        bool use_full_quant_gemv = false;
        if (device_net->output_quant != nullptr) {
            use_full_quant_gemv = GetUseFullQuantGemv(*norm2, *device_net->output_quant);
        }
        if (use_full_quant_gemv)
        {
            auto *norm2_quant = CreateLocalTensor(ElementType::Q8_B32T2,
                norm2->ne[0], norm2->ne[1], 0, true, 0);
            TensorOpr::Quantize(*norm2_quant, *norm2);
            ret = MatrixMultiplication(*out, *norm2_quant, *device_net->output_quant,
                is_b_column_major, device_net->output_b);
        }
        else
        {
            ret = MatrixMultiplication(*out, *norm2, *device_net->output,
                is_b_column_major, device_net->output_b);
        }

        if (config_->debug.is_study_mode)
        {
            PrintTensor(device_net->output, 8, 8, 8, "lm_head:\n");
            PrintTensor(out, 8, 16, 8, "output (1000906):\n");
        }

        // int out_cols = out->Columns();
        //TensorOpr::AssignColumns(*out, 0, out_cols / 2, -1);
        //TensorOpr::AssignColumns(*out, 0, out_cols / 4, -1);
        //TensorOpr::AssignColumns(*out, 0, 0, out_cols / 2);
        //TensorOpr::AssignColumns(*out, 0, out_cols * 1 / 4, out_cols / 4);
        //TensorOpr::AssignColumns(*out, 0, out_cols * 2 / 4, out_cols / 4);
    }

    if (ret && is_encoder_ && device_net->encoder_output != nullptr
        && (is_by_layer_ || id_ + 1 == worker_num_))
    {
        out = CreateLocalTensor(ElementType::F16, device_net->encoder_output->ne[1],
            cy * cz, 0, false, heap_idx);
        ret = MatrixMultiplication(*out, *norm2, *device_net->encoder_output, is_b_column_major);
        if (device_net->encoder_output_b != nullptr) {
            TensorOpr::Add(*out, *out, *device_net->encoder_output_b);
        }

        if (config_->debug.is_study_mode)
        {
            PrintTensor(out, 8, 16, 8, "output (1000907):\n");
        }
    }

    if (device_net->encoder_output_post_norm != nullptr)
    {
        DeviceTensor *act = CreateLocalTensor(*out, false, heap_idx);
        TensorOpr::Activation(*act, *out, model_spec.activation_fn);

        auto *post_norm = CreateLocalTensor(*out, false, heap_idx);
        TensorOpr::LayerNormalization(*post_norm, *act, model_spec.norm_alg,
            device_net->encoder_output_post_norm, device_net->encoder_output_post_norm_b);
        out = post_norm;

        if (config_->debug.is_study_mode)
        {
            PrintTensor(act, 8, 16, 8, "act (1000909):\n");
            PrintTensor(out, 8, 16, 8, "output (1000910):\n");
        }
    }

    if (config_->debug.enable_perf_stat) {
        UpdatePerfStat(1000009, tm);
    }

    Macro_RetIf(nullptr, !ret);
    return out;
}

DeviceTensor* GpuInferenceWorker::ProcessTransLayer(
    const StdDeviceNetwork::SimpleLayer &trans_layer,
    DeviceTensor *layer_input, int heap_idx)
{
    bool ret = true;
    int cy = input_tensor_->ne[1], cz = input_tensor_->ne[2];
    const auto &model_spec = model_ptr_->spec;
    //const auto &hparams = model_spec.hyper_params;
    //const StdDeviceNetwork *device_net = GetDeviceNet();
    bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;

    const auto &pre_norm = trans_layer.pre_norm;
    const auto &dense = trans_layer.dense;
    const auto &post_norm = trans_layer.post_norm;

    ClearLayerLocalMemory();

    DeviceTensor *layer_output = layer_input;
    if (pre_norm.weight != nullptr)
    {
        layer_output = CreateLocalTensor(*layer_input, false, heap_idx);
        TensorOpr::LayerNormalization(*layer_output, *layer_input,
            model_spec.norm_alg, pre_norm.weight, pre_norm.bias);

        if (config_->debug.is_study_mode) {
            PrintTensor(layer_output, 8, 8, 8, "trans_layer.norm_out:\n");
        }
    }

    if (dense.weight != nullptr)
    {
        DeviceTensor *dense_out = CreateLocalTensor(ElementType::F16, dense.weight->ne[1],
            cy * cz, 0, false, heap_idx);
        ret = MatrixMultiplication(*dense_out, *layer_output, *dense.weight, is_b_column_major);
        Macro_RetIf(nullptr, !ret);
        if (dense.bias != nullptr) {
            TensorOpr::Add(*dense_out, *dense_out, *dense.bias);
        }

        layer_output = dense_out;
        if (config_->debug.is_study_mode)
        {
            PrintTensor(dense_out, 8, 16, 8, "trans_layer.dense_out:\n");
        }
    }

    if (post_norm.weight != nullptr)
    {
        DeviceTensor *act = CreateLocalTensor(*layer_output, false, heap_idx);
        TensorOpr::Activation(*act, *layer_output, model_spec.activation_fn);

        auto *post_tensor = CreateLocalTensor(*layer_output, false, heap_idx);
        TensorOpr::LayerNormalization(*post_tensor, *act, model_spec.norm_alg,
            post_norm.weight, post_norm.bias);
        layer_output = post_tensor;

        if (config_->debug.is_study_mode)
        {
            PrintTensor(act, 8, 16, 8, "trans_layer.act:\n");
            PrintTensor(layer_output, 8, 16, 8, "trans_layer.output:\n");
        }
    }

    return layer_output;
}

void GpuInferenceWorker::ClearLocalMemory()
{
    for (int idx = 0; idx < LOCAL_DEVICE_HEAP_NUM; idx++) {
        local_device_heaps_[idx].Clear();
    }
    local_device_heap_.Clear();
    layer_local_device_heap_.Clear();
    local_device_tensor_heap_.Clear();
}

void GpuInferenceWorker::ClearLayerLocalMemory()
{
    layer_local_device_heap_.Clear();
}

DeviceTensor* GpuInferenceWorker::ProcessGpuLayer(int layer_idx,
    const DeviceTensor *layer_input, const InputKV &input_kv,
    int heap_idx)
{
    const auto &model_spec = model_ptr_->spec;
    //LogKeyInfo("Worker %d: Start of layer %d (heap_idx = %d)", id_, layer_idx, heap_idx);
    //int token_num = layer_input->ne[1];
    const StdDeviceNetwork *device_net = GetDeviceNet();
    int layer_base_phase = (layer_idx + 1) * 100;
    //bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;
    const StdDeviceNetwork::EncoderLayer *encoder_layer = is_encoder_
        ? device_net->encoder_layers[layer_idx] : nullptr;
    const StdDeviceNetwork::DecoderLayer *decoder_layer = !is_encoder_
        ? device_net->decoder_layers[layer_idx] : nullptr;
    const StdDeviceNetwork::FeedForwardLayer *ffn = is_encoder_
        ? &encoder_layer->ffn : &decoder_layer->ffn;
    const StdDeviceNetwork::AttentionLayer *self_attn = is_encoder_
        ? &encoder_layer->self_attn : &decoder_layer->self_attn;

    const auto &hparams = model_ptr_->spec.hyper_params;
    //const auto &model_spec = model_ptr_->spec;
    int perf_base = (layer_idx + 1) * 10000;

    if (global_data_ != nullptr)
    {
        int phase_id = layer_base_phase + (int)PhaseId::LAYER_START;
        global_data_->MoveToPhase(phase_id, nullptr, device_id_);
    }
    ClearLayerLocalMemory();

    TaskMonitor tm;
    //int input_rows = layer_input->Rows();
    //int input_cols = layer_input->Columns();
    //bool is_gemv_data_type = true;
    //int w1_cy = ffn->w1.tensor->ne[1];
    //bool use_gemv = input_rows == 1 && input_cols % 32 == 0 && w1_cy % 32 == 0
    //    && input_cols >= 1024 && w1_cy >= 1024
    //    && is_b_column_major && is_gemv_data_type
    //    && config_ex_->gemv_alg == VectorMatrixMulAlg::Alg3;
    /*if (!use_gemv)
    {
        if (is_encoder_)
        {
            DequantizeLayer(encoder_dequant_layer_, *encoder_layer, layer_idx);
            encoder_layer = &encoder_dequant_layer_;
            ffn = &encoder_dequant_layer_.ffn;
            self_attn = &encoder_dequant_layer_.self_attn;
        }
        else
        {
            DequantizeLayer(std_dequant_layer_, *decoder_layer, layer_idx);
            decoder_layer = &std_dequant_layer_;
            ffn = &std_dequant_layer_.ffn;
            self_attn = &std_dequant_layer_.self_attn;
        }
    }*/
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 100, tm);
    }

    /// self-attention
    tm.Start();
    AttentionOutput self_att_out = ProcessGpuLayer_Attention(layer_idx,
        *self_attn, layer_input, nullptr, heap_idx, is_encoder_);
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 300, tm);
    }
    Macro_RetIf(nullptr, self_att_out.output == nullptr);

    if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
    {
        PrintTensor(layer_input, 8, 8, 8, "layer_input:\n", layer_idx);
        PrintTensor(self_att_out.output, 8, 30, 8, "self_att_out (10500):\n", layer_idx);
    }

    if (!model_spec.is_parallel_attn && !model_spec.mlp_attn_share_input)
    {
        //DeviceTensor *ff_input = CreateLocalTensor(*layer_input, true, heap_idx);
        TensorOpr::Add(*self_att_out.output, *layer_input, *self_att_out.output);
        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
        {
            const char *title = "self_attn_out_with_residual_added (10501):\n";
            PrintTensor(self_att_out.output, 8, 30, 8, title, layer_idx);
        }
    }

    DeviceTensor *residual = self_att_out.output;
    if (self_attn->post_norm.tensor != nullptr)
    {
        DeviceTensor *post_norm = CreateLocalTensor(*self_att_out.output, true, heap_idx);
        TensorOpr::LayerNormalization(*post_norm, *self_att_out.output, model_spec.norm_alg,
            self_attn->post_norm.tensor, self_attn->post_norm_b.tensor);
        self_att_out.output = post_norm;
        if (model_spec.is_attn_post_as_residual) {
            residual = post_norm;
        }

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
        {
            const char *title = "self_attn_out_after_post_norm (10502):\n";
            PrintTensor(self_att_out.output, 8, 8, 8, title, layer_idx);
        }
    }

    /// cross-attention
    DeviceTensor *att_out = self_att_out.output;
    if (!is_encoder_ && input_kv.tensor != nullptr)
    {
        tm.Start();
        //calculate the cross-attention
        AttentionOutput cross_att_out = ProcessGpuLayer_Attention(layer_idx,
            decoder_layer->cross_attn, self_att_out.output, &input_kv,
            heap_idx, is_encoder_);
        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
            UpdatePerfStat(perf_base + 500, tm);
        }
        Macro_RetIf(nullptr, cross_att_out.output == nullptr);

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_) {
            PrintTensor(cross_att_out.output, 8, 8, 8, "cross_att_out (10690):\n", layer_idx);
        }

        //DeviceTensor *ff_input = CreateLocalTensor(*layer_input, true, heap_idx);
        TensorOpr::Add(*cross_att_out.output, *self_att_out.output, *cross_att_out.output);
        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_) {
            PrintTensor(cross_att_out.output, 8, 8, 8, "cross_att_out (10691):\n", layer_idx);
        }

        att_out = cross_att_out.output;
        residual = cross_att_out.output;
    }

    /// feed forward
    tm.Start();
    DeviceTensor *ff_in = model_spec.is_parallel_attn ? self_att_out.pre_norm
        : (model_spec.mlp_attn_share_input ? (DeviceTensor*)layer_input : att_out);
    bool is_moe = decoder_layer != nullptr && decoder_layer->moe.gate.tensor != nullptr;
    DeviceTensor *ff_out = nullptr;
    if (!is_encoder_ && is_moe)
    {
        const auto &moe_layer = decoder_layer->moe;
        ff_out = ProcessGpuLayer_Moe(layer_idx, moe_layer, ff_in, heap_idx, is_encoder_);
    }
    else
    {
        ff_out = ProcessGpuLayer_FeedForward(layer_idx, *ffn, ff_in, heap_idx, is_encoder_);
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 700, tm);
    }
    Macro_RetIf(nullptr, ff_out == nullptr);

    if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_) {
        PrintTensor(ff_out, 8, 8, 8, "ff_out (10700):\n", layer_idx);
    }

    tm.Start();
    DeviceTensor *layer_out = CreateLocalTensor(*layer_input, false, heap_idx);
    Macro_RetIf(nullptr, layer_out == nullptr);

    TensorOpr::Add(*layer_out, *ff_out, *residual);
    if (config_->debug.is_study_mode && (layer_idx == layer_idx_for_study_ || layer_idx >= 29)) {
        PrintTensor(layer_out, 8, 8, 8, "layer_out (10750):\n", layer_idx);
    }

    if (model_spec.is_parallel_attn || model_spec.mlp_attn_share_input)
    {
        TensorOpr::Add(*layer_out, *layer_out, *layer_input);
        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_) {
            PrintTensor(layer_out, 8, 8, 8, "layer_out (10751):\n", layer_idx);
        }
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 800, tm);
    }

    int layer_num = is_encoder_ ? hparams.encoder_layers : hparams.decoder_layers;
    if (ffn->post_norm.tensor != nullptr)
    {
        if (config_->debug.is_study_mode && (layer_idx % 1 == 0 || layer_idx + 1 >= layer_num))
        {
            PrintTensor(layer_out, 8, 8, 8, "layer_out (before_post_norm):\n", layer_idx);
        }

        DeviceTensor *post_norm = CreateLocalTensor(*layer_out, false, heap_idx);
        TensorOpr::LayerNormalization(*post_norm, *layer_out, model_spec.norm_alg,
            ffn->post_norm.tensor, ffn->post_norm_b.tensor);
        layer_out = post_norm;
    }

    if (config_->debug.is_study_mode && (layer_idx % 1 == 0 || layer_idx + 1 >= layer_num))
    {
        PrintTensor(layer_out, 8, 8, 8, "layer_out (10800):\n", layer_idx);
    }

    if (!is_by_layer_ && global_data_ != nullptr)
    {
        while (!global_data_->IsInPhase(layer_base_phase + (int)PhaseId::LAYER_END)) {
            Thread::SleepMicro(1);
        }
    }

    //LogKeyInfo("Worker %d: End of layer %d.", id_, layer_idx);
    return layer_out;
}

GpuInferenceWorker::AttentionOutput GpuInferenceWorker::ProcessGpuLayer_Attention(int layer_idx,
    const StdDeviceNetwork::AttentionLayer &layer, const DeviceTensor *input_q,
    const InputKV *input_kv, int heap_idx, bool is_encoder)
{
    AttentionOutput attn_out;
    bool is_succ = true;
    const auto &hparams = model_ptr_->spec.hyper_params;
    const auto &model_spec = model_ptr_->spec;
    int hparams_heads = is_encoder ? hparams.encoder_heads : hparams.decoder_heads;
    int kv_groups = is_encoder ? hparams.encoder_kv_heads : hparams.decoder_kv_heads;
    int heads_per_kv_group = hparams_heads / kv_groups;
    int token_dim = hparams.embd_dims;
    int token_sub_dim = is_by_layer_ ? token_dim : (token_dim / worker_num_);
    int head_num = is_by_layer_ ? hparams_heads : (hparams_heads / worker_num_);
    int kv_head_num = is_by_layer_ ? kv_groups : (kv_groups / worker_num_);
    int kv_sub_dim = token_sub_dim / heads_per_kv_group;
    int head_dim = token_dim / hparams_heads;
    int all_q_token_num = input_q->ne[1];
    int query_num = (int)query_list_.size();
    auto data_type = input_q->data_type;
    int layer_base_phase = (layer_idx + 1) * 100;
    bool is_cross_attn = input_kv != nullptr;
    int perf_base = (layer_idx + 1) * 10000 + (is_cross_attn ? 500 : 300);

    AttentionParams params;
    params.is_cross_attn = is_cross_attn;
    params.layer_idx = layer_idx;
    params.head_num = head_num;
    params.kv_head_num = kv_head_num;
    params.head_dim = head_dim;
    params.heads_per_kv_group = heads_per_kv_group;
    params.heap_idx = heap_idx;

    PosEmbeddingParams pos_embd_params;
    pos_embd_params.heads = hparams_heads; //!!! not head_num
    pos_embd_params.dims = head_dim;
    pos_embd_params.order_type = model_ptr_->spec.qk_column_order;
    pos_embd_params.alg = model_spec.pos_embedding_alg;
    pos_embd_params.rope_theta = model_spec.rope_theta;
    pos_embd_params.partial_rotary_factor = model_spec.partial_rotary_factor;

    bool is_alibi = model_spec.pos_embedding_alg == PositionEmbeddingAlg::ALIBI;
    bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;

    //norm
    TaskMonitor tm;
    const DeviceTensor *input_q_norm = input_q;
    if (layer.pre_norm.tensor != nullptr)
    {
        DeviceTensor *pre_norm = model_spec.use_self_attn_pre_norm
            ? layer.pre_norm.tensor : nullptr;
        DeviceTensor *pre_norm_b = model_spec.use_self_attn_pre_norm
            ? layer.pre_norm_b.tensor : nullptr;

        DeviceTensor *q_norm = CreateLocalTensor(*input_q, true, heap_idx);
        bool ret = TensorOpr::LayerNormalization(*q_norm, *input_q, model_spec.norm_alg,
            pre_norm, pre_norm_b);
        input_q_norm = q_norm;
        attn_out.pre_norm = q_norm;
        Macro_RetIf(attn_out, !ret);

        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
            UpdatePerfStat(perf_base + 10, tm);
        }
    }

    if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
    {
        //PrintTensor(layer.pre_norm.tensor, 8, 8, 8, "attn.pre_norm:\n", layer_idx);
        //PrintTensor(layer.pre_norm_b.tensor, 8, 8, 8, "attn.pre_norm_b:\n", layer_idx);
        PrintTensor(input_q_norm, 8, 8, 8, "attn_q_norm2 (10202):\n", layer_idx);
    }

    //const DeviceTensor *input_kv_tensor = input_kv == nullptr ? q_norm : input_kv->tensor;
    //int kv_token_num = input_kv_tensor->ne[1];

    if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
    {
        if (layer.wq.tensor != nullptr && !is_encoder)
        {
            //PrintTensor(layer.wq.tensor, 8, 8, 8, "wq.weight (10011):\n", layer_idx);
            //PrintTensor(layer.wk.tensor, 8, 8, 8, "wk.weight (10012):\n", layer_idx);
            //PrintTensor(layer.wv.tensor, 8, 8, 8, "wv.weight (10013):\n", layer_idx);
        }
        else if (layer.qkv.tensor != nullptr && !is_encoder)
        {
            //PrintTensor(layer.qkv.tensor, 8, 8, 8, "qkv.weight (10010):\n", layer_idx);
        }
    }

    CurQKV cur_qkv;
    is_succ = Attention_CalculateCurQKV(cur_qkv, layer_idx, layer,
        input_q_norm, input_kv, heap_idx, is_encoder);
    Macro_RetIf(attn_out, !is_succ);

    if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
    {
        //old: (8, 1, 48)
        PrintTensor(cur_qkv.q, 8, 3, 30, "cur_qkv.q (10203):\n", layer_idx);
        if (cur_qkv.k != nullptr) {
            PrintTensor(cur_qkv.k, 8, 30, 2, "cur_qkv.k (10213):\n", layer_idx);
        }
        if (cur_qkv.v != nullptr) {
            PrintTensor(cur_qkv.v, 8, 30, 2, "cur_qkv.v (10222):\n", layer_idx);
        }
        //exit(0);
    }

    tm.Start();
    for (int q_idx = 0; q_idx < (int)query_list_.size(); q_idx++)
    {
        const auto &query = query_list_[q_idx];
        QueryProcData *query_proc_data = query_proc_data_list_[query.proc_id];
        if (query_proc_data == nullptr) {
            continue;
        }

        LayerKVCache *layer_kv_cache = is_cross_attn
            ? query_proc_data->cross_attn_kv_cache.Layer(layer_idx)
            : query_proc_data->kv_cache.Layer(layer_idx);

        if (layer_kv_cache != nullptr && is_cross_attn && query.prefix_len == 0
            && model_spec.has_cross_attn_kv_cache)
        {
            const auto &encoder_q = input_kv->query_list[q_idx];
            int q_token_num = encoder_q.second;
            int q_start_row = encoder_q.first;
            layer_kv_cache->SetKRows(*cur_qkv.k, q_start_row, 0, q_token_num);
            layer_kv_cache->SetVRows(*cur_qkv.v, q_start_row, 0, q_token_num);
        }

        if (layer_kv_cache != nullptr && !is_cross_attn)
        {
            int q_token_num = query.token_num;
            int q_start_row = query.start_row;
            layer_kv_cache->SetKRows(*cur_qkv.k, q_start_row, query.prefix_len, q_token_num);
            layer_kv_cache->SetVRows(*cur_qkv.v, q_start_row, query.prefix_len, q_token_num);
        }
    }

    //TensorOpr::TransposeYZ(*cur_q);
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        //UpdatePerfStat(perf_base + 50, tm);
    }

    tm.Start();
    //kq_scale: reducing the chance of overflow in calculating q*k
    float kq_scale = is_alibi ? 1.0f : model_ptr_->spec.kq_scale;
    bool is_sub_level = true;
    DeviceTensor *kqv_merged = CreateLocalTensor(ElementType::F16,
        token_sub_dim, all_q_token_num, 0, true, heap_idx);
    vector<SourceTensor> kqv_tensors;
    for (int q_idx = 0; q_idx < (int)query_list_.size(); q_idx++)
    {
        const auto &query = query_list_[q_idx];
        QueryProcData *query_proc_data = query_proc_data_list_[query.proc_id];
        if (query_proc_data == nullptr) {
            continue;
        }

        LayerKVCache *layer_kv_cache = is_cross_attn
            ? query_proc_data->cross_attn_kv_cache.Layer(layer_idx)
            : query_proc_data->kv_cache.Layer(layer_idx);
        if (layer_kv_cache == nullptr) {
            continue;
        }

        int q_prefix_len = query.prefix_len;
        int q_token_num = query.token_num;

        TaskMonitor tm_sub;
        DeviceTensor *kq = CalculateProductKQ(cur_qkv, query, *layer_kv_cache,
            input_kv, params, q_idx, kq_scale);
        Macro_RetIf(attn_out, kq == nullptr);

        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_
            && query_num >= 1)
        {
            UpdatePerfStat(perf_base + 61, tm_sub);
        }

        //if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_) {
        //    PrintTensor(kq, 8, 8, 8, "kq (10251):\n", layer_idx);
        //}

        //tm_sub.Start();
        //TensorOpr::Scale(*kq, scale);

        //if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_
        //    && query_num >= 1)
        //{
        //    UpdatePerfStat(perf_base + 62, tm_sub);
        //}
        if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
        {
            PrintTensor(kq, 8, 4, 4, "kq (10252):\n", layer_idx);
        }

        if (is_alibi)
        {
            tm_sub.Start();
            pos_embd_params.alg = PositionEmbeddingAlg::ALIBI;
            pos_embd_params.context_len = query.prefix_len;
            int start_z = 0;
            int z_num = -1;
            int base_z = id_ * head_num * head_dim;
            TensorOpr::PositionEmbedding(*kq, pos_embd_params, start_z, z_num, base_z);

            if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_
                && query_num >= 1)
            {
                UpdatePerfStat(perf_base + 63, tm_sub);
            }
            if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
            {
                PrintTensor(kq, 8, 2, 2, "kq (10253):\n", layer_idx);
            }
        }

        tm_sub.Start();
        float neg_infinity = -std::numeric_limits<float>::infinity();
        if (!is_encoder_ && !is_cross_attn)
        {
            //TensorOpr::DiagMask(*kq, q_prefix_len, neg_infinity);
            //if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_) {
            //    PrintTensor(kq, 8, 2, 2, "kq_mask (10254):\n", layer_idx);
            //}
            //TensorOpr::SoftMax(*kq, -1, neg_infinity, kq_scale, &soft_max_aux_tensor_);
            TensorOpr::SoftMax(*kq, q_prefix_len, neg_infinity, kq_scale); //diag-mask and soft-max
        }
        else
        {
            TensorOpr::SoftMax(*kq, -1, neg_infinity, kq_scale, &soft_max_aux_tensor_);
        }

        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_
            && query_num >= 1)
        {
            UpdatePerfStat(perf_base + 65, tm_sub);
        }

        if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
        {
            PrintTensor(kq, 8, 2, 2, "kq_soft_max (10255):\n", layer_idx);
        }

        tm_sub.Start();
        DeviceTensor *v_trans = nullptr;
        if (is_cross_attn)
        {
            DeviceTensor sub_v(false);
            sub_v.data_type = cur_qkv.q->data_type;
            int encoder_token_num = input_kv->query_list[q_idx].second;
            if (model_spec.has_cross_attn_kv_cache)
            {
                sub_v.data = v_cache_item_.data;
                layer_kv_cache->GetVRows(sub_v, 0, encoder_token_num);
            }
            else
            {
                sub_v.SetStructure(head_dim * head_num, encoder_token_num);
                sub_v.data = cur_qkv.v->RowData(input_kv->query_list[q_idx].first);
            }

            v_trans = CreateLocalTensor(ElementType::F16, encoder_token_num,
                head_dim * head_num, 0, true, heap_idx);
            TensorOpr::Transpose(*v_trans, sub_v);
            is_succ = TensorOpr::Reshape(*v_trans, 3, encoder_token_num, head_dim, head_num);
            Macro_RetxIf(attn_out, !is_succ, LogError("Reshape failed"));
        }
        else
        {
            DeviceTensor v_with_ctx(false);
            v_with_ctx.data_type = ElementType::F16;
            v_with_ctx.data = v_cache_item_.data;
            layer_kv_cache->GetVRows(v_with_ctx, 0, q_prefix_len + q_token_num);

            v_trans = CreateLocalTensor(ElementType::F16, q_prefix_len + q_token_num,
                kv_sub_dim, 0, true, heap_idx);
            TensorOpr::Transpose(*v_trans, v_with_ctx);
            is_succ = TensorOpr::Reshape(*v_trans, 3, q_prefix_len + q_token_num,
                head_dim, kv_head_num);
            Macro_RetxIf(attn_out, !is_succ, LogError("Reshape failed"));
        }

        if (heads_per_kv_group > 1)
        {
            DeviceTensor *v_trans_ex = CreateLocalTensor(ElementType::F16, q_prefix_len + q_token_num,
                head_dim, head_num, true, heap_idx);
            TensorOpr::RepeatKV(*v_trans_ex, *v_trans, heads_per_kv_group);
            v_trans = v_trans_ex;
        }

        if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
        {
            PrintTensor(v_trans, 8, 2, 2, "v_trans (10260):\n", layer_idx);
        }

        //TensorOpr::TransposeYZ(v_trans);
        //PrintTensor(v_trans, 8, 8, 8, "v_trans (10261):\n");

        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_
            && query_num >= 1)
        {
            UpdatePerfStat(perf_base + 66, tm_sub);
        }

        tm_sub.Start();
        DeviceTensor *kqv = CreateLocalTensor(ElementType::F16,
            head_dim, q_token_num, head_num, true, heap_idx);
        is_succ = TensorMul::Gemm(*kqv, *kq, *v_trans, 1.0f, 0, true,
            is_sub_level, MatrixMulAlg::Alg2);
        Macro_RetIf(attn_out, !is_succ);

        if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
        {
            PrintTensor(kqv, 8, 30, 2, "kqv (10262):\n");
        }

        TensorOpr::TransposeYZ(*kqv);
        if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_) {
            PrintTensor(kqv, 8, 2, 30, "kqv_trans (10263):\n");
        }

        //if (is_by_layer_)
        {
            //device-to-device-memcpy is not applicable here, because y and z are transposed in kqv
            TensorOpr::Assign(*kqv_merged, *kqv, 0, query.start_row);
        }
        /*else
        {
            DeviceTensor *kqv_trans = CreateLocalTensor(ElementType::F16,
                head_dim, head_num, q_token_num, true, heap_idx);
            TensorOpr::Assign(*kqv_trans, *kqv);
            SourceTensor source_tensor;
            source_tensor.tensor = kqv_trans;
            source_tensor.start_row = query.start_row;
            kqv_tensors.push_back(source_tensor);
        }*/
        if (layer_idx == layer_idx_for_study_) {
            //PrintTensor(kqv_merged, 8, 8, 8, "kqv_merged (10263):\n");
        }
        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_
            && query_num >= 1)
        {
            UpdatePerfStat(perf_base + 67, tm_sub);
        }
    } //end of query_list

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 60, tm);
    }

    tm.Start();
    DeviceTensor *kqv_merged_quant = kqv_merged;
    bool use_full_quant_gemv = GetUseFullQuantGemv(*kqv_merged, *layer.wo.tensor);
    if (use_full_quant_gemv)
    {
        kqv_merged_quant = CreateLocalTensor(ElementType::Q8_B32T2,
            kqv_merged->ne[0], kqv_merged->ne[1], 0, true, 0);
        TensorOpr::Quantize(*kqv_merged_quant, *kqv_merged);
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 70, tm);
    }

    //if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
    //{
    //    PrintTensor(kqv_merged, 8, 4, 4, "kqv (10267):\n");
    //}

    tm.Start();
    int new_cx = is_b_column_major ? layer.wo.tensor->ne[1] : layer.wo.tensor->ne[0];
    int new_cy = input_q_norm->ne[1];
    DeviceTensor *bias = is_by_layer_ ? layer.wo_b.tensor : nullptr;
    DeviceTensor *out = CreateLocalTensor(data_type, new_cx, new_cy, 0, true, heap_idx);
    MatrixMultiplicationEx(*out, *kqv_merged, *kqv_merged_quant, layer.wo,
        is_b_column_major, bias);
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_)
    {
        PrintTensor(layer.wo.tensor, 8, 8, 8, "wo:\n");
        PrintTensor(out, 8, 30, 8, "self_attn_out (10268):\n");
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_
        && (is_by_layer_ || id_ == 0))
    {
        UpdatePerfStat(perf_base + 80, tm);
    }

    tm.Start();
    if (!is_by_layer_)
    {
        bool is_sum = true;
        out = DistributeAndMergeTensors(out, is_sum,
            layer_base_phase + (int)PhaseId::SELF_ATTN,
            layer_base_phase + (int)PhaseId::FFN, heap_idx);
        if (layer_idx == layer_idx_for_study_) {
            //PrintTensor(out, 8, 8, 8, "self_attn_out.merged:\n");
        }

        if (layer.wo_b.tensor != nullptr) {
            TensorOpr::Add(*out, *out, *layer.wo_b.tensor);
        }
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 90, tm);
    }

    if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
    {
        PrintTensor(out, 8, 8, 8, "self_attention_out (10270):\n");
    }

    attn_out.output = out;
    return attn_out;
}

bool GpuInferenceWorker::Attention_CalculateCurQKV(CurQKV &cur_qkv, int layer_idx,
    const StdDeviceNetwork::AttentionLayer &layer, const DeviceTensor *input_q,
    const InputKV *input_kv, int heap_idx, bool is_encoder)
{
    bool ret = true;
    const auto &hparams = model_ptr_->spec.hyper_params;
    const auto &model_spec = model_ptr_->spec;
    int hparams_heads = is_encoder ? hparams.encoder_heads : hparams.decoder_heads;
    int kv_groups = is_encoder ? hparams.encoder_kv_heads : hparams.decoder_kv_heads;
    int kv_sub_groups = is_by_layer_ ? kv_groups : (kv_groups / worker_num_);
    int heads_per_kv_group = hparams_heads / kv_groups;
    int token_dim = hparams.embd_dims;
    int token_sub_dim = is_by_layer_ ? token_dim : (token_dim / worker_num_);
    int kv_sub_dim = token_sub_dim / heads_per_kv_group;
    int head_num = is_by_layer_ ? hparams_heads : (hparams_heads / worker_num_);
    int kv_head_num = head_num / heads_per_kv_group;
    int head_dim = token_dim / hparams_heads;
    int q_token_num = input_q->ne[1];
    int kv_token_num = input_kv == nullptr ? q_token_num : input_kv->tensor->ne[1];
    //int query_num = (int)query_list_.size();
    auto data_type = input_q->data_type;

    bool is_cross_attention = input_kv != nullptr;
    //int layer_base_phase = (layer_idx + 1) * 100;
    int perf_base = (layer_idx + 1) * 10000 + (is_cross_attention ? 500 : 300);
    const DeviceTensor *input_kv_tensor = input_kv != nullptr ? input_kv->tensor : input_q;

    TaskMonitor tm;
    const auto *wq = layer.qkv.tensor != nullptr ? layer.qkv.tensor : layer.wq.tensor;
    const DeviceTensor *input_q_quant = input_q;
    const DeviceTensor *input_kv_quant = input_kv_tensor;
    bool use_full_quant_gemv = GetUseFullQuantGemv(*input_q, *wq);
    bool kv_use_full_quant_gemv = input_kv != nullptr && GetUseFullQuantGemv(*input_kv->tensor, *wq);
    if (use_full_quant_gemv)
    {
        DeviceTensor *quant_tensor = CreateLocalTensor(ElementType::Q8_B32T2,
            input_q->ne[0], input_q->ne[1], 0, true, 0);
        TensorOpr::Quantize(*quant_tensor, *input_q);
        input_q_quant = quant_tensor;

        if (input_kv == nullptr)
        {
            input_kv_quant = input_q_quant;
        }
        else if (kv_use_full_quant_gemv)
        {
            quant_tensor = CreateLocalTensor(ElementType::Q8_B32T2,
                input_kv->tensor->ne[0], input_kv->tensor->ne[1], 0, true, 0);
            TensorOpr::Quantize(*quant_tensor, *input_kv->tensor);
            input_kv_quant = quant_tensor;
        }
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 20, tm);
    }

    int max_query_prefix_len = 0;
    for (const auto &query : query_list_)
    {
        if (max_query_prefix_len < query.prefix_len) {
            max_query_prefix_len = query.prefix_len;
        }
    }

    PosEmbeddingParams pos_embd_params;
    pos_embd_params.heads = head_num; //to do: check it
    pos_embd_params.dims = head_dim;
    pos_embd_params.order_type = model_ptr_->spec.qk_column_order;
    pos_embd_params.rope_theta = model_spec.rope_theta;
    pos_embd_params.partial_rotary_factor = model_spec.partial_rotary_factor;

    bool is_rope = model_spec.pos_embedding_alg == PositionEmbeddingAlg::ROPE;
    bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;

    tm.Start();
    if (layer.qkv.tensor != nullptr) //in this case, input_q == input_kv
    {
        int new_cx = is_b_column_major ? layer.qkv.tensor->ne[1] : layer.qkv.tensor->ne[0];
        int new_cy = input_q->ne[1];
        DeviceTensor *qkv_tensor = CreateLocalTensor(data_type,
            new_cx, new_cy, 0, true, heap_idx);

        MatrixMultiplicationEx(*qkv_tensor, *input_q, *input_q_quant, layer.qkv,
            is_b_column_major, layer.qkv_b.tensor);

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
        {
            PrintTensor(qkv_tensor, 8, 30, 8, "qkv_tensor (m1):\n");
        }

        cur_qkv.q = CreateLocalTensor(data_type, token_sub_dim, new_cy, 0, true, heap_idx);
        cur_qkv.k = CreateLocalTensor(data_type, kv_sub_dim, new_cy, 0, true, heap_idx);
        cur_qkv.v = CreateLocalTensor(data_type, kv_sub_dim, new_cy, 0, true, heap_idx);

        bool b1 = true, b2 = true, b3 = true, b4 = true;
        //LogKeyInfo("qkv_format: %d", model_spec.qkv_format);
        if (model_spec.qkv_format == 0) // h1.q, h1.k, h1.v, h2.q, h2.k, h2.v...
        {
            if (heads_per_kv_group > 1)
            {
                //int h = is_by_layer_ ? heads_per_kv_group : (heads_per_kv_group / worker_num_);
                int h = heads_per_kv_group;
                b1 = TensorOpr::Reshape(*qkv_tensor, 2, head_dim * (h + 2), q_token_num * kv_sub_groups);
                b2 = TensorOpr::Reshape(*cur_qkv.q, 2, head_dim * h, q_token_num * kv_sub_groups);
                b3 = TensorOpr::Reshape(*cur_qkv.k, 2, head_dim, kv_token_num * kv_sub_groups);
                b4 = TensorOpr::Reshape(*cur_qkv.v, 2, head_dim, kv_token_num * kv_sub_groups);
                if (!b1 || !b2 || !b3 || !b4)
                {
                    LogError("Reshape failed: %s, %s, %s, %s (h = %d, head_dim = %d)",
                        b1 ? "true" : "false", b2 ? "true" : "false",
                        b3 ? "true" : "false", b4 ? "true" : "false",
                        h, head_dim);
                    return false;
                }

                TensorOpr::AssignColumns(*cur_qkv.q, *qkv_tensor, 0);
                TensorOpr::AssignColumns(*cur_qkv.k, *qkv_tensor, head_dim * h);
                TensorOpr::AssignColumns(*cur_qkv.v, *qkv_tensor, head_dim * (h + 1));
            }
            else
            {
                b1 = TensorOpr::Reshape(*qkv_tensor, 2, head_dim * 3, q_token_num * head_num);
                b2 = TensorOpr::Reshape(*cur_qkv.q, 2, head_dim, q_token_num * head_num);
                b3 = TensorOpr::Reshape(*cur_qkv.k, 2, head_dim, kv_token_num * head_num);
                b4 = TensorOpr::Reshape(*cur_qkv.v, 2, head_dim, kv_token_num * head_num);
                Macro_RetxIf(false, !b1 || !b2 || !b3 || !b4, LogError("Reshape failed"));

                TensorOpr::AssignColumns(*cur_qkv.q, *qkv_tensor, 0 * head_dim);
                TensorOpr::AssignColumns(*cur_qkv.k, *qkv_tensor, 1 * head_dim);
                TensorOpr::AssignColumns(*cur_qkv.v, *qkv_tensor, 2 * head_dim);
            }
            //TensorOpr::Reshape(*cur_qkv.q, 2, token_sub_dim, token_num);
            //TensorOpr::Reshape(*cur_qkv.k, 2, kv_sub_dim, token_num);
            //TensorOpr::Reshape(*cur_qkv.v, 2, kv_sub_dim, token_num);
        }
        else //h1.q, h2.q...; h1.k, h2.k...; h1.v, h2.v...
        {
            TensorOpr::AssignColumns(*cur_qkv.q, *qkv_tensor, 0 * token_sub_dim);
            TensorOpr::AssignColumns(*cur_qkv.k, *qkv_tensor, token_sub_dim + 0 * kv_sub_dim);
            TensorOpr::AssignColumns(*cur_qkv.v, *qkv_tensor, token_sub_dim + 1 * kv_sub_dim);
        }
    }
    else
    {
        int new_cx = layer.wq.tensor->ne[is_b_column_major ? 1 : 0];
        cur_qkv.q = CreateLocalTensor(data_type, new_cx, q_token_num, 0, true, heap_idx);
        MatrixMultiplicationEx(*cur_qkv.q, *input_q, *input_q_quant, layer.wq,
            is_b_column_major, layer.wq_b.tensor);

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_) {
            PrintTensor(cur_qkv.q, 8, 3, 6, "cur_qkv.q (before_rope):\n");
        }

        if (!is_cross_attention || max_query_prefix_len == 0 || !model_spec.has_cross_attn_kv_cache)
        {
            new_cx = layer.wk.tensor->ne[is_b_column_major ? 1 : 0];
            cur_qkv.k = CreateLocalTensor(data_type, new_cx, kv_token_num, 0, true, heap_idx);
            MatrixMultiplicationEx(*cur_qkv.k, *input_kv_tensor, *input_kv_quant, layer.wk,
                is_b_column_major, layer.wk_b.tensor);

            cur_qkv.v = CreateLocalTensor(data_type, new_cx, kv_token_num, 0, true, heap_idx);
            MatrixMultiplicationEx(*cur_qkv.v, *input_kv_tensor, *input_kv_quant, layer.wv,
                is_b_column_major, layer.wv_b.tensor);
        }
    }
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 30, tm);
    }

    tm.Start();
    ret = TensorOpr::Reshape(*cur_qkv.q, 3, head_dim, head_num, q_token_num);
    Macro_RetxIf(false, !ret, LogError("Reshape failed"));
    if (is_rope)
    {
        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_) {
            PrintTensor(cur_qkv.q, 8, 3, 30, "cur_qkv.q (before_rope.2):\n");
        }

        for (const auto &query : query_list_)
        {
            pos_embd_params.context_len = query.prefix_len;
            int start_z = query.start_row;
            int z_num = query.token_num;
            TensorOpr::PositionEmbedding(*cur_qkv.q, pos_embd_params, start_z, z_num);
        }
    }

    //if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
    //    UpdatePerfStat(perf_base + 40, tm);
    //}
    //tm.Start();

    //k
    if (!is_cross_attention || max_query_prefix_len == 0 || !model_spec.has_cross_attn_kv_cache)
    {
        ret = TensorOpr::Reshape(*cur_qkv.k, 3, head_dim, kv_head_num, kv_token_num);
        Macro_RetxIf(false, !ret, LogError("Reshape failed"));

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_) {
            PrintTensor(cur_qkv.k, 8, 3, 30, "cur_qkv.k (before_rope.2):\n");
        }

        if (is_rope)
        {
            pos_embd_params.heads = kv_head_num; //to do: check it
            for (const auto &query : query_list_)
            {
                pos_embd_params.context_len = query.prefix_len;
                int start_z = query.start_row;
                int z_num = query.token_num;
                TensorOpr::PositionEmbedding(*cur_qkv.k, pos_embd_params, start_z, z_num);
            }
        }
        TensorOpr::Reshape(*cur_qkv.k, 2, kv_sub_dim, kv_token_num);
    }

    //v
    if (!is_cross_attention || max_query_prefix_len == 0 || !model_spec.has_cross_attn_kv_cache)
    {
        ret = TensorOpr::Reshape(*cur_qkv.v, 2, kv_sub_dim, kv_token_num);
        Macro_RetxIf(false, !ret, LogError("Reshape failed"));
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 50, tm);
    }

    return ret;
}

DeviceTensor* GpuInferenceWorker::CalculateProductKQ(const CurQKV &cur_qkv,
    const QueryProcInput &query, LayerKVCache &layer_kv_cache, const InputKV *input_kv,
    const AttentionParams &params, int q_idx, float kq_scale)
{
    const auto &model_spec = model_ptr_->spec;
    int layer_idx = params.layer_idx;
    int head_num = params.head_num;
    int kv_head_num = params.kv_head_num;
    int head_dim = params.head_dim;
    int heads_per_kv_group = params.heads_per_kv_group;
    int heap_idx = params.heap_idx;
    int q_prefix_len = query.prefix_len;
    int q_token_num = query.token_num;
    bool is_sub_level = true;

    //cur_q structure: (head_dim, head_num, token_num)
    DeviceTensor sub_q;
    sub_q.data_type = cur_qkv.q->data_type;
    sub_q.SetStructure(cur_qkv.q->ne[0], cur_qkv.q->ne[1], q_token_num);
    sub_q.data = cur_qkv.q->RowData(0, query.start_row);
    //sub_q.data = cur_q->data;
    sub_q.SetAutoFree(false);

    TensorOpr::TransposeYZ(sub_q);

    DeviceTensor *kq = nullptr;
    bool is_succ = true;
    float scale = 1.0f / sqrtf(float(head_dim)) / kq_scale;
    if (params.is_cross_attn)
    {
        DeviceTensor sub_k(false);
        sub_k.data_type = cur_qkv.q->data_type;
        int encoder_token_num = input_kv->query_list[q_idx].second;
        if (model_spec.has_cross_attn_kv_cache)
        {
            sub_k.data = k_cache_item_.data;
            layer_kv_cache.GetKRows(sub_k, 0, encoder_token_num);
            is_succ = TensorOpr::Reshape(sub_k, 3, head_dim, head_num, encoder_token_num);
            Macro_RetxIf(nullptr, !is_succ, LogError("Reshape failed"));
        }
        else
        {
            sub_k.SetStructure(head_dim, head_num, encoder_token_num);
            sub_k.data = cur_qkv.k->RowData(input_kv->query_list[q_idx].first);
        }

        TensorOpr::TransposeYZ(sub_k);

        kq = CreateLocalTensor(ElementType::F16, encoder_token_num,
            q_token_num, head_num, true, heap_idx);
        is_succ = TensorMul::Gemm(*kq, sub_q, sub_k, scale, 0, true,
            is_sub_level, MatrixMulAlg::Alg2);
    }
    else
    {
        DeviceTensor k_with_ctx(false);
        k_with_ctx.data_type = cur_qkv.q->data_type;
        k_with_ctx.data = k_cache_item_.data;
        layer_kv_cache.GetKRows(k_with_ctx, 0, q_prefix_len + q_token_num);
        is_succ = TensorOpr::Reshape(k_with_ctx, 3, head_dim, kv_head_num,
            q_prefix_len + q_token_num);
        Macro_RetxIf(nullptr, !is_succ, LogError("Reshape failed"));

        TensorOpr::TransposeYZ(k_with_ctx);
        DeviceTensor *k_with_ctx_ptr = &k_with_ctx;
        if (heads_per_kv_group > 1)
        {
            k_with_ctx_ptr = CreateLocalTensor(ElementType::F16, head_dim,
                q_prefix_len + q_token_num, head_num, true, heap_idx);
            TensorOpr::RepeatKV(*k_with_ctx_ptr, k_with_ctx, heads_per_kv_group);
        }

        if (config_->debug.show_tensors && layer_idx == layer_idx_for_study_)
        {
            //PrintTensor(&sub_q, 8, 6, 3, "sub_q (10231):\n", layer_idx);
            //PrintTensor(k_with_ctx_ptr, 8, 3, 6, "k_with_ctx (10232):\n", layer_idx);
        }

        kq = CreateLocalTensor(ElementType::F16, q_prefix_len + q_token_num,
            q_token_num, head_num, true, heap_idx);
        is_succ = TensorMul::Gemm(*kq, sub_q, *k_with_ctx_ptr, scale, 0, true,
            is_sub_level, MatrixMulAlg::Alg2);
    }

    return is_succ ? kq : nullptr;
}

DeviceTensor* GpuInferenceWorker::ProcessGpuLayer_FeedForward(int layer_idx,
    const StdDeviceNetwork::FeedForwardLayer &layer, DeviceTensor *input_tensor,
    int heap_idx, bool is_encoder, bool enable_tensor_printing)
{
    (void)is_encoder;
    bool ret = true;
    //const auto &hparams = model_ptr_->hyper_params;
    const auto &model_spec = model_ptr_->spec;
    //int hparams_heads = is_encoder ? hparams.encoder_heads : hparams.decoder_heads;
    //int token_dim = hparams.embd_dims;
    //int head_num = hparams_heads;
    //int head_dim = token_dim / head_num;
    int token_num = input_tensor->ne[1];
    int layer_base_phase = (layer_idx + 1) * 100;
    int perf_base = (layer_idx + 1) * 10000;
    enable_tensor_printing = enable_tensor_printing && config_->debug.is_study_mode
        && config_->debug.show_tensors;

    bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;

    //norm
    TaskMonitor tm;
    DeviceTensor *norm2 = input_tensor;
    if (layer.pre_norm.tensor != nullptr)
    {
        norm2 = CreateLocalTensor(*input_tensor, true, heap_idx);
        ret = TensorOpr::LayerNormalization(*norm2, *input_tensor, model_spec.norm_alg,
            layer.pre_norm.tensor, layer.pre_norm_b.tensor);
        Macro_RetIf(nullptr, !ret);
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 710, tm);
    }

    //PrintTensor(layer.ffn_norm.tensor, 8, 8, 8, "ffn_norm:\n");
    //PrintTensor(layer.ffn_norm_b.tensor, 8, 8, 8, "ffn_norm_b:\n");
    if (enable_tensor_printing && layer_idx == layer_idx_for_study_) {
        PrintTensor(norm2, 8, 30, 8, "norm2 (10302):\n");
    }

    tm.Start();
    DeviceTensor *norm_quant = norm2;
    bool use_full_quant_gemv = GetUseFullQuantGemv(*norm2, *layer.w1.tensor);
    if (use_full_quant_gemv)
    {
        norm_quant = CreateLocalTensor(ElementType::Q8_B32T2,
            norm2->ne[0], norm2->ne[1], 0, true, 0);
        TensorOpr::Quantize(*norm_quant, *norm2);
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 720, tm);
    }

    tm.Start();
    int t1_cols = is_b_column_major ? layer.w1.tensor->ne[1] : layer.w1.tensor->ne[0];
    DeviceTensor *t1 = CreateLocalTensor(ElementType::F16,
        t1_cols, token_num, 0, true, heap_idx);
    ret = MatrixMultiplicationEx(*t1, *norm2, *norm_quant, layer.w1,
        is_b_column_major, layer.w1_b.tensor);

    if (enable_tensor_printing && layer_idx == layer_idx_for_study_) {
        PrintTensor(t1, 8, 8, 8, "t1 (10311):\n");
    }
    Macro_RetIf(nullptr, !ret);

    //PrintTensor(layer.w1.tensor, 8, 8, 8, "w1:\n");
    //PrintTensor(t1, 8, 8, 8, "t1 (10312):\n");
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 730, tm);
    }

    tm.Start();
    DeviceTensor *act = CreateActivationTarget(*t1, model_spec.activation_fn, true, heap_idx);
    TensorOpr::Activation(*act, *t1, model_spec.activation_fn);
    if (enable_tensor_printing && layer_idx == layer_idx_for_study_) {
        PrintTensor(act, 8, 8, 8, "act (10321):\n");
    }
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 740, tm);
    }

    DeviceTensor *t2 = nullptr;
    if (ret && layer.w3.tensor != nullptr)
    {
        tm.Start();
        int t2_cols = is_b_column_major ? layer.w3.tensor->ne[1] : layer.w3.tensor->ne[0];
        t2 = CreateLocalTensor(ElementType::F16, t2_cols, token_num, 0, true, heap_idx);
        ret = MatrixMultiplication(*t2, *norm_quant, *layer.w3.tensor, is_b_column_major);
        if (layer.w3.delta != nullptr && !layer.w3.delta->IsEmpty()) {
            TensorMul::GemmSparse(*t2, *norm2, *layer.w3.delta, 1.0f, 1.0f);
        }

        if (layer.w3_b.tensor != nullptr) {
            TensorOpr::Add(*t2, *t2, *layer.w3_b.tensor);
        }
        if (enable_tensor_printing && layer_idx == layer_idx_for_study_) {
            PrintTensor(t2, 8, 8, 8, "t2 (10311):\n");
        }
        Macro_RetIf(nullptr, !ret);

        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
            UpdatePerfStat(perf_base + 750, tm);
        }
    }

    DeviceTensor *t3 = act;
    if (t2 != nullptr)
    {
        tm.Start();
        t3 = CreateLocalTensor(*t1, true, heap_idx);
        TensorOpr::Mul(*t3, *act, *t2);
        if (enable_tensor_printing && layer_idx == layer_idx_for_study_) {
            PrintTensor(t3, 8, 8, 8, "t3 (10322):\n");
        }
        if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
            UpdatePerfStat(perf_base + 760, tm);
        }
    }

    tm.Start();
    DeviceTensor *t3_quant = t3;
    use_full_quant_gemv = GetUseFullQuantGemv(*t3, *layer.w2.tensor);
    if (use_full_quant_gemv)
    {
        t3_quant = CreateLocalTensor(ElementType::Q8_B32T2,
            t3->ne[0], t3->ne[1], 0, true, 0);
        TensorOpr::Quantize(*t3_quant, *t3);
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 770, tm);
    }

    //
    tm.Start();
    int embd_dim_idx = is_b_column_major ? 1 : 0;
    int cx_new = layer.w2.tensor->ne[embd_dim_idx];
    DeviceTensor *out = CreateLocalTensor(ElementType::F16,
        cx_new, token_num, 0, true, heap_idx);
    DeviceTensor *bias = is_by_layer_ ? layer.w2_b.tensor : nullptr;
    ret = MatrixMultiplicationEx(*out, *t3, *t3_quant, layer.w2, is_b_column_major, bias);
    if (enable_tensor_printing && layer_idx == layer_idx_for_study_)
    {
        PrintTensor(layer.w2.tensor, 8, 8, 8, "w2:\n");
        if (bias != nullptr) {
            PrintTensor(bias, 8, 8, 8, "w2_b:\n");
        }
        PrintTensor(out, 8, 8, 8, "ffn_out (10323):\n");
    }
    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 780, tm);
    }

    tm.Start();
    if (!is_by_layer_)
    {
        bool is_sum = true;
        out = DistributeAndMergeTensors(out, is_sum,
            layer_base_phase + (int)PhaseId::FFN,
            layer_base_phase + (int)PhaseId::LAYER_END, heap_idx);
        //out = DistributeAndMergeTensors(out, false, layer_base_phase + (int)PhaseId::W2,
        //    layer_base_phase + (int)PhaseId::LAYER_END, heap_idx);
        //PrintTensor(out, 8, 8, 8, "ffn_out.merged:\n");

        if (layer.w2_b.tensor != nullptr) {
            TensorOpr::Add(*out, *out, *layer.w2_b.tensor);
        }
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 790, tm);
    }

    Macro_RetIf(nullptr, !ret);
    return out;
}

struct ExpertRow
{
    int id = 0;
    int order = 0;
    inferflow_fp16 score = (inferflow_fp16)0.0f;

    ExpertRow(int p_id = 0, int p_order = 0, inferflow_fp16 p_score = (inferflow_fp16)0.0f)
    {
        this->id = p_id;
        this->order = p_order;
        this->score = p_score;
    }
};

struct ExpertState
{
    vector<ExpertRow> rows;
};

DeviceTensor* GpuInferenceWorker::ProcessGpuLayer_Moe(int layer_idx,
    const StdDeviceNetwork::FfnMoeLayer &layer, DeviceTensor *input_tensor,
    int heap_idx, bool is_encoder)
{
    const auto &model_spec = model_ptr_->spec;
    const auto &hparams = model_spec.hyper_params;
    const DeviceTensor &gate_tensor = *layer.gate.tensor;
    int token_num = input_tensor->ne[1];
    int expert_num = hparams.in_use_experts;
    int moe_top_k = hparams.moe_top_k;
    int perf_base = (layer_idx + 1) * 10000 + 600;
    bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;

    if (moe_top_k > RowItemForMoe::MAX_SIZE)
    {
        LogError("moe_top_k > %d is not supported so far.", RowItemForMoe::MAX_SIZE);
        return nullptr;
    }

    TaskMonitor tm;
    tm.Start();
    DeviceTensor *input_quant = input_tensor;
    bool use_full_quant_gemv = GetUseFullQuantGemv(*input_tensor, gate_tensor);
    if (use_full_quant_gemv)
    {
        input_quant = CreateLocalTensor(ElementType::Q8_B32T2,
            input_tensor->ne[0], input_tensor->ne[1], 0, true, 0);
        TensorOpr::Quantize(*input_quant, *input_tensor);
    }

    DeviceTensor *router_logits = CreateLocalTensor(ElementType::F16,
        gate_tensor.ne[1], token_num, 0, true, heap_idx);
    bool ret = MatrixMultiplicationEx(*router_logits, *input_tensor, *input_quant,
        layer.gate, is_b_column_major, layer.gate_b.tensor);

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 10, tm);
    }

    if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
    {
        PrintTensor(input_tensor, 8, 8, 8, "moe_input:\n");
        PrintTensor(router_logits, 64, 8, 8, "router_logits:\n");
    }
    Macro_RetIf(nullptr, !ret);

    tm.Start();
    float neg_infinity = -std::numeric_limits<float>::infinity();
    TensorOpr::SoftMax(*router_logits, -1, neg_infinity, 1.0f, &soft_max_aux_tensor_);

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 20, tm);
    }

    HostTensor host_router_logits;
    router_logits->CopyToHost(host_router_logits);
    //HostTensorOpr::SoftMax(host_router_logits);

    if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
    {
        PrintTensor(router_logits, 64, 8, 8, "router_logits_softmax:\n");
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 30, tm);
    }

    tm.Start();
    vector<RowItemForMoe> row_items;
    HostTensorOpr::BuildRowsForMoE(row_items, host_router_logits,
        moe_top_k, hparams.moe_norm_top_k_prob);

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 40, tm);
    }

    //LogKeyInfo("===== row_items");
    //for (const auto &row_item : row_items)
    //{
    //    stringstream ss;
    //    for (int idx = 0; idx < row_item.size; idx++)
    //    {
    //        const auto &exp = row_item.arr[idx];
    //        if (idx > 0) {
    //            ss << " | ";
    //        }
    //        ss << "(" << exp.id << ", " << exp.weight << ")";
    //    }
    //    LogKeyInfo("%s", ss.str().c_str());
    //}

    tm.Start();
    vector<ExpertState> expert_states(expert_num);
    for (int row_idx = 0; row_idx < token_num; row_idx++)
    {
        const auto &item = row_items[row_idx];
        for (int order = 0; order < item.size; order++)
        {
            int expert_id = item.arr[order].id;
            auto &state = expert_states[expert_id];
            ExpertRow er(row_idx, order, item.arr[order].weight);
            state.rows.push_back(er);
        }
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 50, tm);
    }

    DeviceTensor *out_tensor = nullptr;
    /*if (hparams.has_shared_expert)
    {
        out_tensor = ProcessGpuLayer_FeedForward(layer_idx, layer.shared_expert,
            input_tensor, heap_idx, is_encoder, false);
    }
    else*/
    {
        out_tensor = CreateLocalTensor(*input_tensor, false, heap_idx);
        out_tensor->AssignZero();
    }

    tm.Start();
    vector<int> idx_list;
    DeviceTensor *partial_input = CreateLocalTensor(*input_tensor, true, heap_idx);
    DeviceTensor *weight_tensor = CreateLocalTensor(ElementType::F16,
        1, token_num, 0, true, heap_idx);
    DeviceTensor *idx_tensor = CreateLocalTensor(ElementType::I32,
        token_num, 0, 0, true, heap_idx);
    for (int expert_idx = 0; expert_idx < expert_num; expert_idx++)
    {
        const auto &state = expert_states[expert_idx];
        const auto *expert_layer = layer.experts[expert_idx];
        if (state.rows.empty()) {
            continue;
        }

        idx_list.clear();
        int state_rows = (int)state.rows.size();
        for (int row_idx = 0; row_idx < state_rows; row_idx++)
        {
            const auto &row = state.rows[row_idx];
            aux_buffer_.Set(row_idx, row.score);
            idx_list.push_back(row.id);

            const void *src_data = input_tensor->RowData(row.id);
            void *target_data = partial_input->RowData(row_idx);
            int bytes = (int)TensorCommon::ByteCount(input_tensor->data_type, input_tensor->ne[0]);
            CudaUtil::DeviceToDeviceMemcpy(target_data, src_data, bytes);
        }

        partial_input->SetStructure(input_tensor->ne[0], state_rows, 0);

        int bytes = (int)TensorCommon::ByteCount(ElementType::F16, state_rows);
        CudaUtil::HostToDeviceMemcpy(weight_tensor->data, aux_buffer_.data(), bytes);
        weight_tensor->SetStructure(state_rows, 0, 0);

        bytes = (int)TensorCommon::ByteCount(ElementType::I32, state_rows);
        CudaUtil::HostToDeviceMemcpy(idx_tensor->data, idx_list.data(), bytes);
        idx_tensor->SetStructure(state_rows, 0, 0);

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
        {
            //stringstream ss;
            //for (int idx = 0; idx < (int)idx_list.size(); idx++)
            //{
            //    int row_idx = idx_list[idx];
            //    int order = state.rows[idx].order;
            //    float w = state.rows[idx].score;
            //    ss << row_idx << " (" << order << ", " << w << ") ";
            //}
            //LogKeyInfo("expert_idx: %d, state_rows: %d, idx and weight list: %s",
            //    expert_idx, state_rows, ss.str().c_str());
            PrintTensor(weight_tensor, 8, 8, 8, "===== weight_tensor:\n");
        }

        bool enable_tensor_printing = expert_idx == 2; //false
        DeviceTensor *expert_out_tensor = ProcessGpuLayer_FeedForward(layer_idx,
            *expert_layer, partial_input, heap_idx, is_encoder, enable_tensor_printing);

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
        {
            PrintTensor(partial_input, 8, 8, 8, "partial_input:\n");
            PrintTensor(expert_out_tensor, 8, 8, 8, "expert_out_tensor:\n");
        }

        TensorOpr::AddByRowIndex(*out_tensor, *expert_out_tensor, *idx_tensor, weight_tensor);

        if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
        {
            PrintTensor(out_tensor, 8, 8, 8, "moe_out_tensor.partial:\n");
        }
    } //for each expert

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 70, tm);
    }

    tm.Start();
    if (hparams.has_shared_expert)
    {
        const auto *shared_out_tensor = ProcessGpuLayer_FeedForward(layer_idx,
            layer.shared_expert, input_tensor, heap_idx, is_encoder);
        TensorOpr::Add(*out_tensor, *out_tensor, *shared_out_tensor);
    }

    if (config_->debug.enable_perf_stat && layer_idx == layer_idx_for_study_) {
        UpdatePerfStat(perf_base + 80, tm);
    }

    if (config_->debug.is_study_mode && layer_idx == layer_idx_for_study_)
    {
        //LogKeyInfo("Printing moe_out_tensor...");
        PrintTensor(out_tensor, 8, 8, 8, "moe_out_tensor:\n");
    }

    return out_tensor;
}

DeviceTensor* GpuInferenceWorker::DistributeAndMergeTensors(const DeviceTensor *tensor,
    bool is_sum, int cur_phase, int next_phase, int heap_idx, bool is_study_mode)
{
    if (global_data_ == nullptr) {
        LogError("Null global data");
        return nullptr;
    }

    const DeviceTensor *tensor_to_add = tensor;
    if (is_quant_tensor_exchange_ && id_ != 0)
    {
        DeviceTensor *quant_tensor = CreateLocalTensor(ElementType::Q8_B32T2, tensor->ne[0],
            tensor->ne[1], tensor->ne[2], true, heap_idx);
        TensorOpr::Quantize(*quant_tensor, *tensor);
        tensor_to_add = quant_tensor;
    }

    GpuInfGlobalData &global_data = *global_data_;
    bool ret = global_data.Add(id_, cur_phase, tensor_to_add, device_id_);
    Macro_RetxIf(nullptr, !ret, LogError("Failed to add information to the global data"));

    while (!global_data.IsPhaseDone(cur_phase)) {
        Thread::SleepMicro(1);
    }

    const DeviceTensor *merged_tensor = nullptr;
    if (id_ == 0)
    {
        vector<TensorWithDeviceId> tensor_list;
        global_data.GetOutputTensors(tensor_list, cur_phase);

        //LogKeyInfo("Merge %d tensor(s)", (int)tensor_list.size());
        merged_tensor = MergeTensors(tensor_list, is_sum,
            device_id_, heap_idx, is_study_mode);
        global_data.MoveToPhase(next_phase, merged_tensor, device_id_);
    }
    else
    {
        while (!global_data.IsInPhase(next_phase)) {
            Thread::SleepMicro(1);
        }
        merged_tensor = global_data.GetInputTensor(id_, next_phase);
    }

    //PrintTensor(merged_tensor, 8, 8, 8, "merged_tensor:\n");
    DeviceTensor *ret_tensor = CreateLocalTensor(*merged_tensor, true, heap_idx);
    int bytes = (int)ret_tensor->ByteCount();
    CudaUtil::DeviceToDeviceMemcpy(ret_tensor->data, merged_tensor->data, bytes);
    //PrintTensor(ret_tensor, 8, 8, 8, "ret_tensor:\n");
    return ret_tensor;
}

void GpuInferenceWorker::DistributeAndMergeTensors(DeviceTensor *merged_tensor,
    const vector<SourceTensor> &src_tensors, int cur_phase, int next_phase)
{
    GpuInfGlobalData &global_data = *global_data_;
    bool ret = global_data.Add(id_, cur_phase, src_tensors, device_id_);
    Macro_RetxVoidIf(!ret, LogError("Failed to add source tensors to the global data"));

    while (!global_data.IsPhaseDone(cur_phase)) {
        Thread::SleepMicro(1);
    }

    if (id_ == 0)
    {
        vector<SourceTensor> source_tensors;
        global_data.GetSourceTensors(source_tensors, cur_phase);

        for (const auto &source_tensor : source_tensors)
        {
            int cx = source_tensor.tensor->ne[0];
            int offset = source_tensor.start_row * cx * sizeof(half);
            int bytes = (int)source_tensor.tensor->ByteCount();
            if (offset + bytes > (int)merged_tensor->ByteCount()) {
                LogError("[Error: %d + %d > %d]", offset, bytes, (int)merged_tensor->ByteCount());
                exit(0);
            }
            CudaUtil::DeviceToDeviceMemcpy((uint8_t*)merged_tensor->data + offset,
                source_tensor.tensor->data, bytes);
        }
        global_data.MoveToPhase(next_phase, merged_tensor, device_id_);
    }
    else
    {
        while (!global_data.IsInPhase(next_phase)) {
            Thread::SleepMicro(1);
        }
        const auto *ret_tensor = global_data.GetInputTensor(id_, next_phase);
        int bytes = (int)ret_tensor->ByteCount();
        CudaUtil::DeviceToDeviceMemcpy(merged_tensor->data, ret_tensor->data, bytes);
    }
}

DeviceTensor* GpuInferenceWorker::MergeTensors(const vector<TensorWithDeviceId> &tensor_list,
    bool is_sum, int target_device, int heap_idx, bool is_study_mode)
{
    int cx0 = tensor_list[0].tensor->ne[0];
    int cy = tensor_list[0].tensor->Rows();
    int cx = is_sum ? cx0 : 0;
    if (!is_sum)
    {
        for (const auto &tdi : tensor_list) {
            cx += tdi.tensor->ne[0];
        }
    }

    DeviceTensor *merged_tensor = CreateLocalTensor(ElementType::F16,
        cx, cy, 1, false, heap_idx);
    DeviceTensor *quant_src_tensor = CreateLocalTensor(ElementType::Q8_B32T2,
        cx0, cy, 1, true, heap_idx);
    DeviceTensor *dequant_tensor = CreateLocalTensor(ElementType::F16,
        cx0, cy, 1, true, heap_idx);

    TaskMonitor tm;
    int offset = 0;
    if (is_sum)
    {
        for (int idx = 0; idx < (int)tensor_list.size(); idx++)
        {
            const auto &tdi = tensor_list[idx];
            int bytes = (int)TensorCommon::ByteCount(tdi.tensor->data_type, cx0 * cy);

            if (idx == 0)
            {
                DeviceCopy(merged_tensor->data, target_device,
                    tdi.tensor->data, tdi.device_id, bytes);
                //CudaUtil::DeviceSynchronize();
            }
            else
            {
                DeviceTensor *target_tensor = is_quant_tensor_exchange_
                    ? quant_src_tensor : dequant_tensor;
                DeviceCopy(target_tensor->data, target_device,
                    tdi.tensor->data, tdi.device_id, bytes);
                //CudaUtil::DeviceSynchronize();
                if (is_quant_tensor_exchange_) {
                    TensorOpr::Dequantize(*dequant_tensor, *quant_src_tensor);
                }
                TensorOpr::Add(*merged_tensor, *merged_tensor, *dequant_tensor);
            }
        }
    }
    else
    {
        int bytes = tensor_list[0].tensor->ne[0] * (int)sizeof(half);
        for (int row_idx = 0; row_idx < cy; row_idx++)
        {
            for (const auto &tdi : tensor_list)
            {
                const DeviceTensor *tensor = tdi.tensor;
                uint8_t *target_data = (uint8_t*)merged_tensor->data;
                int source_offset = row_idx * bytes;
                DeviceCopy(target_data + offset, target_device,
                    (uint8_t*)tensor->data + source_offset, tdi.device_id, bytes);
                offset += bytes;
        }
    }
    }

    if (is_study_mode) {
        tm.ShowElapsedTime(L"Merge");
    }

    return merged_tensor;
}

bool GpuInferenceWorker::DeviceCopy(void *dst, int dst_device,
    const void *src, int src_device, int bytes)
{
    (void)src_device; (void)dst_device;
    bool ret = true;
    /*int buffer_capacity = aux_buffer_.capacity() * sizeof(half);
    if (dst_device != src_device && bytes <= buffer_capacity)
    {
        int device_bak = CudaUtil::GetDevice();
        void *host_data = (void*)aux_buffer_.data();
        CudaUtil::SetDevice(src_device);
        ret = ret && CudaUtil::DeviceToHostMemcpy(host_data, src, bytes);
        CudaUtil::SetDevice(dst_device);
        ret = ret && CudaUtil::HostToDeviceMemcpy(dst, host_data, bytes);
        CudaUtil::SetDevice(device_bak);
        return ret;
    }*/

    ret = CudaUtil::DeviceToDeviceMemcpy(dst, src, bytes);
    return ret;
}

bool GpuInferenceWorker::MatrixMultiplicationEx(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &AQ, const DeviceTensorEx &B, bool is_b_column_major,
    const DeviceTensor *bias)
{ //AQ: A quant
    bool ret = true;
    bool has_delta = B.delta != nullptr && !B.delta->IsEmpty();

    if (has_delta)
    {
        ret = MatrixMultiplication(C, AQ, *B.tensor, is_b_column_major);
        TensorMul::GemmSparse(C, A, *B.delta, 1.0f, 1.0f);
        if (bias != nullptr) {
            TensorOpr::Add(C, C, *bias);
        }
    }
    else
    {
        ret = MatrixMultiplication(C, AQ, *B.tensor, is_b_column_major, nullptr);
        if (bias != nullptr) {
            TensorOpr::Add(C, C, *bias);
        }
    }

    return ret;
}

bool GpuInferenceWorker::MatrixMultiplication(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, bool is_b_column_major, const DeviceTensor *bias)
{
    bool ret = true;
    VectorMatrixMulAlg gemv_alg = config_ex_->gemv_alg;
    bool use_bias = bias != nullptr;
    bool use_gemv = A.Rows() == 1 && A.size % 32 == 0 && is_b_column_major
        && gemv_alg == VectorMatrixMulAlg::Alg3
        && B.Columns() <= 65535; //&& B.Rows() <= 65535;

    const DeviceTensor *b_ptr = &B;
    if (B.IsQuantized() && !use_gemv)
    {
        dequant_tensor_.SetStructure(B.ne[0], B.ne[1], B.ne[2]);
        TensorOpr::Dequantize(dequant_tensor_, B);
        b_ptr = &dequant_tensor_;
    }

    if (use_gemv)
    {
        if (is_b_column_major)
        {
            /*if (b_ptr->data_type == ElementType::Q8_B32T2)
            {
                DeviceTensor *x_q = CreateLocalTensor(ElementType::Q8_B32T2,
                    A.ne[0], A.ne[1], 0, true, 0);
                TensorOpr::Quantize(*x_q, A);
                ret = TensorMul::Gemv_AX(C, b_ptr, *x_q, gemv_alg);
            }
            else*/
            {
                ret = TensorMul::Gemv_AX(C, *b_ptr, A, gemv_alg, nullptr, bias);
                use_bias = false;
            }
        }
        else
        {
            ret = TensorMul::Gemv_XA(C, A, *b_ptr, gemv_alg);
        }
    }
    else
    {
        if (config_ex_->matrix_mul_alg == MatrixMulAlg::Cublas)
        {
            if (is_b_column_major)
            {
                int scenario_id = 101;
                DeviceTensor *tensor_trans = CreateLocalTensor(ElementType::F16,
                    C.Rows(), C.Columns(), 0, true, 0, scenario_id);
                ret = cublas_engine_.GemmEx(*tensor_trans, A, *b_ptr, 1.0f, 0, is_b_column_major);
                TensorOpr::Transpose(C, *tensor_trans);
            }
            else
            {
                ret = cublas_engine_.GemmEx(C, A, *b_ptr, 1.0f, 0, is_b_column_major);
            }
        }
        else
        {
            ret = TensorMul::Gemm(C, A, *b_ptr, 1.0f, 0, is_b_column_major,
                false, config_ex_->matrix_mul_alg);
        }
    }

    if (bias != nullptr && use_bias) {
        TensorOpr::Add(C, C, *bias);
    }
    return ret;
}

bool GpuInferenceWorker::DequantizeLayer(StdDeviceNetwork::EncoderLayer &target,
    const StdDeviceNetwork::EncoderLayer &source, int layer_id) const
{
    bool ret = true;
    ret = ret && DequantizeLayer(target.self_attn, source.self_attn, layer_id);
    ret = ret && DequantizeLayer(target.ffn, source.ffn, layer_id);
    return ret;
}

bool GpuInferenceWorker::DequantizeLayer(StdDeviceNetwork::DecoderLayer &target,
    const StdDeviceNetwork::DecoderLayer &source, int layer_id) const
{
    bool ret = true;
    ret = ret && DequantizeLayer(target.self_attn, source.self_attn, layer_id);
    ret = ret && DequantizeLayer(target.cross_attn, source.cross_attn, layer_id);
    ret = ret && DequantizeLayer(target.ffn, source.ffn, layer_id);
    return ret;
}

bool GpuInferenceWorker::DequantizeLayer(StdDeviceNetwork::AttentionLayer &target,
    const StdDeviceNetwork::AttentionLayer &source, int layer_id) const
{
    (void)layer_id;
    bool ret = true;
    bool be_sync = true;
    bool be_transpose = false;//!is_cpu_tensor_row_major_ && is_gpu_tensor_row_major_;

    target.pre_norm = source.pre_norm;
    target.pre_norm_b = source.pre_norm_b;

    target.qkv_b = source.qkv_b;
    target.wq_b = source.wq_b;
    target.wk_b = source.wk_b;
    target.wv_b = source.wv_b;
    target.wo_b = source.wo_b;
    
    ret = ret && DequantizeTensor(target.qkv, source.qkv, be_transpose, be_sync);
    ret = ret && DequantizeTensor(target.wq, source.wq, be_transpose, be_sync);
    ret = ret && DequantizeTensor(target.wk, source.wk, be_transpose, be_sync);
    ret = ret && DequantizeTensor(target.wv, source.wv, be_transpose, be_sync);
    ret = ret && DequantizeTensor(target.wo, source.wo, be_transpose, be_sync);

    if (ret && !be_sync) {
        ret = CudaUtil::DeviceSynchronize();
    }

    return ret;
}

bool GpuInferenceWorker::DequantizeLayer(StdDeviceNetwork::FeedForwardLayer &target,
    const StdDeviceNetwork::FeedForwardLayer &source, int layer_id) const
{
    (void)layer_id;
    bool ret = true;
    bool be_sync = true;
    bool be_transpose = false;//!is_cpu_tensor_row_major_ && is_gpu_tensor_row_major_;

    target.pre_norm = source.pre_norm;
    target.pre_norm_b = source.pre_norm_b;
    target.post_norm = source.post_norm;
    target.post_norm_b = source.post_norm_b;

    target.w1_b = source.w1_b;
    target.w2_b = source.w2_b;
    target.w3_b = source.w3_b;
    target.w1n3_b = source.w1n3_b;

    ret = ret && DequantizeTensor(target.w1, source.w1, be_transpose, be_sync);
    ret = ret && DequantizeTensor(target.w2, source.w2, be_transpose, be_sync);
    ret = ret && DequantizeTensor(target.w3, source.w3, be_transpose, be_sync);
    ret = ret && DequantizeTensor(target.w1n3, source.w1n3, be_transpose, be_sync);

    if (ret && !be_sync) {
        ret = CudaUtil::DeviceSynchronize();
    }

    return ret;
}

bool GpuInferenceWorker::DequantizeTensor(DeviceTensorEx &target,
    const DeviceTensorEx &source, bool be_transpose, bool be_sync) const
{
    bool ret = true;
    if (source.tensor != nullptr && target.tensor != nullptr
        && TensorCommon::IsQuantType(source.tensor->data_type))
    {
        //TaskMonitor tm;
        ret = TensorOpr::Dequantize(*target.tensor, source, be_transpose, be_sync);
        //tm.ShowElapsedTime(L"dequantize time: ");
    }
    else
    {
        target.tensor = source.tensor;
    }

    target.linear_quant_params = source.linear_quant_params;
    target.log_quant_params = source.log_quant_params;
    target.quant_map = source.quant_map;
    target.delta = source.delta;
    return ret;
}

DeviceTensor* GpuInferenceWorker::CreateTensor(ElementType etype,
    const DeviceTensor &ref_tensor, bool be_transpose)
{
    DeviceTensor *new_tensor = new DeviceTensor;
    new_tensor->SetAutoFree(true);
    new_tensor->New(etype, ref_tensor.dim, ref_tensor.ne);

    if (be_transpose)
    {
        TensorOpr::Reshape(*new_tensor, ref_tensor.dim, ref_tensor.ne[1],
            ref_tensor.ne[0], ref_tensor.ne[2]);
    }

    return new_tensor;
}

DeviceTensor* GpuInferenceWorker::CreateActivationTarget(const DeviceTensor &ref_tensor,
    ActivationFn fn, bool is_layer_local, int heap_idx)
{
    bool is_glu = false;
    switch (fn)
    {
    case ActivationFn::GLU_SIGMOID:
    case ActivationFn::GLU_ELU:
    case ActivationFn::GLU_RELU:
    case ActivationFn::GLU_GELU:
    case ActivationFn::GLU_SILU:
        is_glu = true;
        break;
    default: break;
    }

    if (is_glu)
    {
        const auto &ne = ref_tensor.ne;
        return CreateLocalTensor(ref_tensor.data_type, ne[0] / 2, ne[1], ne[2],
            is_layer_local, heap_idx);
    }

    return CreateLocalTensor(ref_tensor, is_layer_local, heap_idx);
}

DeviceTensor* GpuInferenceWorker::CreateLocalTensor(const DeviceTensor &ref_tensor,
    bool is_layer_local, int heap_idx)
{
    DeviceTensor *new_tensor = local_device_tensor_heap_.New(1);
    new_tensor->SetAutoFree(false);
    new_tensor->data_type = ref_tensor.data_type;
    new_tensor->SetStructure(ref_tensor.dim, ref_tensor.ne);

    auto &heap = is_layer_local ? layer_local_device_heap_ : local_device_heaps_[heap_idx];
    new_tensor->data = heap.NewHalfArray(ref_tensor.size);
    if (new_tensor->data == nullptr)
    {
        LogError("Failed to allocate tensor memory (is_layer_local: %s, size: %d (%d, %d, %d))",
            is_layer_local ? "Y" : "N", ref_tensor.size, ref_tensor.ne[0],
            ref_tensor.ne[1], ref_tensor.ne[2]);
        exit(101);
    }

    return new_tensor;
}

DeviceTensor* GpuInferenceWorker::CreateLocalTensor(ElementType etype, int ne0,
    int ne1, int ne2, bool is_layer_local, int heap_idx, int scenario_id)
{
    DeviceTensor *new_tensor = local_device_tensor_heap_.New(1);
    new_tensor->SetAutoFree(false);
    new_tensor->data_type = etype;
    new_tensor->SetStructure(ne0, ne1, ne2);

    auto &heap = is_layer_local ? layer_local_device_heap_
        : (heap_idx >= 0 ? local_device_heaps_[heap_idx] : local_device_heap_);
    new_tensor->data = heap.NewHalfArray(new_tensor->size);
    if (new_tensor->data == nullptr)
    {
        LogError("%s (is_layer_local: %s, heap_idx: %d, size: %d*%d*%d, scenario: %d)",
            "Failed to allocate tensor memory", is_layer_local ? "Y" : "N",
            heap_idx, ne0, ne1, ne2, scenario_id);
        exit(101);
    }

    return new_tensor;
}

DeviceTensor* GpuInferenceWorker::CreateView(DeviceTensor &ref_tensor,
    int y_start, int y_count)
{
    int cx = ref_tensor.ne[0], cy = ref_tensor.ne[1];
    if (ref_tensor.dim != 2 || y_start < 0 || y_count <= 0 || y_start + y_count > cy)
    {
        LogError("Invalid settings in creating the tensor view");
        return nullptr;
    }

    DeviceTensor *new_tensor = local_device_tensor_heap_.New(1);
    new_tensor->SetAutoFree(false);
    new_tensor->data_type = ref_tensor.data_type;
    new_tensor->SetStructure(ref_tensor.dim, cx, y_count);
    new_tensor->data = ref_tensor.RowData(y_start);
    return new_tensor;
}

void GpuInferenceWorker::PrintTensor(const DeviceTensor *tensor, int max_cx,
    int max_cy, int max_cz, const char *title, int layer_id)
{
    if (config_->debug.is_study_mode && config_->debug.show_tensors
        && tensor != nullptr && (id_ == 0 || global_data_ != nullptr))
    {
        if (global_data_ != nullptr) {
            global_data_->Lock4Print();
        }

        ostream &tensor_writer = tensor_writer_ != nullptr ? *tensor_writer_ : cout;
        if (layer_id >= 0) {
            tensor_writer << "worker " << id_ << ", layer: " << layer_id << endl;
        }
        else {
            tensor_writer << "worker " << id_ << endl;
        }
        tensor->Print(tensor_writer, max_cx, max_cy, max_cz, title) << endl;
        tensor_writer.flush();

        if (global_data_ != nullptr) {
            global_data_->Unlock4Print();
        }
    }
}

void GpuInferenceWorker::UpdatePerfStat(int key, const TaskMonitor &tm)
{
    float value = tm.GetElapsedTime(false) / 1000.0f;
    UpdatePerfStat(key, value);
}

void GpuInferenceWorker::UpdatePerfStat(int key, float value)
{
    if (perf_stat_ != nullptr && config_->debug.enable_perf_stat)
    {
        if (global_data_ != nullptr) {
            global_data_->Lock4Print();
        }

        auto iter = perf_stat_->time_map.find(key);
        if (iter == perf_stat_->time_map.end()) {
            perf_stat_->time_map[key] = value;
        }
        else {
            //iter->second = max(value, iter->second);
            iter->second += value;
        }

        if (global_data_ != nullptr) {
            global_data_->Unlock4Print();
        }
    }
}

bool GpuInferenceWorker::GetUseGemv(const DeviceTensor &input_tensor) const
{
    bool is_b_column_major = !config_ex_->is_gpu_tensor_row_major;
    bool use_gemv = input_tensor.Rows() == 1 && input_tensor.ne[0] % 32 == 0
        && is_b_column_major && config_ex_->gemv_alg == VectorMatrixMulAlg::Alg3;
    return use_gemv;
}

bool GpuInferenceWorker::GetUseFullQuantGemv(const DeviceTensor &input_tensor,
    const DeviceTensor &weight_tensor) const
{
    bool is_acceptable_weight_type = false;
    switch (weight_tensor.data_type)
    {
    case ElementType::Q8_B32T2:
    case ElementType::Q6_B64T1:
    case ElementType::Q5_B64T1:
    case ElementType::Q4_B32T1A:
    case ElementType::Q4_B32T1B:
    case ElementType::Q4_B64T1:
    case ElementType::Q3H_B64T1:
        is_acceptable_weight_type = true;
        break;
    default:
        break;
    }

    bool use_gemv = GetUseGemv(input_tensor);
    return use_gemv && config_ex_->enable_full_quant_gemv
        //&& (input_tensor.ne[0] > 1500 || weight_tensor.ne[1] > 1500)
        && is_acceptable_weight_type;
}

TRANSFORMER_END
INFER_FLOW_END
