#include "network_builder.h"
#include "tensor/tensor_util.h"
#if defined(USE_CUDA)
#   include "common/cuda_util.h"
#   include "tensor/device_tensor_util.h"
#endif

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

NetworkBuilder::NetworkBuilder()
{
}

NetworkBuilder::~NetworkBuilder()
{
    Clear();
}

void NetworkBuilder::Clear()
{
    context_ = nullptr;
    ClearDeviceMemory();
}

void NetworkBuilder::ClearDeviceMemory()
{
#if defined(USE_CUDA)
    device_tensor_builder_.Clear();
    is_device_tensor_builder_initialized_ = false;
#endif //USE_CUDA
}

bool NetworkBuilder::Init(const InferenceConfig &config,
    const InferenceConfigEx &config_ex, TransformerContext &ctx)
{
    config_ = &config;
    config_ex_ = &config_ex;
    context_ = &ctx;

    BuildAttnTensorIdMap(attn_tensor_id_map_);
    BuildFfnTensorIdMap(ffn_tensor_id_map_);
    BuildMoeTensorIdMap(moe_tensor_id_map_);
    return true;
}

bool NetworkBuilder::Init(NetworkBuilder &rhs)
{
    config_ = rhs.config_;
    config_ex_ = rhs.config_ex_;
    context_ = rhs.context_;

    BuildAttnTensorIdMap(attn_tensor_id_map_);
    BuildFfnTensorIdMap(ffn_tensor_id_map_);
    BuildMoeTensorIdMap(moe_tensor_id_map_);
    return true;
}

bool NetworkBuilder::InitNetworkStructure(StdNetwork &net, const ModelSpec &spec) const
{
    bool ret = true;
    const auto &hparams = spec.hyper_params;
    auto &host_net = net.host_net;
    //char prefix_buf[64];
    host_net.encoder_layers.Clear(true);
    for (int layer_idx = 0; layer_idx < hparams.encoder_layers; layer_idx++)
    {
        StdHostNetwork::EncoderLayer *layer_ptr = new StdHostNetwork::EncoderLayer;
        host_net.encoder_layers.push_back(layer_ptr);
    }

    host_net.decoder_layers.Clear(true);
    for (int layer_idx = 0; layer_idx < hparams.decoder_layers; layer_idx++)
    {
        StdHostNetwork::DecoderLayer *layer_ptr = new StdHostNetwork::DecoderLayer;
        host_net.decoder_layers.push_back(layer_ptr);

        for (int expert_id = 0; expert_id < spec.hyper_params.experts; expert_id++)
        {
            auto *expert_ptr = new StdHostNetwork::FeedForwardLayer;
            layer_ptr->moe.experts.push_back(expert_ptr);
        }
    }

    //BuildHostTensorMap(host_net);

#if defined(USE_CUDA)
    ret = InitDeviceNetStructure(net, spec);
#endif //USE_CUDA

    return ret;
}

bool NetworkBuilder::BuildHostNetwork(TransformerModel &model, const ModelSpec &spec,
    bool is_cpu_only)
{
    bool ret = false;
    ret = BuildHostNetwork_Std(model, spec, is_cpu_only);

    switch (spec.network_structure)
    {
    case NetworkType::EncoderDecoder_Transformer:
        break;
    case NetworkType::EncoderOnly_Transformer:
        break;
    case NetworkType::DecoderOnly_Transformer:
        break;
    default: break;
    }

    BuildHostTensorMap(model.std_network.host_net);
    return ret;
}

#if defined(USE_CUDA)

bool NetworkBuilder::BuildDeviceNetwork(TransformerModel &model, const ModelSpec &spec,
    const ModelPartition &model_partition, int builder_count)
{
    bool ret = false;
    int device_group_num = (int)model.spec.device_groups.size();
    auto &host_net = model.std_network.host_net;
    auto &device_net = model.std_network.device_net;
    auto &device_sub_nets = model.std_network.device_sub_nets;

    int decoder_group_num = (int)model_partition.decoder_assignments.size();
    if (decoder_group_num > device_group_num)
    {
        LogError("Something is wrong with the model partition");
        return false;
    }

    //int device_num_per_group = (int)model.spec.device_groups[0].size();

    switch (spec.multi_gpu_strategy)
    {
    case MultiGpuStrategy::BY_LAYER:
        ret = BuildDeviceNetwork_ByLayer(device_net, model, host_net, builder_count);
        break;
    case MultiGpuStrategy::BY_TENSOR:
    case MultiGpuStrategy::HYBRID:
        for (int group_idx = 0; group_idx < decoder_group_num; group_idx++)
        {
            const auto &la = model_partition.decoder_assignments[group_idx];
            ret = BuildDeviceNetwork_ByTensor(device_sub_nets, model, host_net,
                la, builder_count);
        }
        break;
    default: break;
    }

    return ret;
}

bool NetworkBuilder::InitDeviceNetStructure(StdNetwork &net, const ModelSpec &spec) const
{
    bool ret = true;
    int device_group_num = (int)spec.device_groups.size();
    //auto &host_net = net.host_net;
    auto &device_net = net.device_net;
    auto &device_sub_nets = net.device_sub_nets;

    ModelPartition model_partition;
    NetworkBuilder::GetDeviceAssignments(model_partition.encoder_assignments,
        spec, true, spec.encoder_cpu_layer_count);
    NetworkBuilder::GetDeviceAssignments(model_partition.decoder_assignments,
        spec, false, spec.decoder_cpu_layer_count);

    int decoder_group_num = (int)model_partition.decoder_assignments.size();
    if (decoder_group_num > device_group_num)
    {
        LogError("Something is wrong with the model partition");
        return false;
    }

    int device_num_per_group = (int)spec.device_groups[0].size();

    switch (spec.multi_gpu_strategy)
    {
    case MultiGpuStrategy::BY_LAYER:
        device_net.encoder_layers.Clear(true);
        device_net.decoder_layers.Clear(true);
        for (int group_idx = 0; group_idx < decoder_group_num; group_idx++)
        {
            const auto &la = model_partition.decoder_assignments[group_idx];
            if (group_idx == 0 && la.start_layer > 0) //in the case of GPU/CPU hybrid inference
            {
                for (int layer_idx = 0; layer_idx < la.start_layer; layer_idx++)
                {
                    device_net.decoder_layers.push_back(nullptr);
                }
            }

            for (int layer_id = la.start_layer; layer_id < la.end_layer; layer_id++)
            {
                auto *gpu_decoder_layer = new StdDeviceNetwork::DecoderLayer;
                BuildLayerTensorMap(gpu_decoder_layer->self_attn);
                BuildLayerTensorMap(gpu_decoder_layer->cross_attn);
                BuildLayerTensorMap(gpu_decoder_layer->ffn);
                BuildLayerTensorMap(gpu_decoder_layer->moe);

                for (int expert_id = 0; expert_id < spec.hyper_params.experts; expert_id++)
                {
                    auto *expert_ptr = new StdDeviceNetwork::FeedForwardLayer;
                    BuildLayerTensorMap(*expert_ptr);
                    gpu_decoder_layer->moe.experts.push_back(expert_ptr);
                }

                device_net.decoder_layers.push_back(gpu_decoder_layer);
            }
        }
        break;
    case MultiGpuStrategy::BY_TENSOR:
    case MultiGpuStrategy::HYBRID:
        device_sub_nets.Clear(true);
        for (int device_idx = 0; device_idx < device_num_per_group; device_idx++)
        {
            auto *new_net = new StdDeviceNetwork;
            device_sub_nets.push_back(new_net);
        }

        for (int group_idx = 0; group_idx < decoder_group_num; group_idx++)
        {
            const auto &la = model_partition.decoder_assignments[group_idx];
            if (group_idx == 0 && la.start_layer > 0) //in the case of GPU/CPU hybrid inference
            {
                for (int device_idx = 0; device_idx < device_num_per_group; device_idx++)
                {
                    for (int layer_idx = 0; layer_idx < la.start_layer; layer_idx++)
                    {
                        device_sub_nets[device_idx]->decoder_layers.push_back(nullptr);
                    }
                }
            }
        }
        break;
    default: break;
    }

    NetworkStructure::BuildTensorNameToIdMap(device_net.tensor_map);

    return ret;
}

//static
// todo: add encoder support
bool NetworkBuilder::ParseTensorName(TensorNameInfo &tni, const string &tensor_name,
    const StdDeviceNetwork &device_net, const ModelSpec &spec)
{
    tni.is_dec = strncasecmp(tensor_name.c_str(), "dec.", 4) == 0;
    if (!tni.is_dec) {
        return false;
    }

    auto pos = tensor_name.find(".", 4);
    if (pos == string::npos) {
        return false;
    }

    string str = tensor_name.substr(4, pos - 4);
    tni.layer_id = atoi(str.c_str());
    if (tni.layer_id < 0 || tni.layer_id >= spec.hyper_params.decoder_layers)
    {
        LogError("Invalid layer_id: %d", tni.layer_id);
        return false;
    }

    str = tensor_name.substr(pos + 1);
    const char *moe_prefix = "moe.shared_expert.";
    int moe_prefix_len = (int)strlen(moe_prefix);
    if ((int)str.size() > moe_prefix_len)
    {
        bool is_moe = strncasecmp(str.c_str(), moe_prefix, moe_prefix_len) == 0;
        if (is_moe)
        {
            tni.is_shared_expert = true;
            str = "feed_forward." + str.substr(moe_prefix_len);
        }
    }

    moe_prefix = "moe.expert.";
    moe_prefix_len = (int)strlen(moe_prefix);
    tni.expert_id = -1;
    if ((int)str.size() > moe_prefix_len && !tni.is_shared_expert)
    {
        bool is_moe = strncasecmp(str.c_str(), moe_prefix, moe_prefix_len) == 0;
        if (is_moe)
        {
            pos = str.find(".", moe_prefix_len);
            if (pos == string::npos) {
                return false;
            }

            string expert_id_str = str.substr(moe_prefix_len, pos - moe_prefix_len);
            tni.expert_id = atoi(expert_id_str.c_str());

            if (tni.expert_id >= 0) {
                str = "feed_forward." + str.substr(pos + 1);
            }
        }
    }

    auto iter_find = device_net.tensor_map.find(str);
    if (iter_find == device_net.tensor_map.end()) {
        return false;
    }

    tni.layer_type = iter_find->second.first;
    tni.tensor_id = iter_find->second.second;
    return true;
}

bool NetworkBuilder::BuildDeviceTensor(StdDeviceNetwork &device_net,
    const HostTensor &cpu_tensor, const TensorNameInfo &tni,
    const ModelSpec &spec)
{
    auto &target_layer = device_net.decoder_layers[tni.layer_id];

    const map<LayerTensorId, int> *tensor_id_map = nullptr;
    const StdDeviceNetwork::SubLayer *sub_layer = nullptr;
    switch (tni.layer_type)
    {
    case LayerType::SELF_ATTN:
        sub_layer = &target_layer->self_attn;
        tensor_id_map = &attn_tensor_id_map_;
        break;
    case LayerType::CROSS_ATTN:
        sub_layer = &target_layer->cross_attn;
        tensor_id_map = &attn_tensor_id_map_;
        break;
    case LayerType::FFN:
        sub_layer = &target_layer->ffn;
        tensor_id_map = &ffn_tensor_id_map_;
        break;
    case LayerType::MOE:
        sub_layer = &target_layer->moe;
        tensor_id_map = &moe_tensor_id_map_;
        break;
    default:
        break;
    }

    int expert_num = (int)target_layer->moe.experts.size();
    if (tni.expert_id >= 0 && tni.expert_id < expert_num)
    {
        sub_layer = target_layer->moe.experts[tni.expert_id];
    }

    if (tni.is_shared_expert) {
        sub_layer = &target_layer->moe.shared_expert;
    }

    if (sub_layer == nullptr || tensor_id_map == nullptr) {
        return false;
    }

    auto tensor_iter = sub_layer->tensor_map.find(tni.tensor_id);
    auto id_iter = tensor_id_map->find(tni.tensor_id);
    DeviceTensorEx *target_tensor = tensor_iter == sub_layer->tensor_map.end()
        ? nullptr : tensor_iter->second;
    int tensor_type = id_iter == tensor_id_map->end() ? 0 : id_iter->second;
    if (target_tensor == nullptr) {
        return false;
    }

    DeviceTensorBuilder::Task task;
    bool be_trans = false;
    BuildTask_Std(task, *target_tensor, cpu_tensor, tni.tensor_id, tensor_type, spec, be_trans);

    if (!is_device_tensor_builder_initialized_)
    {
        const auto &hparams = spec.hyper_params;
        int max_intermediate_size = max(hparams.decoder_intermediate_size, 3 * hparams.embd_dims);
        int max_tensor_size = hparams.embd_dims * 2 * max_intermediate_size;
        int embd_tensor_size = hparams.embd_dims * max(hparams.vocab_size, hparams.padded_vocab_size);
        int aux_buffer_capacity = max(max_tensor_size, embd_tensor_size);
        int aux_tensor_size = aux_buffer_capacity;
        int device_id = spec.device_groups[0][0];
        device_tensor_builder_.Init(device_id, aux_buffer_capacity, aux_tensor_size);

        is_device_tensor_builder_initialized_ = true;
    }

    device_tensor_builder_.Build(task);
    return true;
}

#endif //USE_CUDA

bool NetworkBuilder::BuildHostNetwork_Std(TransformerModel &model, const ModelSpec &spec,
    bool is_cpu_only)
{
    const auto &hparams = spec.hyper_params;
    //const auto &tensor_table = model.tensor_spec_table;
    auto &host_net = model.std_network.host_net;

    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(spec.network_structure);
    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(spec.network_structure);

    model.encoder_embeddings = model.FindHostTensor("enc.token_embeddings.weight");
    model.decoder_embeddings = model.FindHostTensor("dec.token_embeddings.weight");
    if (is_decoder_only && model.decoder_embeddings == nullptr)
    {
        model.decoder_embeddings = model.FindHostTensor("token_embeddings.weight");
    }
    if (is_encoder_only && model.encoder_embeddings == nullptr)
    {
        model.encoder_embeddings = model.FindHostTensor("token_embeddings.weight");
    }

    host_net.encoder_pos_embeddings = model.FindHostTensor("enc.pos_embeddings.weight");
    host_net.encoder_token_type_embeddings = model.FindHostTensor("enc.token_type_embeddings.weight");
    host_net.decoder_pos_embeddings = model.FindHostTensor("dec.pos_embeddings.weight");

    host_net.encoder_input_norm = model.FindHostTensor("enc.input_norm.weight");
    host_net.encoder_input_norm_b = model.FindHostTensor("enc.input_norm.bias");
    host_net.decoder_input_norm = model.FindHostTensor("dec.input_norm.weight");
    host_net.decoder_input_norm_b = model.FindHostTensor("dec.input_norm.bias");

    host_net.encoder_output_norm = model.FindHostTensor("enc.output_norm.weight");
    host_net.encoder_output_norm_b = model.FindHostTensor("enc.output_norm.bias");
    host_net.encoder_output_post_norm = model.FindHostTensor("enc.output_post_norm.weight");
    host_net.encoder_output_post_norm_b = model.FindHostTensor("enc.output_post_norm.bias");
    host_net.decoder_output_norm = model.FindHostTensor("dec.output_norm.weight");
    host_net.decoder_output_norm_b = model.FindHostTensor("dec.output_norm.bias");

    auto *output_tensor = model.FindHostTensor("enc.output.weight");
    host_net.encoder_output = output_tensor;

    host_net.encoder_output_b = model.FindHostTensor("enc.output.bias");

    output_tensor = model.FindHostTensor("output.weight");
    host_net.output = output_tensor;
    host_net.output_b = model.FindHostTensor("output.bias");

    if (host_net.output != nullptr && spec.normalize_lm_head
        && (host_net.output->data_type == ElementType::F16
            || host_net.output->data_type == ElementType::F32))
    {
        TensorUtil::NormalizeByRow(*output_tensor);
    }

    host_net.input_transform.dense.weight = model.FindHostTensor("input_transform.weight");
    host_net.input_transform.dense.bias = model.FindHostTensor("input_transform.bias");
    host_net.input_transform.pre_norm.weight = model.FindHostTensor("input_transform.pre_norm.weight");
    host_net.input_transform.pre_norm.bias = model.FindHostTensor("input_transform.pre_norm.bias");
    host_net.input_transform.post_norm.weight = model.FindHostTensor("input_transform.post_norm.weight");
    host_net.input_transform.post_norm.bias = model.FindHostTensor("input_transform.post_norm.bias");

    host_net.output_transform.dense.weight = model.FindHostTensor("output_transform.weight");
    host_net.output_transform.dense.bias = model.FindHostTensor("output_transform.bias");
    host_net.output_transform.pre_norm.weight = model.FindHostTensor("output_transform.pre_norm.weight");
    host_net.output_transform.pre_norm.bias = model.FindHostTensor("output_transform.pre_norm.bias");
    host_net.output_transform.post_norm.weight = model.FindHostTensor("output_transform.post_norm.weight");
    host_net.output_transform.post_norm.bias = model.FindHostTensor("output_transform.post_norm.bias");

    char prefix_buf[64];
    for (int layer_idx = 0; layer_idx < hparams.encoder_layers; layer_idx++)
    {
        auto *layer_ptr = host_net.encoder_layers[layer_idx];

        sprintf(prefix_buf, "enc.%d.self_attn", layer_idx);
        SetAttentionLayer(layer_ptr->self_attn, model, prefix_buf);

        sprintf(prefix_buf, "enc.%d.feed_forward", layer_idx);
        SetFeedForwardLayer(layer_ptr->ffn, model, prefix_buf);

        sprintf(prefix_buf, "enc.%d.moe", layer_idx);
        SetFfnMoeLayer(layer_ptr->moe, model, prefix_buf);
    }

    int decoder_cpu_layer_count = is_cpu_only ? hparams.decoder_layers
        : spec.decoder_cpu_layer_count;
    int end_layer = spec.is_eager_device_building
        ? min(decoder_cpu_layer_count, hparams.decoder_layers)
        : hparams.decoder_layers;
    for (int layer_idx = 0; layer_idx < end_layer; layer_idx++)
    {
        auto *layer_ptr = host_net.decoder_layers[layer_idx];

        sprintf(prefix_buf, "dec.%d.self_attn", layer_idx);
        SetAttentionLayer(layer_ptr->self_attn, model, prefix_buf);

        sprintf(prefix_buf, "dec.%d.cross_attn", layer_idx);
        SetAttentionLayer(layer_ptr->cross_attn, model, prefix_buf);

        sprintf(prefix_buf, "dec.%d.feed_forward", layer_idx);
        SetFeedForwardLayer(layer_ptr->ffn, model, prefix_buf);

        sprintf(prefix_buf, "dec.%d.moe", layer_idx);
        SetFfnMoeLayer(layer_ptr->moe, model, prefix_buf);
    }

    return true;
}

bool NetworkBuilder::BuildGgmlNetwork(TransformerModel &model, const ModelSpec &spec,
    ggml_context* &ctx, int encoder_layer_count, int decoder_layer_count)
{
    auto &host_net = model.std_network.host_net;
    auto &ggml_net = model.std_network.ggml_net;

    ggml_net.encoder_layers.Clear(true);
    ggml_net.decoder_layers.Clear(true);

    // embeddings
    ggml_net.encoder_embeddings = ConvertHostToGgml(model.encoder_embeddings, ctx);
    ggml_net.decoder_embeddings = ConvertHostToGgml(model.decoder_embeddings, ctx);

    ggml_net.encoder_pos_embeddings = ConvertHostToGgml(host_net.encoder_pos_embeddings, ctx);
    ggml_net.encoder_token_type_embeddings = ConvertHostToGgml(host_net.encoder_token_type_embeddings, ctx);
    ggml_net.decoder_pos_embeddings = ConvertHostToGgml(host_net.decoder_pos_embeddings, ctx);

    ggml_net.encoder_input_norm = ConvertHostToGgml(host_net.encoder_input_norm, ctx);
    ggml_net.encoder_input_norm_b = ConvertHostToGgml(host_net.encoder_input_norm_b, ctx);
    ggml_net.decoder_input_norm = ConvertHostToGgml(host_net.decoder_input_norm, ctx);
    ggml_net.decoder_input_norm_b = ConvertHostToGgml(host_net.decoder_input_norm_b, ctx);

    ggml_net.encoder_output_norm = ConvertHostToGgml(host_net.encoder_output_norm, ctx);
    ggml_net.encoder_output_norm_b = ConvertHostToGgml(host_net.encoder_output_norm_b, ctx);
    ggml_net.encoder_output_post_norm = ConvertHostToGgml(host_net.encoder_output_post_norm, ctx);
    ggml_net.encoder_output_post_norm_b = ConvertHostToGgml(host_net.encoder_output_post_norm_b, ctx);
    ggml_net.encoder_output = ConvertHostToGgml(host_net.encoder_output, ctx);
    ggml_net.encoder_output_b = ConvertHostToGgml(host_net.encoder_output_b, ctx);

    ggml_net.decoder_output_norm = ConvertHostToGgml(host_net.decoder_output_norm, ctx);
    ggml_net.decoder_output_norm_b = ConvertHostToGgml(host_net.decoder_output_norm_b, ctx);
    ggml_net.output = ConvertHostToGgml(host_net.output, ctx);

    // input_transform & output_transform
    // ggml_net.output_transform = new StdGgmlNetwork::SimpleLayer;
    BuildGgmlNetwork_SimpleLayer(ggml_net.input_transform, host_net.input_transform, ctx);
    BuildGgmlNetwork_SimpleLayer(ggml_net.output_transform, host_net.output_transform, ctx);

    // enocder layers
    int layer_num = (int)host_net.encoder_layers.size();
    for (int layer_idx = 0; layer_idx < min(layer_num, encoder_layer_count); layer_idx++)
    {
        const auto &host_layer = host_net.encoder_layers[layer_idx];
        auto *ggml_layer = new StdGgmlNetwork::EncoderLayer;
        BuildGgmlNetwork_EncoderLayer(ggml_layer, host_layer, ctx, spec);
        ggml_net.encoder_layers.push_back(ggml_layer);
    }

    // decoder layers
    layer_num = (int)host_net.decoder_layers.size();
    for (int layer_idx = 0; layer_idx < min(layer_num, decoder_layer_count); layer_idx++)
    {
        const auto &host_layer = host_net.decoder_layers[layer_idx];
        auto *ggml_layer = new StdGgmlNetwork::DecoderLayer;
        BuildGgmlNetwork_DecoderLayer(ggml_layer, host_layer, ctx, spec);
        ggml_net.decoder_layers.push_back(ggml_layer);
    }

    return true;
}

void NetworkBuilder::SetAttentionLayer(StdHostNetwork::AttentionLayer &layer,
    const TransformerModel &model, const char *prefix)
{
    char buf[255];
    sprintf(buf, "%s.pre_norm.weight", prefix);
    layer.pre_norm = model.FindHostTensor(buf);
    sprintf(buf, "%s.pre_norm.bias", prefix);
    layer.pre_norm_b = model.FindHostTensor(buf);
    sprintf(buf, "%s.post_norm.weight", prefix);
    layer.post_norm = model.FindHostTensor(buf);
    sprintf(buf, "%s.post_norm.bias", prefix);
    layer.post_norm_b = model.FindHostTensor(buf);

    sprintf(buf, "%s.qkv.weight", prefix);
    layer.qkv = model.FindHostTensor(buf);
    sprintf(buf, "%s.qkv.bias", prefix);
    layer.qkv_b = model.FindHostTensor(buf);

    sprintf(buf, "%s.wq.weight", prefix);
    layer.wq = model.FindHostTensor(buf);
    sprintf(buf, "%s.wq.bias", prefix);
    layer.wq_b = model.FindHostTensor(buf);
    sprintf(buf, "%s.wk.weight", prefix);
    layer.wk = model.FindHostTensor(buf);
    sprintf(buf, "%s.wk.bias", prefix);
    layer.wk_b = model.FindHostTensor(buf);
    sprintf(buf, "%s.wv.weight", prefix);
    layer.wv = model.FindHostTensor(buf);
    sprintf(buf, "%s.wv.bias", prefix);
    layer.wv_b = model.FindHostTensor(buf);
    sprintf(buf, "%s.wo.weight", prefix);
    layer.wo = model.FindHostTensor(buf);
    sprintf(buf, "%s.wo.bias", prefix);
    layer.wo_b = model.FindHostTensor(buf);
}

void NetworkBuilder::SetFeedForwardLayer(StdHostNetwork::FeedForwardLayer &layer,
    const TransformerModel &model, const char *prefix)
{
    char buf[255];
    sprintf(buf, "%s.pre_norm.weight", prefix);
    layer.pre_norm = model.FindHostTensor(buf);
    sprintf(buf, "%s.pre_norm.bias", prefix);
    layer.pre_norm_b = model.FindHostTensor(buf);
    sprintf(buf, "%s.post_norm.weight", prefix);
    layer.post_norm = model.FindHostTensor(buf);
    sprintf(buf, "%s.post_norm.bias", prefix);
    layer.post_norm_b = model.FindHostTensor(buf);

    sprintf(buf, "%s.w1.weight", prefix);
    layer.w1 = model.FindHostTensor(buf);
    sprintf(buf, "%s.w1.bias", prefix);
    layer.w1_b = model.FindHostTensor(buf);
    sprintf(buf, "%s.w2.weight", prefix);
    layer.w2 = model.FindHostTensor(buf);
    sprintf(buf, "%s.w2.bias", prefix);
    layer.w2_b = model.FindHostTensor(buf);
    sprintf(buf, "%s.w3.weight", prefix);
    layer.w3 = model.FindHostTensor(buf);
    sprintf(buf, "%s.w3.bias", prefix);
    layer.w3_b = model.FindHostTensor(buf);
}

void NetworkBuilder::SetFfnMoeLayer(StdHostNetwork::FfnMoeLayer &layer,
    const TransformerModel &model, const char *prefix)
{
    char buf[255], prefix_buf[255];
    sprintf(buf, "%s.gate.weight", prefix);
    layer.gate.weight = model.FindHostTensor(buf);
    sprintf(buf, "%s.gate.bias", prefix);
    layer.gate.bias = model.FindHostTensor(buf);

    sprintf(prefix_buf, "%s.shared_expert", prefix);
    SetFeedForwardLayer(layer.shared_expert, model, prefix_buf);

    int expert_num = model.spec.hyper_params.experts;
    for (int expert_id = 0; expert_id < expert_num; expert_id++)
    {
        auto *ffn_layer = new StdHostNetwork::FeedForwardLayer;
        layer.experts.push_back(ffn_layer);

        sprintf(prefix_buf, "%s.expert.%d", prefix, expert_id);
        SetFeedForwardLayer(*ffn_layer, model, prefix_buf);
    }
}

#if defined(USE_CUDA)

bool NetworkBuilder::BuildDeviceNetwork_Embd(StdDeviceNetwork &device_net,
    TransformerModel &model, const StdHostNetwork &host_net,
    HostHalfBuffer &aux_buffer)
{
    int embd_tensor_size1 = model.encoder_embeddings == nullptr ? 0
        : (int)TensorUtil::ElementCount(*model.encoder_embeddings);
    int embd_tensor_size2 = model.decoder_embeddings == nullptr ? 0
        : (int)TensorUtil::ElementCount(*model.decoder_embeddings);

    if (model.spec.be_host_embeddings)
    {
        device_net.encoder_embeddings = nullptr;
        device_net.decoder_embeddings = nullptr;

        if (model.encoder_embeddings != nullptr)
        {
            auto &embd = model.std_network.host_net.encoder_embeddings;
            embd.New(embd_tensor_size1);
            DeviceTensorUtil::GetFP16List(embd, *model.encoder_embeddings);
        }

        if (model.decoder_embeddings != nullptr) 
        {
            auto &embd = model.std_network.host_net.decoder_embeddings;
            embd.New(embd_tensor_size2);
            DeviceTensorUtil::GetFP16List(embd, *model.decoder_embeddings);
        }
    }
    else
    {
        if (model.encoder_embeddings != nullptr)
        {
            device_net.encoder_embeddings = BuildDeviceTensor_ForceDequant(
                *model.encoder_embeddings, aux_buffer, model.spec);
        }

        if (model.decoder_embeddings != nullptr)
        {
            device_net.decoder_embeddings = BuildDeviceTensor_ForceDequant(
                *model.decoder_embeddings, aux_buffer, model.spec);
        }
    }

    if (host_net.encoder_pos_embeddings != nullptr)
    {
        device_net.encoder_pos_embeddings = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_pos_embeddings, aux_buffer, model.spec);
    }

    if (host_net.encoder_token_type_embeddings != nullptr)
    {
        device_net.encoder_token_type_embeddings = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_token_type_embeddings, aux_buffer, model.spec);
    }

    if (host_net.decoder_pos_embeddings != nullptr)
    {
        device_net.decoder_pos_embeddings = BuildDeviceTensor_ForceDequant(
            *host_net.decoder_pos_embeddings, aux_buffer, model.spec);
    }

    return true;
}

bool NetworkBuilder::BuildDeviceNetwork_SimpleLayer(StdDeviceNetwork::SimpleLayer &device_layer,
    TransformerModel &model, const StdHostNetwork::SimpleLayer &host_layer,
    HostHalfBuffer &aux_buffer)
{
    if (host_layer.dense.weight != nullptr)
    {
        device_layer.dense.weight = BuildDeviceTensor_ForceDequant(
            *host_layer.dense.weight, aux_buffer, model.spec);
    }

    if (host_layer.dense.bias != nullptr)
    {
        device_layer.dense.bias = BuildDeviceTensor_ForceDequant(
            *host_layer.dense.bias, aux_buffer, model.spec);
    }

    if (host_layer.pre_norm.weight != nullptr)
    {
        device_layer.pre_norm.weight = BuildDeviceTensor_ForceDequant(
            *host_layer.pre_norm.weight, aux_buffer, model.spec);
    }

    if (host_layer.pre_norm.bias != nullptr)
    {
        device_layer.pre_norm.bias = BuildDeviceTensor_ForceDequant(
            *host_layer.pre_norm.bias, aux_buffer, model.spec);
    }

    if (host_layer.post_norm.weight != nullptr)
    {
        device_layer.post_norm.weight = BuildDeviceTensor_ForceDequant(
            *host_layer.post_norm.weight, aux_buffer, model.spec);
    }

    if (host_layer.post_norm.bias != nullptr)
    {
        device_layer.post_norm.bias = BuildDeviceTensor_ForceDequant(
            *host_layer.post_norm.bias, aux_buffer, model.spec);
    }

    return true;
}

bool NetworkBuilder::BuildDeviceNetwork_EncoderOut(StdDeviceNetwork &device_net,
    TransformerModel &model, const StdHostNetwork &host_net,
    HostHalfBuffer &aux_buffer, DeviceTensor &aux_tensor)
{
    const auto &config_ex = *config_ex_;
    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(
        model.spec.network_structure);
    bool be_trans = config_ex.is_gpu_tensor_row_major && !model.is_cpu_tensor_row_major;

    if (host_net.encoder_output_norm != nullptr)
    {
        device_net.encoder_output_norm = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_output_norm, aux_buffer, model.spec);
    }

    if (host_net.encoder_output_norm_b != nullptr)
    {
        device_net.encoder_output_norm_b = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_output_norm_b, aux_buffer, model.spec);
    }

    if (host_net.encoder_output_post_norm != nullptr)
    {
        device_net.encoder_output_post_norm = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_output_post_norm, aux_buffer, model.spec);
    }

    if (host_net.encoder_output_post_norm_b != nullptr)
    {
        device_net.encoder_output_post_norm_b = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_output_post_norm_b, aux_buffer, model.spec);
    }

    const auto *host_output = host_net.encoder_output != nullptr ? host_net.encoder_output
        : model.encoder_embeddings;
    if (host_output != nullptr && is_encoder_only)
    {
        device_net.encoder_output = BuildDeviceTensor_ForceDequant(*host_output,
            aux_buffer, model.spec, be_trans, &aux_tensor);
    }

    if (host_net.encoder_output_b != nullptr)
    {
        device_net.encoder_output_b = BuildDeviceTensor_ForceDequant(*host_net.encoder_output_b,
            aux_buffer, model.spec, be_trans, &aux_tensor);
    }

    return true;
}

bool NetworkBuilder::BuildDeviceNetwork_DecoderOut(StdDeviceNetwork &device_net,
    TransformerModel &model, const StdHostNetwork &host_net,
    HostHalfBuffer &aux_buffer, DeviceTensor &aux_tensor)
{
    const auto &config_ex = *config_ex_;
    bool be_trans = config_ex.is_gpu_tensor_row_major && !model.is_cpu_tensor_row_major;

    if (host_net.decoder_output_norm != nullptr)
    {
        device_net.decoder_output_norm = BuildDeviceTensor_ForceDequant(
            *host_net.decoder_output_norm, aux_buffer, model.spec);
    }

    if (host_net.decoder_output_norm_b != nullptr)
    {
        device_net.decoder_output_norm_b = BuildDeviceTensor_ForceDequant(
            *host_net.decoder_output_norm_b, aux_buffer, model.spec);
    }

    const auto *host_output = host_net.output != nullptr
        ? host_net.output : model.decoder_embeddings;
    if (host_output != nullptr)
    {
        device_net.output = BuildDeviceTensor_ForceDequant(*host_output,
            aux_buffer, model.spec, be_trans, &aux_tensor);
    }

    if (host_net.output_b != nullptr)
    {
        device_net.output_b = BuildDeviceTensor_ForceDequant(
            *host_net.output_b, aux_buffer, model.spec);
    }

    uint32_t decoder_layer_num = (uint32_t)host_net.decoder_layers.size();
    if (TensorCommon::IsQuantType(model.spec.device_weight_data_type)
        && decoder_layer_num <= 20 && host_output != nullptr)
    {
        device_net.output_quant = BuildDeviceTensor_Quant(*host_output,
            aux_buffer, model.spec, be_trans, &aux_tensor);
    }

    return true;
}

bool NetworkBuilder::BuildDeviceNetwork_Input(StdDeviceNetwork &device_net,
    TransformerModel &model, const StdHostNetwork &host_net, HostHalfBuffer &aux_buffer)
{
    if (host_net.encoder_input_norm != nullptr)
    {
        device_net.encoder_input_norm = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_input_norm, aux_buffer, model.spec);
    }

    if (host_net.encoder_input_norm_b != nullptr)
    {
        device_net.encoder_input_norm_b = BuildDeviceTensor_ForceDequant(
            *host_net.encoder_input_norm_b, aux_buffer, model.spec);
    }

    if (host_net.decoder_input_norm != nullptr)
    {
        device_net.decoder_input_norm = BuildDeviceTensor_ForceDequant(
            *host_net.decoder_input_norm, aux_buffer, model.spec);
    }

    if (host_net.decoder_input_norm_b != nullptr)
    {
        device_net.decoder_input_norm_b = BuildDeviceTensor_ForceDequant(
            *host_net.decoder_input_norm_b, aux_buffer, model.spec);
    }

    return true;
}

bool NetworkBuilder::BuildDeviceNets_EncoderOut(PtrVector<StdDeviceNetwork> &device_sub_nets,
    TransformerModel &model, const StdHostNetwork &host_net,
    const vector<int> &device_group, HostHalfBuffer &aux_buffer)
{
    bool ret = true;
    TensorPartitionType partition_type = TensorPartitionType::DUP;
    PtrVector<DeviceTensorEx> tensor_list;

    if (ret && host_net.encoder_output_norm != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: output_norm");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.encoder_output_norm,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->encoder_output_norm = tensor_list[idx]->tensor;
        }
    }

    if (ret && host_net.encoder_output_norm_b != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: output_norm_b");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.encoder_output_norm_b,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->encoder_output_norm_b = tensor_list[idx]->tensor;
        }
    }

    return ret;
}

bool NetworkBuilder::BuildDeviceNets_DecoderOut(PtrVector<StdDeviceNetwork> &device_sub_nets,
    TransformerModel &model, const StdHostNetwork &host_net, const vector<int> &device_group,
    HostHalfBuffer &aux_buffer, DeviceTensor &aux_tensor)
{
    (void)aux_tensor;
    bool ret = true;
    TensorPartitionType partition_type = TensorPartitionType::DUP;
    PtrVector<DeviceTensorEx> tensor_list;

    if (ret && host_net.decoder_output_norm != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: output_norm");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.decoder_output_norm,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->decoder_output_norm = tensor_list[idx]->tensor;
        }
    }

    if (ret && host_net.decoder_output_norm_b != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: output_norm_b");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.decoder_output_norm_b,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->decoder_output_norm_b = tensor_list[idx]->tensor;
        }
    }

    return ret;
}

bool NetworkBuilder::BuildDeviceNets_Input(PtrVector<StdDeviceNetwork> &device_sub_nets,
    TransformerModel &model, const StdHostNetwork &host_net,
    const vector<int> &device_group, HostHalfBuffer &aux_buffer)
{
    bool ret = true;
    TensorPartitionType partition_type = TensorPartitionType::DUP;
    PtrVector<DeviceTensorEx> tensor_list;

    if (ret && host_net.encoder_input_norm != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: norm");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.encoder_input_norm,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->encoder_input_norm = tensor_list[idx]->tensor;
        }
    }

    if (ret && host_net.encoder_input_norm_b != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: norm_b");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.encoder_input_norm_b,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->encoder_input_norm_b = tensor_list[idx]->tensor;
        }
    }

    if (ret && host_net.decoder_input_norm != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: norm");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.decoder_input_norm,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->decoder_input_norm = tensor_list[idx]->tensor;
        }
    }

    if (ret && host_net.decoder_input_norm_b != nullptr)
    {
        //LogKeyInfo("BuildDeviceTensor: norm_b");
        ret = BuildDeviceTensors_ForceDequant(tensor_list, *host_net.decoder_input_norm_b,
            aux_buffer, model.spec, device_group, false, partition_type);
        for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            device_sub_nets[idx]->decoder_input_norm_b = tensor_list[idx]->tensor;
        }
    }

    return ret;
}

bool NetworkBuilder::BuildDeviceNetwork_ByLayer(StdDeviceNetwork &device_net,
    TransformerModel &model, const StdHostNetwork &host_net, int builder_count)
{
    bool ret = true;
    const auto &config = *config_;
    const auto &config_ex = *config_ex_;

    uint32_t encoder_layer_num = (uint32_t)host_net.encoder_layers.size();
    uint32_t decoder_layer_num = (uint32_t)host_net.decoder_layers.size();
    int encoder_start_layer = max(0, model.spec.encoder_cpu_layer_count);
    int decoder_start_layer = max(0, model.spec.decoder_cpu_layer_count);
    int encoder_end_layer = (int)encoder_layer_num;
    int decoder_end_layer = (int)decoder_layer_num;
    if (encoder_start_layer >= encoder_end_layer
        && decoder_start_layer >= decoder_end_layer)
    {
        return true;
    }

    int max_tensor_size = host_net.MaxTensorSize();
    int embd_tensor_size1 = model.encoder_embeddings == nullptr ? 0
        : (int)TensorUtil::ElementCount(*model.encoder_embeddings);
    int embd_tensor_size2 = model.decoder_embeddings == nullptr ? 0
        : (int)TensorUtil::ElementCount(*model.decoder_embeddings);
    int max_embd_tensor_size = max(embd_tensor_size1, embd_tensor_size2);
    int aux_buffer_size = max(max_embd_tensor_size, max_tensor_size);
    //LogKeyInfo("max_tensor_size: %d, aux_buffer_size: %d",
    //    max_tensor_size, aux_buffer_size);

    const vector<int> &last_group = *model.spec.device_groups.rbegin();
    CudaUtil::SetDevice(last_group[0]);

    HostHalfBuffer aux_buffer;
    aux_buffer.New(aux_buffer_size);

    DeviceTensor aux_tensor;
    //if (config_ex.is_gpu_tensor_row_major) {
        aux_tensor.New(ElementType::F16, aux_buffer_size);
    //}

    //ElementType data_type = model.spec.device_data_type;
    //LogKeyInfo("device_data_type: %d", data_type);
    bool be_trans = config_ex.is_gpu_tensor_row_major && !model.is_cpu_tensor_row_major;

    vector<LayerAssignment> encoder_layer_assignments, decoder_layer_assignments;
    SplitGpuLayers(encoder_layer_assignments, encoder_start_layer, encoder_end_layer, model.spec);
    SplitGpuLayers(decoder_layer_assignments, decoder_start_layer, decoder_end_layer, model.spec);

    TaskMonitor tm;
    BuildDeviceNetwork_Embd(device_net, model, host_net, aux_buffer);

    BuildDeviceNetwork_SimpleLayer(device_net.input_transform, model,
        host_net.input_transform, aux_buffer);
    BuildDeviceNetwork_SimpleLayer(device_net.output_transform, model,
        host_net.output_transform, aux_buffer);

    BuildDeviceNetwork_EncoderOut(device_net, model, host_net, aux_buffer, aux_tensor);

    BuildDeviceNetwork_DecoderOut(device_net, model, host_net, aux_buffer, aux_tensor);

    if (encoder_layer_assignments.size() > 1)
    {
        const auto &la = encoder_layer_assignments[0];
        CudaUtil::SetDevice(*la.devices->rbegin()); //!!!
    }
    else if (decoder_layer_assignments.size() > 1)
    {
        const auto &la = decoder_layer_assignments[0];
        CudaUtil::SetDevice(*la.devices->rbegin()); //!!!
    }

    BuildDeviceNetwork_Input(device_net, model, host_net, aux_buffer);

    tm.Progress(1);
    uint32_t proc_count = 1;

    bool is_decoder = true;
    int la_count = model.spec.is_eager_device_building ? 0
        : (int)decoder_layer_assignments.size();
    for (int la_idx = 0; ret && la_idx < la_count; la_idx++)
    {
        const auto &la = decoder_layer_assignments[la_idx];
        int device_id = *la.devices->rbegin();
        CudaUtil::SetDevice(device_id); //!!!
        //LogKeyInfo("device_id: %d, decoder layer range: [%d, %d)", la.device_id,
        //    la.start_layer, la.end_layer);

        PtrVector<DeviceTensorBuilder> builder_list;
        int aux_tensor_size = max_tensor_size;
        for (int builder_idx = 0; builder_idx < builder_count; builder_idx++)
        {
            auto *builder = new DeviceTensorBuilder;
            builder->Init(device_id, aux_buffer_size, aux_tensor_size);
            builder_list.push_back(builder);
        }

        for (int layer_id = la.start_layer; ret && layer_id < la.end_layer; layer_id++)
        {
            const auto &cpu_decoder_layer = *host_net.decoder_layers[layer_id];
            auto *gpu_decoder_layer = device_net.decoder_layers[layer_id];
            //LogKeyInfo("layer %d", layer_id);

            int builder_idx = layer_id % builder_count;
            auto *builder = builder_list[builder_idx];
            ret = AddLayerTasks_Std(*builder, gpu_decoder_layer->self_attn,
                cpu_decoder_layer.self_attn, model.spec, layer_id,
                be_trans, is_decoder, false);
            ret = ret && AddLayerTasks_Std(*builder, gpu_decoder_layer->cross_attn,
                cpu_decoder_layer.cross_attn, model.spec, layer_id,
                be_trans, is_decoder, true);
            ret = ret && AddLayerTasks_Std(*builder, gpu_decoder_layer->ffn,
                cpu_decoder_layer.ffn, model.spec, layer_id, be_trans);

            if (config.debug.show_tensors && layer_id == 0)
            {
                //if (gpu_layer.wq != nullptr) {
                //    gpu_layer.wq->Print(cout, 8, 8, 8, "gpu_layer.wq: ") << endl;
                //}
            }
        }

        //run
        for (auto *builder : builder_list) {
            builder->Create();
        }

        for (auto *builder : builder_list)
        {
            builder->Join();

            bool has_error = builder->HasError();
            if (has_error)
            {
                LogError("Error occurred in building device tensors");
                return false;
            }
        }

        tm.Progress(la_idx + 1 + proc_count);
    }
    proc_count += la_count;

    is_decoder = false;
    la_count = (int)encoder_layer_assignments.size();
    for (int la_idx = 0; ret && la_idx < la_count; la_idx++)
    {
        const auto &la = encoder_layer_assignments[la_idx];
        int device_id = *la.devices->rbegin();
        CudaUtil::SetDevice(device_id); //!!!
        //LogKeyInfo("device_id: %d, encoder layer range: [%d, %d)", la.device_id,
        //    la.start_layer, la.end_layer);

        PtrVector<DeviceTensorBuilder> builder_list;
        int aux_tensor_size = max_tensor_size;
        for (int builder_idx = 0; builder_idx < builder_count; builder_idx++)
        {
            auto *builder = new DeviceTensorBuilder;
            builder->Init(device_id, aux_buffer_size, aux_tensor_size);
            builder_list.push_back(builder);
        }

        for (int layer_id = la.start_layer; ret && layer_id < la.end_layer; layer_id++)
        {
            const auto &cpu_encoder_layer = *host_net.encoder_layers[layer_id];
            StdDeviceNetwork::EncoderLayer *gpu_encoder_layer = new StdDeviceNetwork::EncoderLayer;
            BuildLayerTensorMap(gpu_encoder_layer->self_attn);
            BuildLayerTensorMap(gpu_encoder_layer->ffn);
            device_net.encoder_layers.push_back(gpu_encoder_layer);
            //LogKeyInfo("layer %d", layer_id);

            int builder_idx = layer_id % builder_count;
            auto *builder = builder_list[builder_idx];
            ret = AddLayerTasks_Std(*builder, gpu_encoder_layer->self_attn,
                cpu_encoder_layer.self_attn, model.spec, layer_id,
                be_trans, is_decoder, false);
            ret = ret && AddLayerTasks_Std(*builder, gpu_encoder_layer->ffn,
                cpu_encoder_layer.ffn, model.spec, layer_id, be_trans);
        }

        //run
        for (auto *builder : builder_list) {
            builder->Create();
        }

        for (auto *builder : builder_list)
        {
            builder->Join();

            bool has_error = builder->HasError();
            if (has_error)
            {
                LogError("Error occurred in building device tensors");
                return false;
            }
        }

        tm.Progress(la_idx + 1 + proc_count);
    }
    tm.End();

    return ret;
}

bool NetworkBuilder::BuildDeviceNetwork_ByTensor(
    PtrVector<StdDeviceNetwork> &device_sub_nets,
    TransformerModel &model, const StdHostNetwork &host_net,
    const LayerAssignment &la, int builder_count)
{
    bool ret = true;
    const auto &config_ex = *config_ex_;
    int device_num = (int)la.devices->size();
    const auto &hparams = model.spec.hyper_params;

    if (hparams.decoder_heads % device_num != 0)
    {
        LogError("Tensor parallelism requires that %s (heads: %d, devices: %d)",
            "the head number is a multiple of the device count",
            hparams.decoder_heads, device_num);
        return false;
    }

    if (model.spec.qkv_format != 0)
    {
        LogError("qkv_format %d is not compatible with tensor parallelism",
            model.spec.qkv_format);
        return false;
    }

    if (la.start_layer >= la.end_layer) {
        return true;
    }

    int max_tensor_size = StdHostNetwork::MaxTensorSize(*host_net.decoder_layers[0]);
    int embd_tensor_size1 = model.encoder_embeddings == nullptr ? 0
        : (int)TensorUtil::ElementCount(*model.encoder_embeddings);
    int embd_tensor_size2 = model.decoder_embeddings == nullptr ? 0
        : (int)TensorUtil::ElementCount(*model.decoder_embeddings);
    int max_embd_tensor_size = max(embd_tensor_size1, embd_tensor_size2);
    int aux_buffer_size = max(max_embd_tensor_size, max_tensor_size);
    //LogKeyInfo("max_tensor_size: %d, aux_buffer_size: %d",
    //    max_tensor_size, aux_buffer_size);

    HostHalfBuffer aux_buffer;
    aux_buffer.New(aux_buffer_size);

    DeviceTensor aux_tensor;
    //if (config_ex.is_gpu_tensor_row_major) {
        aux_tensor.New(ElementType::F16, max_tensor_size);
    //}

    //ElementType data_type = model.spec.device_data_type;
    //LogKeyInfo("device_data_type: %d", data_type);
    bool be_trans = config_ex.is_gpu_tensor_row_major && !model.is_cpu_tensor_row_major;

    TaskMonitor tm;
    if (la.start_layer == 0)
    {
        BuildDeviceNetwork_Embd(model.std_network.device_net, model, host_net, aux_buffer);

        BuildDeviceNetwork_SimpleLayer(model.std_network.device_net.input_transform,
            model, host_net.input_transform, aux_buffer);
    }

    CudaUtil::SetDevice(*la.devices->rbegin());

    if (la.end_layer >= model.spec.hyper_params.encoder_layers)
    {
        BuildDeviceNetwork_SimpleLayer(model.std_network.device_net.output_transform,
            model, host_net.output_transform, aux_buffer);

        BuildDeviceNets_EncoderOut(device_sub_nets, model, host_net, *la.devices, aux_buffer);
    }

    if (la.end_layer >= model.spec.hyper_params.decoder_layers)
    {
        BuildDeviceNets_DecoderOut(device_sub_nets, model, host_net,
            *la.devices, aux_buffer, aux_tensor);

        const auto *host_output = host_net.output != nullptr
            ? host_net.output : model.decoder_embeddings;
        if (ret && host_output != nullptr)
        {
            //LogKeyInfo("BuildDeviceTensor: output");
            CudaUtil::SetDevice(*la.devices->rbegin());
            DeviceTensor *output = BuildDeviceTensor_ForceDequant(*host_output,
                aux_buffer, model.spec, be_trans, &aux_tensor);
            for (int idx = 0; idx < device_num; idx++)
            {
                device_sub_nets[idx]->output = idx + 1 == device_num ? output : nullptr;
            }
            //ret = BuildDeviceTensor_ForceDequant(tensor_list, *host_output, aux_buffer,
            //    model.spec, be_trans, TensorPartitionType::BY_COL, &aux_tensor);
            //for (int idx = 0; idx < (int)tensor_list.size(); idx++) {
            //    device_sub_nets[idx]->output = tensor_list[idx]->tensor;
            //}
        }

        if (ret && host_net.output_b != nullptr)
        {
            CudaUtil::SetDevice(*la.devices->rbegin());
            DeviceTensor *output = BuildDeviceTensor_ForceDequant(*host_net.output_b,
                aux_buffer, model.spec);
            for (int idx = 0; idx < device_num; idx++)
            {
                device_sub_nets[idx]->output_b = idx + 1 == device_num ? output : nullptr;
            }
        }
    }

    if (la.start_layer == 0)
    {
        CudaUtil::SetDevice((*la.devices)[0]);
        BuildDeviceNets_Input(device_sub_nets, model, host_net, *la.devices, aux_buffer);
    }

    tm.Progress(1);
    PtrVector<DeviceTensorBuilder> builder_list;
    int aux_tensor_size = max_tensor_size;
    for (int builder_idx = 0; builder_idx < builder_count; builder_idx++)
    {
        auto *builder = new DeviceTensorBuilder;
        builder->Init(*la.devices, aux_buffer_size, aux_tensor_size);
        builder_list.push_back(builder);
    }

    for (int layer_id = la.start_layer; layer_id < la.end_layer; layer_id++)
    {
        const auto &cpu_layer = *host_net.decoder_layers[layer_id];

        vector<StdDeviceNetwork::SubLayer*> self_attn_ptr_list, cross_attn_ptr_list, ffn_ptr_list;
        for (int device_idx = 0; device_idx < device_num; device_idx++)
        {
            auto *sub_net = device_sub_nets[device_idx];
            StdDeviceNetwork::DecoderLayer *gpu_layer_ptr = new StdDeviceNetwork::DecoderLayer;
            BuildLayerTensorMap(gpu_layer_ptr->self_attn);
            BuildLayerTensorMap(gpu_layer_ptr->cross_attn);
            BuildLayerTensorMap(gpu_layer_ptr->ffn);
            sub_net->decoder_layers.push_back(gpu_layer_ptr);

            self_attn_ptr_list.push_back(&gpu_layer_ptr->self_attn);
            cross_attn_ptr_list.push_back(&gpu_layer_ptr->cross_attn);
            ffn_ptr_list.push_back(&gpu_layer_ptr->ffn);
        }

        int builder_idx = layer_id % builder_count;
        auto *builder = builder_list[builder_idx];
        AddLayerTasks_TensorParallel(*builder, self_attn_ptr_list, cpu_layer.self_attn,
            model.spec, layer_id, be_trans, false);
        AddLayerTasks_TensorParallel(*builder, cross_attn_ptr_list, cpu_layer.cross_attn,
            model.spec, layer_id, be_trans, false);
        AddLayerTasks_TensorParallel(*builder, ffn_ptr_list, cpu_layer.ffn,
            model.spec, layer_id, be_trans, true);
    }

    //run
    for (auto *builder : builder_list) {
        builder->Create();
    }

    for (auto *builder : builder_list)
    {
        builder->Join();

        bool has_error = builder->HasError();
        if (has_error)
        {
            LogError("Error occurred in building device tensors");
            return false;
        }
    }

    tm.End();

    return ret;
}

#endif //USE_CUDA

void NetworkBuilder::BuildAttnTensorIdMap(map<LayerTensorId, int> &tensor_id_map)
{
    tensor_id_map.clear();
    tensor_id_map[LayerTensorId::ATTN_PRE_NORM] = 0;
    tensor_id_map[LayerTensorId::ATTN_PRE_NORM_B] = 0;
    tensor_id_map[LayerTensorId::ATTN_POST_NORM] = 0;
    tensor_id_map[LayerTensorId::ATTN_POST_NORM_B] = 0;
    tensor_id_map[LayerTensorId::QKV_B] = 1;
    tensor_id_map[LayerTensorId::WQ_B] = 1;
    tensor_id_map[LayerTensorId::WK_B] = 1;
    tensor_id_map[LayerTensorId::WV_B] = 1;
    tensor_id_map[LayerTensorId::WO_B] = 0;
    tensor_id_map[LayerTensorId::QKV] = 2;
    tensor_id_map[LayerTensorId::WQ] = 2;
    tensor_id_map[LayerTensorId::WK] = 2;
    tensor_id_map[LayerTensorId::WV] = 2;
    tensor_id_map[LayerTensorId::WO] = 3;
}

void NetworkBuilder::BuildFfnTensorIdMap(map<LayerTensorId, int> &tensor_id_map)
{
    tensor_id_map.clear();

    tensor_id_map[LayerTensorId::FFN_PRE_NORM] = 0;
    tensor_id_map[LayerTensorId::FFN_PRE_NORM_B] = 0;
    tensor_id_map[LayerTensorId::FFN_POST_NORM] = 0;
    tensor_id_map[LayerTensorId::FFN_POST_NORM_B] = 0;
    tensor_id_map[LayerTensorId::W1_B] = 1;
    tensor_id_map[LayerTensorId::W2_B] = 0;
    tensor_id_map[LayerTensorId::W3_B] = 1;
    tensor_id_map[LayerTensorId::W1] = 2;
    tensor_id_map[LayerTensorId::W3] = 2;
    tensor_id_map[LayerTensorId::W2] = 5;
}

void NetworkBuilder::BuildMoeTensorIdMap(map<LayerTensorId, int> &tensor_id_map)
{
    tensor_id_map.clear();

    tensor_id_map[LayerTensorId::MOE_GATE] = 0;
}

#if defined(USE_CUDA)

bool NetworkBuilder::AddLayerTasks_Std(DeviceTensorBuilder &builder,
    StdDeviceNetwork::AttentionLayer &gpu_layer,
    const StdHostNetwork::AttentionLayer &cpu_layer,
    const ModelSpec &model_spec, int layer_id, bool be_trans,
    bool is_decoder, bool is_cross_attention)
{
    bool ret = true;
    const auto &hparams = model_spec.hyper_params;
    ElementType data_type = model_spec.device_weight_data_type;
    DeviceTensorBuilder::Task task;
    task.delta_ratio = model_spec.delta_tensor_ratio;

    map<LayerTensorId, int> tensor_id_map;
    BuildAttnTensorIdMap(tensor_id_map);

    bool enable_adding_qkv = model_spec.qkv_format == 1
        && !config_ex_->is_gpu_tensor_row_major
        && hparams.decoder_heads == hparams.decoder_kv_heads;
    bool is_gpu_quant = TensorCommon::IsQuantType(data_type);
    bool add_qkv = enable_adding_qkv && is_decoder
        && !is_cross_attention && !is_gpu_quant
        && cpu_layer.qkv == nullptr && cpu_layer.wq != nullptr
        && cpu_layer.wk != nullptr && cpu_layer.wv != nullptr;

    auto iter_id = tensor_id_map.begin();
    for (; iter_id != tensor_id_map.end(); iter_id++)
    {
        LayerTensorId tensor_id = iter_id->first;
        int tensor_type = iter_id->second;

        //LogKeyInfo("layer: %d, tensor id: %d", layer_id, (int)tensor_id);
        auto iter_cpu = cpu_layer.tensor_map.find(tensor_id);
        const auto *cpu_tensor = iter_cpu == cpu_layer.tensor_map.end()
            ? nullptr : iter_cpu->second;
        if (cpu_tensor == nullptr) {
            continue;
        }

        if (tensor_id == LayerTensorId::QKV || tensor_id == LayerTensorId::WQ
            || tensor_id == LayerTensorId::WK || tensor_id == LayerTensorId::WV)
        {
            if (add_qkv) {
                continue;
            }
        }

        auto iter_gpu = gpu_layer.tensor_map.find(tensor_id);
        if (iter_gpu == gpu_layer.tensor_map.end())
        {
            LogError("Cannot find tensor %d in the current GPU layer (tensor map size: %d)",
                (int)tensor_id, (int)gpu_layer.tensor_map.size());
            return false;
        }

        auto &target_tensor = iter_gpu->second;
        task.id = 10000 * (layer_id + 1) + (int)tensor_id;
        BuildTask_Std(task, *target_tensor, *cpu_tensor, tensor_id, tensor_type,
            model_spec, be_trans);

        builder.AddTask(task);
    }

    if (add_qkv)
    {
        vector<const HostTensor*> source_tensors = { cpu_layer.wq, cpu_layer.wk, cpu_layer.wv };
        ElementType tensor_data_type = data_type;
        if (3 * (int)cpu_layer.wq->size < model_spec.tensor_quant_threshold
            && TensorCommon::IsQuantType(data_type))
        {
            tensor_data_type = ElementType::F16;
        }

        task.Set(nullptr, nullptr, tensor_data_type, false, be_trans);
        gpu_layer.qkv.tensor = context_->device_tensor_heap.New(1);
        gpu_layer.qkv.delta = context_->sparse_matrix_heap.New(1);
        task.SetSourceTarget(&gpu_layer.qkv, source_tensors);
        int heads = is_decoder ? hparams.decoder_heads : hparams.encoder_heads;
        task.cx_in_merging = hparams.embd_dims / heads;
        builder.AddTask(task);
    }

    return ret;
}

bool NetworkBuilder::AddLayerTasks_Std(DeviceTensorBuilder &builder,
    StdDeviceNetwork::FeedForwardLayer &gpu_layer,
    const StdHostNetwork::FeedForwardLayer &cpu_layer,
    const ModelSpec &model_spec, int layer_id, bool be_trans)
{
    bool ret = true;
    DeviceTensorBuilder::Task task;
    task.delta_ratio = model_spec.delta_tensor_ratio;

    map<LayerTensorId, int> tensor_id_map;
    BuildFfnTensorIdMap(tensor_id_map);

    auto iter_id = tensor_id_map.begin();
    for (; iter_id != tensor_id_map.end(); iter_id++)
    {
        LayerTensorId tensor_id = iter_id->first;
        int tensor_type = iter_id->second;

        //LogKeyInfo("layer: %d, tensor id: %d", layer_id, (int)tensor_id);
        auto iter_cpu = cpu_layer.tensor_map.find(tensor_id);
        const auto *cpu_tensor = iter_cpu == cpu_layer.tensor_map.end()
            ? nullptr : iter_cpu->second;
        if (cpu_tensor == nullptr) {
            continue;
        }

        auto iter_gpu = gpu_layer.tensor_map.find(tensor_id);
        if (iter_gpu == gpu_layer.tensor_map.end())
        {
            LogError("Cannot find tensor %d in the current GPU layer (tensor map size: %d)",
                (int)tensor_id, (int)gpu_layer.tensor_map.size());
            return false;
        }

        auto &target_tensor = iter_gpu->second;
        task.id = 10000 * (layer_id + 1) + (int)tensor_id;
        BuildTask_Std(task, *target_tensor, *cpu_tensor, tensor_id, tensor_type,
            model_spec, be_trans);

        builder.AddTask(task);
    }

    return ret;
}

void NetworkBuilder::BuildTask_Std(DeviceTensorBuilder::Task &task,
    DeviceTensorEx &target_tensor, const HostTensor &cpu_tensor,
    LayerTensorId tensor_id, int tensor_type,
    const ModelSpec &model_spec, bool be_trans)
{
    ElementType data_type = model_spec.device_weight_data_type;
    auto fine_type = model_spec.device_weight_data_types[(int)tensor_id];
    if (fine_type != ElementType::Auto) {
        data_type = fine_type;
    }

    ElementType tensor_data_type = data_type;
    if ((int)cpu_tensor.size < model_spec.tensor_quant_threshold
        && TensorCommon::IsQuantType(data_type))
    {
        tensor_data_type = ElementType::F16;
    }

    switch (tensor_type)
    {
    case 2:
    case 3:
        //parameter-4: force_dequant
        task.Set(nullptr, nullptr, tensor_data_type, false, be_trans);
        break;
    case 5:
        //task.Set(nullptr, nullptr, tensor_data_type, true, be_trans);
        if (tensor_data_type == ElementType::Q8_GL) {
            tensor_data_type = ElementType::Q8_B32T2;
        }
        //if (tensor_data_type >= ElementType::Q3_B32T1A) {
        //    tensor_data_type = ElementType::Q4_B32T1A;
        //}
        task.Set(nullptr, nullptr, tensor_data_type, false, be_trans);
        break;
    case 0:
    case 1:
    default:
        task.Set(nullptr, nullptr, tensor_data_type, true, false);
        break;
    }

    target_tensor.tensor = context_->device_tensor_heap.New(1);
    target_tensor.delta = context_->sparse_matrix_heap.New(1);
    if (tensor_type == 1) {
        target_tensor.quant_map = context_->device_tensor_heap.New(1);
    }

    task.SetSourceTarget(&target_tensor, &cpu_tensor);
}

bool NetworkBuilder::AddLayerTasks_TensorParallel(DeviceTensorBuilder &builder,
    vector<StdDeviceNetwork::SubLayer*> &gpu_sub_net_layers,
    const StdHostNetwork::SubLayer &cpu_layer,
    const ModelSpec &model_spec, int layer_id, bool be_trans, bool is_ffn)
{
    bool ret = true;
    ElementType data_type = model_spec.device_weight_data_type;
    int target_device_num = (int)gpu_sub_net_layers.size();
    DeviceTensorBuilder::Task task;
    task.delta_ratio = model_spec.delta_tensor_ratio;

    map<LayerTensorId, int> tensor_id_map;
    if (is_ffn) {
        BuildFfnTensorIdMap(tensor_id_map);
    }
    else {
        BuildAttnTensorIdMap(tensor_id_map);
    }

    vector<DeviceTensorEx*> tensor_list;
    auto iter_id = tensor_id_map.begin();
    for (; iter_id != tensor_id_map.end(); iter_id++)
    {
        LayerTensorId tensor_id = iter_id->first;
        int tensor_type = iter_id->second;

        auto iter_cpu = cpu_layer.tensor_map.find(tensor_id);
        if (iter_cpu == cpu_layer.tensor_map.end() || iter_cpu->second == nullptr) {
            continue;
        }

        const HostTensor *cpu_tensor = iter_cpu->second;

        tensor_list.clear();
        for (auto *sub_net_layer : gpu_sub_net_layers)
        {
            auto iter_gpu = sub_net_layer->tensor_map.find(tensor_id);
            if (iter_gpu == sub_net_layer->tensor_map.end())
            {
                LogError("Cannot find tensor %d in the current layer of one sub_net",
                    (int)tensor_id);
                return false;
            }

            auto &target_tensor_ex = iter_gpu->second;
            target_tensor_ex->tensor = context_->device_tensor_heap.New(1);
            target_tensor_ex->delta = context_->sparse_matrix_heap.New(1);
            if (tensor_type == 1) {
                target_tensor_ex->quant_map = context_->device_tensor_heap.New(1);
            }
            tensor_list.push_back(target_tensor_ex);
        }

        ElementType tensor_data_type = data_type;
        if ((int)cpu_tensor->size / target_device_num < model_spec.tensor_quant_threshold
            && TensorCommon::IsQuantType(data_type))
        {
            tensor_data_type = ElementType::F16;
        }

        switch (tensor_type)
        {
        case 2:
            //parameter-4: force_dequant
            task.Set(nullptr, nullptr, tensor_data_type, false, be_trans);
            break;
        case 3:
            task.Set(nullptr, nullptr, tensor_data_type, false, be_trans, TensorPartitionType::BY_ROW);
            break;
        case 5:
            //task.Set(nullptr, nullptr, tensor_data_type, true, be_trans);
            //task.Set(nullptr, nullptr, tensor_data_type, true, be_trans, TensorPartitionType::BY_ROW);
            if (tensor_data_type == ElementType::Q8_GL) {
                tensor_data_type = ElementType::Q8_B32T1;
            }
            task.Set(nullptr, nullptr, tensor_data_type, false, be_trans, TensorPartitionType::BY_ROW);
            break;
        case 1:
            task.Set(nullptr, nullptr, tensor_data_type, true, false);
            break;
        case 0:
        default:
            task.Set(nullptr, nullptr, tensor_data_type, true, false, TensorPartitionType::DUP);
            break;
        }

        task.id = 10000 * (layer_id + 1) + (int)tensor_id;
        task.SetSourceTarget(tensor_list, cpu_tensor);
        builder.AddTask(task);
    }

    return ret;
}

#endif //USE_CUDA

bool NetworkBuilder::CheckHostModel(const TransformerModel &model, bool is_cpu_only) const
{
    bool ret = true;
    const auto &hparams = model.spec.hyper_params;
    bool is_encoder_only = hparams.decoder_layers <= 0;
    bool is_decoder_only = hparams.encoder_layers <= 0;
    const auto &host_net = model.std_network.host_net;

    int vocab_size = max(hparams.vocab_size, hparams.padded_vocab_size);

    //LogKeyInfo("network_structure: %d", (int)model.spec.network_structure);
    if (!is_encoder_only)
    {
        Macro_RetxFalseIf(model.decoder_embeddings == nullptr,
            LogError("Null decoder token embeddings tensor"));
        int row_num = model.decoder_embeddings->ne[1];
        if (row_num > vocab_size)
        {
            LogError("The number of %s should NOT be larger than the vocabulary size: %d vs. %d",
                "decoder embedding rows", row_num, vocab_size);
            return false;
        }
    }

    if (!is_decoder_only)
    {
        Macro_RetxFalseIf(model.encoder_embeddings == nullptr,
            LogError("Null encoder token embeddings tensor"));
        int row_num = model.encoder_embeddings->ne[1];
        if (row_num > vocab_size)
        {
            LogError("The number of %s should NOT be larger than the vocabulary size: %d vs. %d",
                "encoder embedding rows", row_num, vocab_size);
            return false;
        }
    }

    bool is_encoder = true;
    for (int layer_id = 0; ret && layer_id < (int)host_net.encoder_layers.size(); layer_id++)
    {
        const auto &layer = *host_net.encoder_layers[layer_id];

        ret = ret && CheckModelLayer(layer.self_attn, layer_id, is_encoder, true);
        ret = ret && CheckModelLayer(layer.ffn, layer_id, is_encoder, FfnLayerType::Std);
        Macro_RetFalseIf(!ret);
    }

    if (!is_decoder_only)
    {
        //bool is_null_output_norm = host_net.encoder_output_norm == nullptr
        //    && host_net.encoder_output_post_norm == nullptr;
        //Macro_RetxFalseIf(is_null_output_norm, LogError("Null encoder output_norm and output_post_norm"));
    }

    if (!is_encoder_only)
    {
        if (host_net.decoder_output_norm == nullptr) {
            LogWarning("Null decoder output_norm");
        }
    }

    is_encoder = false;
    int decoder_cpu_layer_count = is_cpu_only ? hparams.decoder_layers
        : model.spec.decoder_cpu_layer_count;
    int end_layer = model.spec.is_eager_device_building
        ? min(decoder_cpu_layer_count, hparams.decoder_layers)
        : hparams.decoder_layers;
    for (int layer_id = 0; ret && layer_id < end_layer; layer_id++)
    {
        const auto &layer = *host_net.decoder_layers[layer_id];

        ret = ret && CheckModelLayer(layer.self_attn, layer_id, is_encoder, true);
        if (!is_decoder_only) {
            ret = ret && CheckModelLayer(layer.cross_attn, layer_id, is_encoder, false);
        }
        ret = ret && CheckModelLayer(layer.ffn, layer_id, is_encoder, FfnLayerType::Std);
        Macro_RetFalseIf(!ret);
    }

    if (model.spec.network_structure == NetworkType::LLAMA)
    {
        //LogKeyInfo("host_net.norm == nullptr? %s", host_net.norm == nullptr ? "Y" : "N");
        //Macro_RetxFalseIf(host_net.input_norm == nullptr, LogError("Null input_norm tensor"));
        Macro_RetxFalseIf(host_net.output == nullptr, LogError("Null output tensor"));
    }
    else if (model.spec.network_structure == NetworkType::BLOOM)
    {
        Macro_RetxFalseIf(host_net.decoder_input_norm == nullptr, LogError("Null input_norm tensor"));
        Macro_RetxFalseIf(host_net.decoder_input_norm_b == nullptr, LogError("Null input_norm_b tensor"));
        Macro_RetxFalseIf(host_net.decoder_output_norm_b == nullptr, LogError("Null output_norm_b tensor"));
    }

    return true;
}

#if defined(USE_CUDA)

bool NetworkBuilder::CheckDeviceModel(const TransformerModel &model) const
{
    bool ret = true;
    bool is_by_layer = model.spec.multi_gpu_strategy == MultiGpuStrategy::BY_LAYER;
    if (is_by_layer)
    {
        ret = CheckDeviceModel(model.std_network.device_net, model.spec, -1);
    }
    else
    {
        int sub_net_num = (int)model.std_network.device_sub_nets.size();
        for (int sub_idx = 0; ret && sub_idx < sub_net_num; sub_idx++)
        {
            const auto *sub_net_ptr = model.std_network.device_sub_nets[sub_idx];
            ret = CheckDeviceModel(*sub_net_ptr, model.spec, sub_idx);
        }
    }

    return ret;
}

bool NetworkBuilder::CheckDeviceModel(const StdDeviceNetwork &net,
    const ModelSpec &model_spec, int sub_net_idx) const
{
    bool ret = true;
    const auto &hparams = model_spec.hyper_params;
    bool is_encoder_only = hparams.decoder_layers <= 0;
    bool is_decoder_only = hparams.encoder_layers <= 0;

    bool is_encoder = true;
    int layer_id = model_spec.encoder_cpu_layer_count;
    for (; layer_id < (int)net.encoder_layers.size(); layer_id++)
    {
        const auto &layer = *net.encoder_layers[layer_id];

        ret = ret && CheckModelLayer(layer.self_attn, layer_id, is_encoder, true);
        ret = ret && CheckModelLayer(layer.ffn, layer_id, is_encoder, FfnLayerType::Std);
        Macro_RetFalseIf(!ret);
    }

    if (!is_decoder_only)
    {
        //bool is_null_output_norm = net.encoder_output_norm == nullptr
        //    && net.encoder_output_post_norm == nullptr;
        //Macro_RetxFalseIf(is_null_output_norm, LogError("Null encoder output_norm and output_post_norm"));
    }

    if (!is_encoder_only && sub_net_idx < 0)
    {
        if (net.decoder_output_norm == nullptr) {
            LogWarning("Null decoder output_norm");
        }
    }

    is_encoder = false;
    int start_layer = model_spec.decoder_cpu_layer_count;
    int end_layer = (int)net.decoder_layers.size();
    for (layer_id = start_layer; layer_id < end_layer; layer_id++)
    {
        const auto &layer = *net.decoder_layers[layer_id];

        ret = ret && CheckModelLayer(layer.self_attn, layer_id, is_encoder, true);
        if (!is_decoder_only) {
            ret = ret && CheckModelLayer(layer.cross_attn, layer_id, is_encoder, false);
        }

        int moe_layer_end = model_spec.hyper_params.moe_layer_end < 0 ? INT32_MAX
            : model_spec.hyper_params.moe_layer_end;
        bool is_moe = model_spec.hyper_params.experts > 0
            && layer_id >= model_spec.hyper_params.moe_layer_start
            && layer_id < moe_layer_end;
        if (!is_moe)
        {
            ret = ret && CheckModelLayer(layer.ffn, layer_id, is_encoder, FfnLayerType::Std);
        }
        else //MOE
        {
            if (layer.moe.gate.tensor == nullptr) {
                LogError("Null MOE gate");
                return false;
            }

            if (hparams.has_shared_expert)
            {
                ret = CheckModelLayer(layer.moe.shared_expert, layer_id, is_encoder,
                    FfnLayerType::SharedExpert);
            }

            for (int expert_id = 0; expert_id < hparams.in_use_experts; expert_id++)
            {
                const auto *expert_ptr = layer.moe.experts[expert_id];
                ret = ret && CheckModelLayer(*expert_ptr, layer_id, is_encoder,
                    FfnLayerType::Expert, expert_id);
            }
        }

        Macro_RetFalseIf(!ret);
    }

    return true;
}

//static
int NetworkBuilder::GetDeviceByLayer(const ModelPartition &mp, int layer_id, bool is_decoder)
{
    const auto &assignments = is_decoder ? mp.decoder_assignments : mp.encoder_assignments;
    for (const auto &la : assignments)
    {
        if (layer_id >= la.start_layer && layer_id < la.end_layer)
        {
            return (*la.devices)[0];
        }
    }

    return -1;
}

//static
int NetworkBuilder::GetDeviceGroupIndex(const ModelPartition &mp, int layer_id, bool is_decoder)
{
    const auto &assignments = is_decoder ? mp.decoder_assignments : mp.encoder_assignments;
    for (int la_idx = 0; la_idx < (int)assignments.size(); la_idx++)
    {
        const auto &la = assignments[la_idx];
        if (layer_id >= la.start_layer && layer_id < la.end_layer)
        {
            return la_idx;
        }
    }

    return 0;
}

#endif //USE_CUDA

bool NetworkBuilder::CheckModelLayer(const StdHostNetwork::AttentionLayer &layer,
    int layer_id, bool is_encoder, bool is_self_attn) const
{
    char buf[128];
    sprintf(buf, "%s layer %d of the %s", is_self_attn ? "self_attn" : "cross_attn",
        layer_id, is_encoder ? "encoder" : "decoder");

    if (layer.pre_norm == nullptr && layer.post_norm == nullptr && layer_id == 0) {
        LogWeakWarning("Null pre_norm and post_norm in %s.", buf);
    }

    bool is_qkv_null = layer.qkv == nullptr;
    Macro_RetxFalseIf(is_qkv_null && layer.wq == nullptr,
        LogError("Null qkv and wq tensor in %s.", buf));
    Macro_RetxFalseIf(is_qkv_null && layer.wk == nullptr,
        LogError("Null qkv and wk tensor in %s.", buf));
    Macro_RetxFalseIf(is_qkv_null && layer.wv == nullptr,
        LogError("Null qkv and wv tensor in %s.", buf));

    Macro_RetxFalseIf(layer.wo == nullptr, LogError("Null wo tensor in %s.", buf));
    return true;
}

bool NetworkBuilder::CheckModelLayer(const StdHostNetwork::FeedForwardLayer &layer,
    int layer_id, bool is_encoder, FfnLayerType layer_type, int expert_id) const
{
    char buf[64];
    if (layer_type == FfnLayerType::Expert) {
        sprintf(buf, "expert %d of layer %d of the %s", expert_id, layer_id, is_encoder ? "encoder" : "decoder");
    }
    else if (layer_type == FfnLayerType::SharedExpert) {
        sprintf(buf, "shared-expert of layer %d of the %s", layer_id, is_encoder ? "encoder" : "decoder");
    }
    else {
        sprintf(buf, "ffn layer %d of the %s", layer_id, is_encoder ? "encoder" : "decoder");
    }

    bool is_null_pre_and_post = layer.pre_norm == nullptr && layer.post_norm == nullptr;
    if (is_null_pre_and_post && layer_id == 0) {
        LogWarning("Null pre_norm and post_norm in %s.", buf);
    }

    Macro_RetxFalseIf(layer.w1 == nullptr, LogError("Null w1 tensor in %s.", buf));
    Macro_RetxFalseIf(layer.w2 == nullptr, LogError("Null w2 tensor in %s.", buf));
    return true;
}

#if defined(USE_CUDA)

bool NetworkBuilder::CheckModelLayer(const StdDeviceNetwork::AttentionLayer &layer,
    int layer_id, bool is_encoder, bool is_self_attn) const
{
    char buf[128];
    sprintf(buf, "%s layer %d of the %s", is_self_attn ? "self_attn" : "cross_attn",
        layer_id, is_encoder ? "encoder" : "decoder");

    if (layer.pre_norm.tensor == nullptr && layer.post_norm.tensor == nullptr
        && layer_id == 0)
    {
        LogWeakWarning("Null pre_norm and post_norm in %s.", buf);
    }

    bool is_qkv_null = layer.qkv.tensor == nullptr;
    Macro_RetxFalseIf(is_qkv_null && layer.wq.tensor == nullptr,
        LogError("Null qkv and wq tensor in %s.", buf));
    Macro_RetxFalseIf(is_qkv_null && layer.wk.tensor == nullptr,
        LogError("Null qkv and wk tensor in %s.", buf));
    Macro_RetxFalseIf(is_qkv_null && layer.wv.tensor == nullptr,
        LogError("Null qkv and wv tensor in %s.", buf));

    Macro_RetxFalseIf(layer.wo.tensor == nullptr,
        LogError("Null wo tensor in %s.", buf));
    return true;
}

bool NetworkBuilder::CheckModelLayer(const StdDeviceNetwork::FeedForwardLayer &layer,
    int layer_id, bool is_encoder, FfnLayerType layer_type, int expert_id) const
{
    char buf[64];
    if (layer_type == FfnLayerType::Expert) {
        sprintf(buf, "expert %d of layer %d of the %s", expert_id, layer_id, is_encoder ? "encoder" : "decoder");
    }
    else if (layer_type == FfnLayerType::SharedExpert) {
        sprintf(buf, "shared-expert of layer %d of the %s", layer_id, is_encoder ? "encoder" : "decoder");
    }
    else {
        sprintf(buf, "ffn layer %d of the %s", layer_id, is_encoder ? "encoder" : "decoder");
    }

    bool is_null_pre_and_post = layer.pre_norm.tensor == nullptr
        && layer.post_norm.tensor == nullptr;
    if (is_null_pre_and_post && layer_id == 0) {
        LogWarning("Null pre_norm and post_norm in %s.", buf);
    }

    Macro_RetxFalseIf(layer.w1.tensor == nullptr, LogError("Null w1 tensor in %s.", buf));
    Macro_RetxFalseIf(layer.w2.tensor == nullptr, LogError("Null w2 tensor in %s.", buf));
    return true;
}

#endif //USE_CUDA

#if defined(USE_CUDA)

DeviceTensor* NetworkBuilder::BuildDeviceTensor_Quant(
    const HostTensor &cpu_tensor, HostHalfBuffer &aux_buffer,
    const ModelSpec &model_spec, bool be_trans, DeviceTensor *aux_tensor)
{
    DeviceTensor *device_tensor = context_->device_tensor_heap.New(1);

    DeviceTensorEx target;
    target.tensor = device_tensor;
    bool force_dequant = false;
    bool ret = DeviceTensorBuilder::Build(target, cpu_tensor, aux_buffer,
        model_spec.device_weight_data_type, force_dequant, be_trans, aux_tensor);

    if (!ret)
    {
        device_tensor->Clear();
        device_tensor = nullptr;
        exit(2);
    }

    return device_tensor;
}

DeviceTensor* NetworkBuilder::BuildDeviceTensor_ForceDequant(
    const HostTensor &cpu_tensor, HostHalfBuffer &aux_buffer,
    const ModelSpec &model_spec, bool be_trans, DeviceTensor *aux_tensor)
{
    DeviceTensor *device_tensor = context_->device_tensor_heap.New(1);

    DeviceTensorEx target;
    target.tensor = device_tensor;
    bool force_dequant = true;
    bool ret = DeviceTensorBuilder::Build(target, cpu_tensor, aux_buffer,
        model_spec.device_weight_data_type, force_dequant, be_trans, aux_tensor);

    if (!ret)
    {
        device_tensor->Clear();
        device_tensor = nullptr;
        exit(2);
    }

    return device_tensor;
}

bool NetworkBuilder::BuildDeviceTensors_ForceDequant(PtrVector<DeviceTensorEx> &tensor_list,
    const HostTensor &cpu_tensor, HostHalfBuffer &aux_buffer, const ModelSpec &model_spec,
    const vector<int> &device_group, bool be_trans, TensorPartitionType partition_type,
    DeviceTensor *aux_tensor)
{
    tensor_list.Clear(true);
    int device_num = (int)device_group.size();
    for (int idx = 0; idx < device_num; idx++)
    {
        DeviceTensorEx *tensor_ex = new DeviceTensorEx;
        tensor_ex->tensor = context_->device_tensor_heap.New(1);
        tensor_list.push_back(tensor_ex);
    }

    bool force_dequant = true;
    bool ret = DeviceTensorBuilder::Build(tensor_list, cpu_tensor, aux_buffer,
        device_group, model_spec.device_weight_data_type, force_dequant,
        be_trans, partition_type, aux_tensor);
    return ret;
}

#endif //USE_CUDA

//static
void NetworkBuilder::SplitGpuLayers(vector<LayerAssignment> &layer_assignments,
    int start_layer, int end_layer, const ModelSpec &spec)
{
    layer_assignments.clear();
    int device_group_num = (int)spec.device_groups.size();
    if (device_group_num <= 0) {
        return;
    }

    int worker_group_num = device_group_num;
    int layer_num = end_layer - start_layer;
    int layer_num_per_group = (layer_num + worker_group_num - 1) / worker_group_num;
    for (int group_id = 0; group_id < worker_group_num; group_id++)
    {
        LayerAssignment la;
        la.devices = &spec.device_groups[group_id];

        la.start_layer = start_layer + group_id * layer_num_per_group;
        la.end_layer = group_id + 1 == worker_group_num ? end_layer
            : (group_id + 1) * layer_num_per_group;
        if (la.end_layer > la.start_layer) {
            layer_assignments.push_back(la);
        }
    }
}

//static
void NetworkBuilder::GetDeviceAssignments(vector<LayerAssignment> &layer_assignments,
    const ModelSpec &model_spec, bool is_encoder, int start_gpu_layer)
{
    layer_assignments.clear();

    const auto &hparams = model_spec.hyper_params;
    if (is_encoder && hparams.encoder_layers > 0)
    {
        int layer_num = hparams.encoder_layers;
        //int end_gpu_layer = min(layer_num, model_spec.gpu_encoder_layer_count);
        int end_gpu_layer = layer_num;
        NetworkBuilder::SplitGpuLayers(layer_assignments,
            start_gpu_layer, end_gpu_layer, model_spec);
    }

    if (!is_encoder && hparams.decoder_layers > 0)
    {
        int layer_num = hparams.decoder_layers;
        //int end_gpu_layer = min(layer_num, model_spec.gpu_decoder_layer_count);
        int end_gpu_layer = layer_num;
        NetworkBuilder::SplitGpuLayers(layer_assignments,
            start_gpu_layer, end_gpu_layer, model_spec);
    }
}

//static
void NetworkBuilder::BuildHostTensorMap(StdHostNetwork &net)
{
    for (auto *layer : net.encoder_layers)
    {
        BuildLayerTensorMap(layer->self_attn);
        BuildLayerTensorMap(layer->ffn);
    }

    for (auto *layer : net.decoder_layers)
    {
        BuildLayerTensorMap(layer->self_attn);
        BuildLayerTensorMap(layer->cross_attn);
        BuildLayerTensorMap(layer->ffn);
    }
}

//static
void NetworkBuilder::BuildLayerTensorMap(StdHostNetwork::AttentionLayer &layer)
{
    auto &tensor_map = layer.tensor_map;
    tensor_map[LayerTensorId::ATTN_PRE_NORM] = layer.pre_norm;
    tensor_map[LayerTensorId::ATTN_PRE_NORM_B] = layer.pre_norm_b;
    tensor_map[LayerTensorId::ATTN_POST_NORM] = layer.post_norm;
    tensor_map[LayerTensorId::ATTN_POST_NORM_B] = layer.post_norm_b;

    tensor_map[LayerTensorId::QKV] = layer.qkv;
    tensor_map[LayerTensorId::QKV_B] = layer.qkv_b;
    tensor_map[LayerTensorId::WQ] = layer.wq;
    tensor_map[LayerTensorId::WQ_B] = layer.wq_b;
    tensor_map[LayerTensorId::WK] = layer.wk;
    tensor_map[LayerTensorId::WK_B] = layer.wk_b;
    tensor_map[LayerTensorId::WV] = layer.wv;
    tensor_map[LayerTensorId::WV_B] = layer.wv_b;
    tensor_map[LayerTensorId::WO] = layer.wo;
    tensor_map[LayerTensorId::WO_B] = layer.wo_b;
}

//static
void NetworkBuilder::BuildLayerTensorMap(StdHostNetwork::FeedForwardLayer &layer)
{
    auto &tensor_map = layer.tensor_map;
    tensor_map[LayerTensorId::FFN_PRE_NORM] = layer.pre_norm;
    tensor_map[LayerTensorId::FFN_PRE_NORM_B] = layer.pre_norm_b;
    tensor_map[LayerTensorId::FFN_POST_NORM] = layer.post_norm;
    tensor_map[LayerTensorId::FFN_POST_NORM_B] = layer.post_norm_b;

    tensor_map[LayerTensorId::W1] = layer.w1;
    tensor_map[LayerTensorId::W1_B] = layer.w1_b;
    tensor_map[LayerTensorId::W2] = layer.w2;
    tensor_map[LayerTensorId::W2_B] = layer.w2_b;
    tensor_map[LayerTensorId::W3] = layer.w3;
    tensor_map[LayerTensorId::W3_B] = layer.w3_b;
    tensor_map[LayerTensorId::W1N3] = layer.w3;
    tensor_map[LayerTensorId::W1N3_B] = layer.w3_b;
}

#if defined(USE_CUDA)

//static
void NetworkBuilder::BuildLayerTensorMap(StdDeviceNetwork::AttentionLayer &layer)
{
    auto &tensor_map = layer.tensor_map;
    tensor_map[LayerTensorId::ATTN_PRE_NORM] = &layer.pre_norm;
    tensor_map[LayerTensorId::ATTN_PRE_NORM_B] = &layer.pre_norm_b;
    tensor_map[LayerTensorId::ATTN_POST_NORM] = &layer.post_norm;
    tensor_map[LayerTensorId::ATTN_POST_NORM_B] = &layer.post_norm_b;

    tensor_map[LayerTensorId::QKV] = &layer.qkv;
    tensor_map[LayerTensorId::QKV_B] = &layer.qkv_b;
    tensor_map[LayerTensorId::WQ] = &layer.wq;
    tensor_map[LayerTensorId::WQ_B] = &layer.wq_b;
    tensor_map[LayerTensorId::WK] = &layer.wk;
    tensor_map[LayerTensorId::WK_B] = &layer.wk_b;
    tensor_map[LayerTensorId::WV] = &layer.wv;
    tensor_map[LayerTensorId::WV_B] = &layer.wv_b;
    tensor_map[LayerTensorId::WO] = &layer.wo;
    tensor_map[LayerTensorId::WO_B] = &layer.wo_b;
}

//static
void NetworkBuilder::BuildLayerTensorMap(StdDeviceNetwork::FeedForwardLayer &layer)
{
    auto &tensor_map = layer.tensor_map;
    tensor_map[LayerTensorId::FFN_PRE_NORM] = &layer.pre_norm;
    tensor_map[LayerTensorId::FFN_PRE_NORM_B] = &layer.pre_norm_b;
    tensor_map[LayerTensorId::FFN_POST_NORM] = &layer.post_norm;
    tensor_map[LayerTensorId::FFN_POST_NORM_B] = &layer.post_norm_b;

    tensor_map[LayerTensorId::W1] = &layer.w1;
    tensor_map[LayerTensorId::W1_B] = &layer.w1_b;
    tensor_map[LayerTensorId::W2] = &layer.w2;
    tensor_map[LayerTensorId::W2_B] = &layer.w2_b;
    tensor_map[LayerTensorId::W3] = &layer.w3;
    tensor_map[LayerTensorId::W3_B] = &layer.w3_b;
    tensor_map[LayerTensorId::W1N3] = &layer.w3;
    tensor_map[LayerTensorId::W1N3_B] = &layer.w3_b;
}

//static
void NetworkBuilder::BuildLayerTensorMap(StdDeviceNetwork::FfnMoeLayer &layer)
{
    auto &tensor_map = layer.tensor_map;
    tensor_map[LayerTensorId::MOE_GATE] = &layer.gate;
    tensor_map[LayerTensorId::MOE_GATE_B] = &layer.gate_b;

    BuildLayerTensorMap(layer.shared_expert);
}

#endif //USE_CUDA

//static
bool  NetworkBuilder::ConvertGgmlToHost(HostTensor &host_tensor, const ggml_tensor *ggml_tensor)
{
    if (ggml_tensor == nullptr) {
        return false;
    }

    ElementType data_type = TensorUtil::ToElementType(ggml_tensor->type);
    host_tensor.Set(data_type, ggml_tensor->data, (int)ggml_tensor->ne[0],
        (int)ggml_tensor->ne[1], (int)ggml_tensor->ne[2], false);
    return true;
}

ggml_tensor* NetworkBuilder::ConvertHostToGgml(const HostTensor *host_tensor, ggml_context* &ctx)
{
    if (host_tensor == nullptr) {
        return nullptr;
    }
    ggml_type data_type_ggml = TensorUtil::ToGgmlType(host_tensor->data_type);

    ggml_tensor *new_tensor = nullptr;
    switch (host_tensor->dim) 
    {
        case 1:
            new_tensor = ggml_new_tensor_1d(ctx, data_type_ggml, host_tensor->ne[0]);
            break;
        case 2:
            new_tensor = ggml_new_tensor_2d(ctx, data_type_ggml, host_tensor->ne[0], host_tensor->ne[1]);
            break;
        case 3:
            new_tensor = ggml_new_tensor_3d(ctx, data_type_ggml, host_tensor->ne[0], host_tensor->ne[1], host_tensor->ne[2]);
            break;
        default:
            LogError("host tensor dim not right");
            return nullptr;
    }
    new_tensor->data = (void *)host_tensor->data;
    return new_tensor;
}

bool NetworkBuilder::BuildGgmlNetwork_EncoderLayer(StdGgmlNetwork::EncoderLayer *ggml_layer,
    const StdHostNetwork::EncoderLayer *host_layer, ggml_context * &ctx, const ModelSpec &spec)
{
    (void)spec;
    // attn
    auto &ggml_self_attn = ggml_layer->self_attn;
    auto &host_self_attn = host_layer->self_attn;

    ggml_self_attn.pre_norm = ConvertHostToGgml(host_self_attn.pre_norm, ctx);
    ggml_self_attn.pre_norm_b = ConvertHostToGgml(host_self_attn.pre_norm_b, ctx);
    ggml_self_attn.post_norm = ConvertHostToGgml(host_self_attn.post_norm, ctx);
    ggml_self_attn.post_norm_b = ConvertHostToGgml(host_self_attn.post_norm_b, ctx);

    ggml_self_attn.qkv = ConvertHostToGgml(host_self_attn.qkv, ctx);
    ggml_self_attn.qkv_b = ConvertHostToGgml(host_self_attn.qkv_b, ctx);
    ggml_self_attn.wq = ConvertHostToGgml(host_self_attn.wq, ctx);
    ggml_self_attn.wq_b = ConvertHostToGgml(host_self_attn.wq_b, ctx);
    ggml_self_attn.wk = ConvertHostToGgml(host_self_attn.wk, ctx);
    ggml_self_attn.wk_b = ConvertHostToGgml(host_self_attn.wk_b, ctx);
    ggml_self_attn.wv = ConvertHostToGgml(host_self_attn.wv, ctx);
    ggml_self_attn.wv_b = ConvertHostToGgml(host_self_attn.wv_b, ctx);
    ggml_self_attn.wo = ConvertHostToGgml(host_self_attn.wo, ctx);
    ggml_self_attn.wo_b = ConvertHostToGgml(host_self_attn.wo_b, ctx);

    // feed forward
    auto &ggml_ffn = ggml_layer->ffn;
    auto &host_ffn = host_layer->ffn;

    ggml_ffn.pre_norm = ConvertHostToGgml(host_ffn.pre_norm, ctx);
    ggml_ffn.pre_norm_b = ConvertHostToGgml(host_ffn.pre_norm_b, ctx);
    ggml_ffn.post_norm = ConvertHostToGgml(host_ffn.post_norm, ctx);
    ggml_ffn.post_norm_b = ConvertHostToGgml(host_ffn.post_norm_b, ctx);
    ggml_ffn.w1 = ConvertHostToGgml(host_ffn.w1, ctx);
    ggml_ffn.w1_b = ConvertHostToGgml(host_ffn.w1_b, ctx);
    ggml_ffn.w2 = ConvertHostToGgml(host_ffn.w2, ctx);
    ggml_ffn.w2_b = ConvertHostToGgml(host_ffn.w2_b, ctx);
    ggml_ffn.w3 = ConvertHostToGgml(host_ffn.w3, ctx);
    ggml_ffn.w3_b = ConvertHostToGgml(host_ffn.w3_b, ctx);
    ggml_ffn.w1n3 = ConvertHostToGgml(host_ffn.w1n3, ctx);
    ggml_ffn.w1n3_b = ConvertHostToGgml(host_ffn.w1n3_b, ctx);

    return true;
}


bool NetworkBuilder::BuildGgmlNetwork_DecoderLayer(StdGgmlNetwork::DecoderLayer *ggml_layer,
    const StdHostNetwork::DecoderLayer *host_layer, ggml_context* &ctx, const ModelSpec &spec)
{
    BuildGgmlNetwork_EncoderLayer(ggml_layer, host_layer, ctx, spec);

    // cross attn
    auto &ggml_cross_attn = ggml_layer->cross_attn;
    auto &host_cross_attn = host_layer->cross_attn;

    ggml_cross_attn.pre_norm = ConvertHostToGgml(host_cross_attn.pre_norm, ctx);
    ggml_cross_attn.pre_norm_b = ConvertHostToGgml(host_cross_attn.pre_norm_b, ctx);
    ggml_cross_attn.post_norm = ConvertHostToGgml(host_cross_attn.post_norm, ctx);
    ggml_cross_attn.post_norm_b = ConvertHostToGgml(host_cross_attn.post_norm_b, ctx);

    ggml_cross_attn.qkv = ConvertHostToGgml(host_cross_attn.qkv, ctx);
    ggml_cross_attn.qkv_b = ConvertHostToGgml(host_cross_attn.qkv_b, ctx);
    ggml_cross_attn.wq = ConvertHostToGgml(host_cross_attn.wq, ctx);
    ggml_cross_attn.wq_b = ConvertHostToGgml(host_cross_attn.wq_b, ctx);
    ggml_cross_attn.wk = ConvertHostToGgml(host_cross_attn.wk, ctx);
    ggml_cross_attn.wk_b = ConvertHostToGgml(host_cross_attn.wk_b, ctx);
    ggml_cross_attn.wv = ConvertHostToGgml(host_cross_attn.wv, ctx);
    ggml_cross_attn.wv_b = ConvertHostToGgml(host_cross_attn.wv_b, ctx);
    ggml_cross_attn.wo = ConvertHostToGgml(host_cross_attn.wo, ctx);
    ggml_cross_attn.wo_b = ConvertHostToGgml(host_cross_attn.wo_b, ctx);
    return true;
}

bool NetworkBuilder::BuildGgmlNetwork_SimpleLayer(StdGgmlNetwork::SimpleLayer &ggml_layer,
    const StdHostNetwork::SimpleLayer &host_layer, ggml_context* &ctx)
{
    BuildGgmlNetwork_AtomicLayer(ggml_layer.dense, host_layer.dense, ctx);
    BuildGgmlNetwork_AtomicLayer(ggml_layer.pre_norm, host_layer.pre_norm, ctx);
    BuildGgmlNetwork_AtomicLayer(ggml_layer.post_norm, host_layer.post_norm, ctx);
    return true;
}

bool NetworkBuilder::BuildGgmlNetwork_AtomicLayer(StdGgmlNetwork::AtomicLayer &ggml_layer,
    const StdHostNetwork::AtomicLayer &host_layer,  ggml_context* &ctx)
{
    ggml_layer.weight = ConvertHostToGgml(host_layer.weight, ctx);
    ggml_layer.bias = ConvertHostToGgml(host_layer.bias, ctx);

    return true;
}

TRANSFORMER_END
INFER_FLOW_END
