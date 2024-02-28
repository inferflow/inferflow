#include "model.h"
#include <algorithm>
#include "tensor/tensor_util.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// class StdDeviceNetwork

#if defined(USE_CUDA)

int StdDeviceNetwork::MaxTensorSize() const
{
    int max_tensor_size = 0;
    if (!this->encoder_layers.empty()) {
        max_tensor_size = MaxTensorSize(*encoder_layers[0]);
    }
    if (!this->decoder_layers.empty())
    {
        int layer_num = (int)this->decoder_layers.size();
        for (int layer_id = 0; layer_id < layer_num; layer_id++)
        {
            const auto *layer = decoder_layers[layer_id];
            if (layer != nullptr)
            {
                int max_decoder_tensor_size = MaxTensorSize(*layer);
                max_tensor_size = max(max_tensor_size, max_decoder_tensor_size);
            }
        }
    }

    return max_tensor_size;
}

//static
int StdDeviceNetwork::MaxTensorSize(const EncoderLayer &layer)
{
    int m1 = MaxTensorSize(layer.self_attn);
    int m2 = MaxTensorSize(layer.ffn);
    int m3 = MaxTensorSize(layer.moe);
    return max(m1, max(m2, m3));
}

//static
int StdDeviceNetwork::MaxTensorSize(const DecoderLayer &layer)
{
    int m1 = MaxTensorSize(layer.self_attn);
    int m2 = MaxTensorSize(layer.cross_attn);
    int m3 = MaxTensorSize(layer.ffn);
    int m4 = MaxTensorSize(layer.moe);
    return max(max(m1, m2), max(m3, m4));
}

//static
int StdDeviceNetwork::MaxTensorSize(const AttentionLayer &layer)
{
    const DeviceTensor *tensor_array[] =
    {
        layer.pre_norm.tensor,
        layer.qkv.tensor, layer.wq.tensor, layer.wo.tensor
    };

    int max_size = 0;
    int tensor_num = sizeof(tensor_array) / sizeof(tensor_array[0]);
    for (int idx = 0; idx < tensor_num; idx++)
    {
        if (tensor_array[idx] == nullptr) {
            continue;
        }

        int size = (int)tensor_array[idx]->size;
        if (max_size < size) {
            max_size = size;
        }
    }

    return max_size;
}

//static
int StdDeviceNetwork::MaxTensorSize(const FeedForwardLayer &layer)
{
    const DeviceTensor *tensor_array[] =
    {
        layer.pre_norm.tensor, layer.post_norm.tensor,
        layer.w1.tensor, layer.w2.tensor, layer.w3.tensor, layer.w1n3.tensor
    };

    int max_size = 0;
    int tensor_num = sizeof(tensor_array) / sizeof(tensor_array[0]);
    for (int idx = 0; idx < tensor_num; idx++)
    {
        if (tensor_array[idx] == nullptr) {
            continue;
        }

        int size = (int)tensor_array[idx]->size;
        if (max_size < size) {
            max_size = size;
        }
    }

    return max_size;
}

//static
int StdDeviceNetwork::MaxTensorSize(const FfnMoeLayer &layer)
{
    const DeviceTensor *tensor_array[] =
    {
        layer.gate.tensor
    };

    int max_size = 0;
    int tensor_num = sizeof(tensor_array) / sizeof(tensor_array[0]);
    for (int idx = 0; idx < tensor_num; idx++)
    {
        if (tensor_array[idx] == nullptr) {
            continue;
        }

        int size = (int)tensor_array[idx]->size;
        if (max_size < size) {
            max_size = size;
        }
    }

    for (const auto *expert_ptr : layer.experts)
    {
        if (expert_ptr != nullptr)
        {
            int size = MaxTensorSize(*expert_ptr);
            if (max_size < size) {
                max_size = size;
            }
        }
    }

    return max_size;
}

#endif //USE_CUDA

////////////////////////////////////////////////////////////////////////////////
// class StdHostNetwork

int StdHostNetwork::MaxTensorSize() const
{
    int max_tensor_size = 0;
    if (!this->encoder_layers.empty()) {
        max_tensor_size = MaxTensorSize(*encoder_layers[0]);
    }
    if (!this->decoder_layers.empty())
    {
        int max_decoder_tensor_size = MaxTensorSize(*decoder_layers[0]);
        max_tensor_size = max(max_tensor_size, max_decoder_tensor_size);
    }

    return max_tensor_size;
}

//static
int StdHostNetwork::MaxTensorSize(const EncoderLayer &layer)
{
    int m1 = MaxTensorSize(layer.self_attn);
    int m2 = MaxTensorSize(layer.ffn);
    return max(m1, m2);
}

//static
int StdHostNetwork::MaxTensorSize(const DecoderLayer &layer)
{
    int m1 = MaxTensorSize(layer.self_attn);
    int m2 = MaxTensorSize(layer.cross_attn);
    int m3 = MaxTensorSize(layer.ffn);
    return max(m1, max(m2, m3));
}

//static
int StdHostNetwork::MaxTensorSize(const AttentionLayer &layer)
{
    const HostTensor *tensor_array[] =
    {
        layer.pre_norm,
        layer.qkv, layer.wq, layer.wk, layer.wv, layer.wo
    };

    int max_size = 0;
    int tensor_num = sizeof(tensor_array) / sizeof(tensor_array[0]);
    for (int idx = 0; idx < tensor_num; idx++)
    {
        if (tensor_array[idx] == nullptr) {
            continue;
        }

        int size = (int)TensorUtil::ElementCount(*tensor_array[idx]);
        if (max_size < size) {
            max_size = size;
        }
    }

    return max_size;
}

//static
int StdHostNetwork::MaxTensorSize(const FeedForwardLayer &layer)
{
    const HostTensor *tensor_array[] =
    {
        layer.pre_norm, layer.post_norm,
        layer.w1, layer.w2, layer.w3, layer.w1n3
    };

    int max_size = 0;
    int tensor_num = sizeof(tensor_array) / sizeof(tensor_array[0]);
    for (int idx = 0; idx < tensor_num; idx++)
    {
        if (tensor_array[idx] == nullptr) {
            continue;
        }

        int size = (int)TensorUtil::ElementCount(*tensor_array[idx]);
        if (max_size < size) {
            max_size = size;
        }
    }

    return max_size;
}

////////////////////////////////////////////////////////////////////////////////
// class StdDeviceNetwork

#if defined(USE_CUDA)

void StdDeviceNetwork::CalculateStat(NetworkStat &stat) const
{
    vector<const DeviceTensor*> tensor_list;

    stat.encoder_layer_num = (int)this->encoder_layers.size();
    memset(stat.encoder_layer_size, 0, sizeof(stat.encoder_layer_size));
    if (!this->encoder_layers.empty())
    {
        const auto *layer0 = this->encoder_layers[0];
        auto attn_iter = layer0->self_attn.tensor_map.begin();
        for (; attn_iter != layer0->self_attn.tensor_map.end(); attn_iter++)
        {
            const auto *tensor = attn_iter->second->tensor;
            if (tensor != nullptr)
            {
                stat.encoder_layer_size[1] += tensor->MemoryCost_GB();
                tensor_list.push_back(tensor);
            }
        }

        auto ffn_iter = layer0->ffn.tensor_map.begin();
        for (; ffn_iter != layer0->ffn.tensor_map.end(); ffn_iter++)
        {
            const auto *tensor = ffn_iter->second->tensor;
            if (tensor != nullptr)
            {
                stat.encoder_layer_size[2] += tensor->MemoryCost_GB();
                tensor_list.push_back(tensor);
            }
        }
    }
    stat.encoder_layer_size[0] = stat.encoder_layer_size[1] + stat.encoder_layer_size[2];

    stat.decoder_layer_num = 0;
    for (const auto *layer_ptr : this->decoder_layers)
    {
        if (layer_ptr != nullptr) {
            stat.decoder_layer_num++;
        }
    }

    memset(stat.decoder_layer_size, 0, sizeof(stat.decoder_layer_size));
    if (!this->decoder_layers.empty())
    {
        const auto *layer0 = *this->decoder_layers.rbegin();
        auto attn_iter = layer0->self_attn.tensor_map.begin();
        for (; attn_iter != layer0->self_attn.tensor_map.end(); attn_iter++)
        {
            const auto *tensor = attn_iter->second->tensor;
            if (tensor != nullptr)
            {
                stat.decoder_layer_size[1] += tensor->MemoryCost_GB();
                tensor_list.push_back(tensor);
            }
        }

        attn_iter = layer0->cross_attn.tensor_map.begin();
        for (; attn_iter != layer0->cross_attn.tensor_map.end(); attn_iter++)
        {
            const auto *tensor = attn_iter->second->tensor;
            if (tensor != nullptr)
            {
                stat.decoder_layer_size[2] += tensor->MemoryCost_GB();
                tensor_list.push_back(tensor);
            }
        }

        auto ffn_iter = layer0->ffn.tensor_map.begin();
        for (; ffn_iter != layer0->ffn.tensor_map.end(); ffn_iter++)
        {
            const auto *tensor = ffn_iter->second->tensor;
            if (tensor != nullptr)
            {
                stat.decoder_layer_size[3] += tensor->MemoryCost_GB();
                tensor_list.push_back(tensor);
            }
        }

        auto moe_iter = layer0->moe.tensor_map.begin();
        //LogKeyInfo("moe_tensor_map size: %d", layer0->moe.tensor_map.size());
        for (; moe_iter != layer0->moe.tensor_map.end(); moe_iter++)
        {
            const auto *tensor = moe_iter->second->tensor;
            if (tensor != nullptr)
            {
                stat.decoder_layer_size[3] += tensor->MemoryCost_GB();
                tensor_list.push_back(tensor);
            }
        }

        for (const auto *expert_ptr : layer0->moe.experts)
        {
            ffn_iter = expert_ptr->tensor_map.begin();
            for (; ffn_iter != expert_ptr->tensor_map.end(); ffn_iter++)
            {
                const auto *tensor = ffn_iter->second->tensor;
                if (tensor != nullptr)
                {
                    stat.decoder_layer_size[3] += tensor->MemoryCost_GB();
                    tensor_list.push_back(tensor);
                }
            }
        }
    }
    stat.decoder_layer_size[0] = stat.decoder_layer_size[1] + stat.decoder_layer_size[2]
        + stat.decoder_layer_size[3];

    stat.pre_layer_size = 0;
    if (this->encoder_input_norm != nullptr)
    {
        stat.pre_layer_size += this->encoder_input_norm->MemoryCost_GB();
        tensor_list.push_back(this->encoder_input_norm);
    }
    if (this->encoder_input_norm_b != nullptr)
    {
        stat.pre_layer_size += this->encoder_input_norm_b->MemoryCost_GB();
        tensor_list.push_back(this->encoder_input_norm_b);
    }

    if (this->decoder_input_norm != nullptr)
    {
        stat.pre_layer_size += this->decoder_input_norm->MemoryCost_GB();
        tensor_list.push_back(this->decoder_input_norm);
    }
    if (this->decoder_input_norm_b != nullptr)
    {
        stat.pre_layer_size += this->decoder_input_norm_b->MemoryCost_GB();
        tensor_list.push_back(this->decoder_input_norm_b);
    }

    stat.post_layer_size = 0;
    if (this->encoder_output_norm != nullptr)
    {
        stat.post_layer_size += this->encoder_output_norm->MemoryCost_GB();
        tensor_list.push_back(this->encoder_output_norm);
    }
    if (this->encoder_output_norm_b != nullptr)
    {
        stat.post_layer_size += this->encoder_output_norm_b->MemoryCost_GB();
        tensor_list.push_back(this->encoder_output_norm_b);
    }
    if (this->decoder_output_norm != nullptr)
    {
        stat.post_layer_size += this->decoder_output_norm->MemoryCost_GB();
        tensor_list.push_back(this->decoder_output_norm);
    }
    if (this->decoder_output_norm_b != nullptr)
    {
        stat.post_layer_size += this->decoder_output_norm_b->MemoryCost_GB();
        tensor_list.push_back(this->decoder_output_norm_b);
    }
    if (this->output != nullptr)
    {
        stat.post_layer_size += this->output->MemoryCost_GB();
        tensor_list.push_back(this->output);
    }
    if (this->output_quant != nullptr)
    {
        stat.post_layer_size += this->output_quant->MemoryCost_GB();
        tensor_list.push_back(this->output_quant);
    }

    stat.embedding_size = 0;
    if (this->encoder_embeddings != nullptr)
    {
        stat.embedding_size += this->encoder_embeddings->MemoryCost_GB();
        tensor_list.push_back(this->encoder_embeddings);
    }
    if (this->decoder_embeddings != nullptr)
    {
        stat.embedding_size += this->decoder_embeddings->MemoryCost_GB();
        tensor_list.push_back(this->decoder_embeddings);
    }

    stat.all_memory_usage = stat.encoder_layer_num * stat.encoder_layer_size[0]
        + stat.decoder_layer_num * stat.decoder_layer_size[0]
        + stat.pre_layer_size + stat.post_layer_size
        + stat.embedding_size;

    for (const auto *tensor : tensor_list)
    {
        auto iter_find = stat.tensor_num_by_element_type.find(tensor->data_type);
        if (iter_find == stat.tensor_num_by_element_type.end()) {
            stat.tensor_num_by_element_type[tensor->data_type] = 1;
        }
        else {
            iter_find->second++;
        }
    }
}

#endif //USE_CUDA

////////////////////////////////////////////////////////////////////////////////
// class TransformerModel

const HostTensor* TransformerModel::FindHostTensor(const string &tensor_name) const
{
    int idx = tensor_spec_table.GetIndex(tensor_name);
    if (idx < 0 || idx >= tensor_spec_table.Size()) {
        return nullptr;
    }

    auto iter = std_network.tensor_map.find(idx);
    return iter != std_network.tensor_map.end() ? iter->second : nullptr;

    //const auto *tensor_spec = &tensor_spec_table.tensor_array[idx];
    //if (tensor_spec->host_tensor == nullptr && tensor_spec->data_source >= 0) {
    //    tensor_spec = &tensor_array[tensor_spec->data_source];
    //}

    //return tensor_spec->host_tensor;
}

HostTensor* TransformerModel::FindHostTensor(const string &tensor_name)
{
    int table_size = tensor_spec_table.Size();
    int idx = tensor_spec_table.GetIndex(tensor_name);
    if (idx < 0 || idx >= table_size) {
        return nullptr;
    }

    auto iter = std_network.tensor_map.find(idx);
    if (iter == std_network.tensor_map.end())
    {
        int data_source = tensor_spec_table.tensor_array[idx].data_source;
        if (data_source >= 0 && data_source < table_size) {
            iter = std_network.tensor_map.find(data_source);
        }
    }
    return iter != std_network.tensor_map.end() ? iter->second : nullptr;

    //auto *tensor_spec = &tensor_spec_table.tensor_array[idx];
    //if (tensor_spec->host_tensor == nullptr && tensor_spec->data_source >= 0) {
    //    tensor_spec = &tensor_array[tensor_spec->data_source];
    //}

    //return tensor_spec->host_tensor;
}

//static
bool TransformerModel::CheckModelSpec(const ModelSpec &spec)
{
    const auto &hparams = spec.hyper_params;
    if (spec.multi_gpu_strategy != MultiGpuStrategy::BY_LAYER)
    {
        int device_num_per_group = (int)spec.device_groups[0].size();
        if (hparams.decoder_heads % device_num_per_group != 0)
        {
            LogError("Tensor parallelism requires that %s (heads: %d, devices: %d)",
                "the head number is a multiple of the device count",
                hparams.decoder_heads, device_num_per_group);
            return false;
        }

        if (hparams.decoder_kv_heads % device_num_per_group != 0)
        {
            LogError("Tensor parallelism requires that %s (kv_heads: %d, devices: %d)",
                "decoder_kv_heads is a multiple of the device count",
                hparams.decoder_kv_heads, device_num_per_group);
            return false;
        }

        if (spec.qkv_format != 0)
        {
            LogError("qkv_format %d is not compatible with tensor parallelism",
                spec.qkv_format);
            return false;
        }
    }

    return true;
}

void TransformerModel::InitModelFileFormatMap(ModelFileFormatMap &the_map)
{
    the_map["std"] = ModelFileFormat::Std;
    the_map["pickle"] = ModelFileFormat::Pickle;
    the_map["safetensors"] = ModelFileFormat::Safetensors;
    the_map["ggml"] = ModelFileFormat::GGML;
    the_map["gguf"] = ModelFileFormat::GGUF;
    the_map["llama2.c"] = ModelFileFormat::LLAMA2_C;
}

void TransformerModel::InitNetworkStructureMap(NetworkStructureMap &the_map)
{
    the_map["general"] = NetworkType::General;
    the_map["transformer"] = NetworkType::Transformer;
    the_map["transformer.encoder_decoder"] = NetworkType::EncoderDecoder_Transformer;
    the_map["transformer.encoder_only"] = NetworkType::EncoderOnly_Transformer;
    the_map["transformer.decoder_only"] = NetworkType::DecoderOnly_Transformer;
    the_map["transformer.decoder_only.sparse_moe"] = NetworkType::SparseMoe_DecoderOnly_Transformer;
    the_map["bert"] = NetworkType::BERT;
    the_map["llama"] = NetworkType::LLAMA;
    the_map["transformer.llama"] = NetworkType::LLAMA;
    the_map["bloom"] = NetworkType::BLOOM;
    the_map["transformer.bloom"] = NetworkType::BLOOM;
}

//static
void TransformerModel::InitMultiGpuStrategyMap(MultiGpuStrategyMap &the_map)
{
    the_map["by_layer"] = MultiGpuStrategy::BY_LAYER;
    the_map["by_tensor"] = MultiGpuStrategy::BY_TENSOR;
}

TRANSFORMER_END
INFER_FLOW_END
