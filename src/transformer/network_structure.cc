#include "network_structure.h"
#include "sslib/string.h"
#include "sslib/log.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

////////////////////////////////////////////////////////////////////////////////
// class NetworkStructure

bool NetworkStructure::Init(NetworkType network_type, int encoder_layer_count,
    int decoder_layer_count, const string &tensor_name_prefix,
    const map<string, string> *tensor_map_ptr)
{
    map<string, string> tensor_map;
    BuildTensorNameMap(tensor_map, network_type, tensor_name_prefix);

    if (tensor_map_ptr != nullptr)
    {
        auto iter = tensor_map_ptr->begin();
        for (; iter != tensor_map_ptr->end(); iter++)
        {
            tensor_map[iter->first] = iter->second;
            if (!tensor_name_prefix.empty()) {
                tensor_map[tensor_name_prefix + iter->first] = iter->second;
            }
        }
    }

    int max_layer_count = max(encoder_layer_count, decoder_layer_count);
    ExpandTensorNameMap(tensor_map_ex_, tensor_map, max_layer_count);

    return true;
}

bool NetworkStructure::UpdateTensorSpecTable(TensorSpecTable &spec_table) const
{
    int failure_num = 0;
    auto &tensor_array = spec_table.tensor_array;
    int tensor_num = (int)tensor_array.size();
    for (int tensor_idx = 0; tensor_idx < tensor_num; tensor_idx++)
    {
        auto &tensor_spec = tensor_array[tensor_idx];
        auto iter_find = tensor_map_ex_.find(tensor_spec.name);
        if (iter_find != tensor_map_ex_.end())
        {
            const string &target_name = iter_find->second;
            //if (target_name == "dec.input_norm.weight") {
            //    LogError("### %s --> %s", tensor_spec.name.c_str(), target_name.c_str());
            //}
            spec_table.name2idx[target_name] = tensor_idx;
            tensor_spec.name = target_name;
        }
        else
        {
            failure_num++;
            if (failure_num <= 3) {
                LogWarning("Cannot map tensor \"%s\" to a target", tensor_spec.name.c_str());
            }
        }
    }

    return true;
}

bool NetworkStructure::TransTensorName(string &target_name, const string &src_name) const
{
    target_name.clear();
    auto iter_find = tensor_map_ex_.find(src_name);
    if (iter_find != tensor_map_ex_.end())
    {
        target_name = iter_find->second;
        return true;
    }

    return false;
}

//static
bool NetworkStructure::IsEncoderDecoderTransformer(NetworkType t)
{
    return t >= NetworkType::EncoderDecoder_Transformer
        && t < NetworkType::EncoderOnly_Transformer;
}

//static
bool NetworkStructure::IsEncoderOnlyTransformer(NetworkType t)
{
    return t >= NetworkType::EncoderOnly_Transformer
        && t < NetworkType::DecoderOnly_Transformer;
}

//static
bool NetworkStructure::IsDecoderOnlyTransformer(NetworkType t)
{
    return t >= NetworkType::DecoderOnly_Transformer;
}

//static
void NetworkStructure::BuildTensorNameToIdMap(map<string, LayerTypeAndTensorId> &tensor_map)
{
    tensor_map.clear();

    LayerTypeAndTensorId v(LayerType::SELF_ATTN, LayerTensorId::ATTN_PRE_NORM);

    for (int attn_type = 0; attn_type < 2; attn_type++)
    {
        string prefix = attn_type == 0 ? "self_attn." : "cross_attn.";
        v.first = attn_type == 0 ? LayerType::SELF_ATTN : LayerType::CROSS_ATTN;

        v.second = LayerTensorId::ATTN_PRE_NORM;
        tensor_map[prefix + "pre_norm.weight"] = v;
        v.second = LayerTensorId::ATTN_PRE_NORM_B;
        tensor_map[prefix + "pre_norm.bias"] = v;
        v.second = LayerTensorId::ATTN_POST_NORM;
        tensor_map[prefix + "post_norm.weight"] = v;
        v.second = LayerTensorId::ATTN_POST_NORM_B;
        tensor_map[prefix + "post_norm.bias"] = v;

        v.second = LayerTensorId::QKV;
        tensor_map[prefix + "qkv.weight"] = v;
        v.second = LayerTensorId::QKV_B;
        tensor_map[prefix + "qkv.bias"] = v;
        v.second = LayerTensorId::WQ;
        tensor_map[prefix + "wq.weight"] = v;
        v.second = LayerTensorId::WQ_B;
        tensor_map[prefix + "wq.bias"] = v;
        v.second = LayerTensorId::WK;
        tensor_map[prefix + "wk.weight"] = v;
        v.second = LayerTensorId::WK_B;
        tensor_map[prefix + "wk.bias"] = v;
        v.second = LayerTensorId::WV;
        tensor_map[prefix + "wv.weight"] = v;
        v.second = LayerTensorId::WV_B;
        tensor_map[prefix + "wv.bias"] = v;
        v.second = LayerTensorId::WO;
        tensor_map[prefix + "wo.weight"] = v;
        v.second = LayerTensorId::WO_B;
        tensor_map[prefix + "wo.bias"] = v;
    }

    v.first = LayerType::FFN;
    v.second = LayerTensorId::FFN_PRE_NORM;
    tensor_map["feed_forward.pre_norm.weight"] = v;
    v.second = LayerTensorId::FFN_PRE_NORM_B;
    tensor_map["feed_forward.pre_norm.bias"] = v;
    v.second = LayerTensorId::FFN_POST_NORM;
    tensor_map["feed_forward.post_norm.weight"] = v;
    v.second = LayerTensorId::FFN_POST_NORM_B;
    tensor_map["feed_forward.post_norm.bias"] = v;

    v.second = LayerTensorId::W1;
    tensor_map["feed_forward.w1.weight"] = v;
    v.second = LayerTensorId::W1_B;
    tensor_map["feed_forward.w1.bias"] = v;
    v.second = LayerTensorId::W2;
    tensor_map["feed_forward.w2.weight"] = v;
    v.second = LayerTensorId::W2_B;
    tensor_map["feed_forward.w2.bias"] = v;
    v.second = LayerTensorId::W3;
    tensor_map["feed_forward.w3.weight"] = v;
    v.second = LayerTensorId::W3_B;
    tensor_map["feed_forward.w3.bias"] = v;
    v.second = LayerTensorId::W1N3;
    tensor_map["feed_forward.w1n3.weight"] = v;
    v.second = LayerTensorId::W1N3_B;
    tensor_map["feed_forward.w1n3.bias"] = v;
}

void NetworkStructure::ExpandTensorNameMap(map<string, string> &tensor_map_ex,
    const map<string, string> &tensor_map, int layers) const
{
    tensor_map_ex.clear();

    string source_str, target_str;
    char buf[32];
    for (auto iter = tensor_map.begin(); iter != tensor_map.end(); iter++)
    {
        bool is_hidden_layer = iter->first.find("{i}") != string::npos;
        if (!is_hidden_layer)
        {
            tensor_map_ex[iter->first] = iter->second;
            //LogKeyInfo("%s --> %s", iter->first.c_str(), iter->second.c_str());
            continue;
        }

        for (int layer_id = 0; layer_id < layers; layer_id++)
        {
            source_str = iter->first;
            target_str = iter->second;
            sprintf(buf, "%d", layer_id);
            String::ReplaceAll(source_str, "{i}", buf);
            String::ReplaceAll(target_str, "{i}", buf);
            tensor_map_ex[source_str] = target_str;

            //if (layer_id == 0) {
            //    LogKeyInfo("%s --> %s", source_str.c_str(), target_str.c_str());
            //}
        }
    }
}

void NetworkStructure::BuildTensorNameMap(map<string, string> &tensor_map,
    NetworkType net_type, const string &tensor_name_prefix)
{
    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(net_type);
    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(net_type);

    auto &tmap = tensor_map;
    const string &pre = tensor_name_prefix;
    tmap[pre + "shared.weight"] = "shared.weight";

    if (!is_decoder_only) {
        BuildTensorNameMap_TransformerEncoder(tensor_map, tensor_name_prefix);
    }

    if (!is_encoder_only)
    {
        switch (net_type)
        {
        case NetworkType::LLAMA:
            BuildTensorNameMap_Llama(tensor_map, tensor_name_prefix);
            break;
        case NetworkType::BLOOM:
            BuildTensorNameMap_Bloom(tensor_map, tensor_name_prefix);
            break;
        case NetworkType::DecoderOnly_Transformer:
            BuildTensorNameMap_TransformerDecoderOnly(tensor_map, tensor_name_prefix);
            break;
        default:
            BuildTensorNameMap_TransformerDecoder(tensor_map, tensor_name_prefix);
            break;
        }
    }
}

void NetworkStructure::BuildTensorNameMap_TransformerEncoder(
    map<string, string> &tensor_map, const string &tensor_name_prefix)
{
    auto &tmap = tensor_map;
    string pre = tensor_name_prefix + "encoder.";
    tmap[pre + "embed_tokens.weight"] = "enc.token_embeddings.weight";

    pre = tensor_name_prefix + "encoder.layers.{i}.";
    for (int type_idx = 0; type_idx < 2; type_idx++)
    {
        string suffix = type_idx == 0 ? ".weight" : ".bias";
        tmap[pre + "self_attn_layer_norm" + suffix] = "enc.{i}.self_attn.pre_norm" + suffix;
        tmap[pre + "self_attn.qkv" + suffix] = "enc.{i}.self_attn.qkv" + suffix;
        tmap[pre + "self_attn.q_proj" + suffix] = "enc.{i}.self_attn.wq" + suffix;
        tmap[pre + "self_attn.k_proj" + suffix] = "enc.{i}.self_attn.wk" + suffix;
        tmap[pre + "self_attn.v_proj" + suffix] = "enc.{i}.self_attn.wv" + suffix;
        tmap[pre + "self_attn.out_proj" + suffix] = "enc.{i}.self_attn.wo" + suffix;

        //tmap[pre + "fc1" + suffix] = "enc.{i}.feed_forward.w1" + suffix;
        //tmap[pre + "fc2" + suffix] = "enc.{i}.feed_forward.w2" + suffix;
        //tmap[pre + "fc3" + suffix] = "enc.{i}.feed_forward.w3" + suffix;

        //tmap[pre + "final_layer_norm" + suffix] = "enc.{i}.feed_forward.post_norm" + suffix;
    }
}

void NetworkStructure::BuildTensorNameMap_TransformerDecoder(
    map<string, string> &tensor_map, const string &tensor_name_prefix)
{
    auto &tmap = tensor_map;
    string pre = tensor_name_prefix + "decoder.";
    tmap[pre + "embed_tokens.weight"] = "dec.token_embeddings.weight";

    pre = tensor_name_prefix;
    tensor_map[pre + "lm_head.weight"] = "output.weight";
    tensor_map[pre + "output.weight"] = "output.weight";
    tensor_map["lm_head.weight"] = "output.weight";
    tensor_map["output.weight"] = "output.weight";

    pre = tensor_name_prefix + "decoder.layers.{i}.";
    for (int type_idx = 0; type_idx < 2; type_idx++)
    {
        string suffix = type_idx == 0 ? ".weight" : ".bias";
        tmap[pre + "self_attn_layer_norm" + suffix] = "dec.{i}.self_attn.pre_norm" + suffix;
        tmap[pre + "self_attn.qkv" + suffix] = "dec.{i}.self_attn.qkv" + suffix;
        tmap[pre + "self_attn.q_proj" + suffix] = "dec.{i}.self_attn.wq" + suffix;
        tmap[pre + "self_attn.k_proj" + suffix] = "dec.{i}.self_attn.wk" + suffix;
        tmap[pre + "self_attn.v_proj" + suffix] = "dec.{i}.self_attn.wv" + suffix;
        tmap[pre + "self_attn.out_proj" + suffix] = "dec.{i}.self_attn.wo" + suffix;

        tmap[pre + "encoder_attn_layer_norm" + suffix] = "dec.{i}.cross_attn.pre_norm" + suffix;
        tmap[pre + "encoder_attn.qkv" + suffix] = "dec.{i}.cross_attn.qkv" + suffix;
        tmap[pre + "encoder_attn.q_proj" + suffix] = "dec.{i}.cross_attn.wq" + suffix;
        tmap[pre + "encoder_attn.k_proj" + suffix] = "dec.{i}.cross_attn.wk" + suffix;
        tmap[pre + "encoder_attn.v_proj" + suffix] = "dec.{i}.cross_attn.wv" + suffix;
        tmap[pre + "encoder_attn.out_proj" + suffix] = "dec.{i}.cross_attn.wo" + suffix;
    }
}

void NetworkStructure::BuildTensorNameMap_TransformerDecoderOnly(
    map<string, string> &tensor_map, const string &tensor_name_prefix)
{
    auto &tmap = tensor_map;
    string pre = tensor_name_prefix;

    tmap[pre + "embed_tokens.weight"] = "dec.token_embeddings.weight";
    tensor_map[pre + "lm_head.weight"] = "output.weight";
    tensor_map[pre + "output.weight"] = "output.weight";
    tensor_map["lm_head.weight"] = "output.weight";
    tensor_map["output.weight"] = "output.weight";

    pre = tensor_name_prefix + "layers.{i}.";
    for (int type_idx = 0; type_idx < 2; type_idx++)
    {
        string suffix = type_idx == 0 ? ".weight" : ".bias";
        tmap[pre + "self_attn.qkv" + suffix] = "dec.{i}.self_attn.qkv" + suffix;
        tmap[pre + "self_attn.q_proj" + suffix] = "dec.{i}.self_attn.wq" + suffix;
        tmap[pre + "self_attn.k_proj" + suffix] = "dec.{i}.self_attn.wk" + suffix;
        tmap[pre + "self_attn.v_proj" + suffix] = "dec.{i}.self_attn.wv" + suffix;
        tmap[pre + "self_attn.o_proj" + suffix] = "dec.{i}.self_attn.wo" + suffix;
        tmap[pre + "self_attn.out_proj" + suffix] = "dec.{i}.self_attn.wo" + suffix;

        tmap[pre + "mlp.gate_proj" + suffix] = "dec.{i}.feed_forward.w1" + suffix;
        tmap[pre + "mlp.down_proj" + suffix] = "dec.{i}.feed_forward.w2" + suffix;
        tmap[pre + "mlp.up_proj" + suffix] = "dec.{i}.feed_forward.w3" + suffix;

        tmap[pre + "input_layernorm" + suffix] = "dec.{i}.self_attn.pre_norm" + suffix;
        tmap[pre + "post_attention_layernorm" + suffix] = "dec.{i}.feed_forward.pre_norm" + suffix;
    }
}

void NetworkStructure::BuildTensorNameMap_Llama(map<string, string> &tensor_map,
    const string &tensor_name_prefix)
{
    const string &pre = tensor_name_prefix;
    tensor_map[pre + "embed_tokens.weight"]     = "token_embeddings.weight";
    tensor_map[pre + "tok_embeddings.weight"]   = "token_embeddings.weight";
    tensor_map[pre + "lm_head.weight"]          = "output.weight";
    tensor_map[pre + "output.weight"]           = "output.weight";
    tensor_map["token_embd.weight"]             = "token_embeddings.weight";
    tensor_map["lm_head.weight"]                = "output.weight";
    tensor_map["output.weight"]                 = "output.weight";

    tensor_map[pre + "norm.weight"] = "dec.output_norm.weight";
    tensor_map["output_norm.weight"]            = "dec.output_norm.weight";

    tensor_map[pre + "layers.{i}.self_attn.q_proj.weight"]  = "dec.{i}.self_attn.wq.weight";
    tensor_map[pre + "layers.{i}.self_attn.k_proj.weight"]  = "dec.{i}.self_attn.wk.weight";
    tensor_map[pre + "layers.{i}.self_attn.v_proj.weight"]  = "dec.{i}.self_attn.wv.weight";
    tensor_map[pre + "layers.{i}.self_attn.o_proj.weight"]  = "dec.{i}.self_attn.wo.weight";
    tensor_map[pre + "layers.{i}.attention.wq.weight"]      = "dec.{i}.self_attn.wq.weight";
    tensor_map[pre + "layers.{i}.attention.wk.weight"]      = "dec.{i}.self_attn.wk.weight";
    tensor_map[pre + "layers.{i}.attention.wv.weight"]      = "dec.{i}.self_attn.wv.weight";
    tensor_map[pre + "layers.{i}.attention.wo.weight"]      = "dec.{i}.self_attn.wo.weight";
    tensor_map[pre + "layers.{i}.self_attn.W_pack.weight"] = "dec.{i}.self_attn.qkv.weight";
    tensor_map[pre + "{i}.attn_q.weight"]       = "dec.{i}.self_attn.wq.weight";
    tensor_map[pre + "{i}.attn_k.weight"]       = "dec.{i}.self_attn.wk.weight";
    tensor_map[pre + "{i}.attn_v.weight"]       = "dec.{i}.self_attn.wv.weight";
    tensor_map[pre + "{i}.attn_output.weight"]  = "dec.{i}.self_attn.wo.weight";

    tensor_map[pre + "layers.{i}.mlp.gate_proj.weight"]     = "dec.{i}.feed_forward.w1.weight";
    tensor_map[pre + "layers.{i}.mlp.down_proj.weight"]     = "dec.{i}.feed_forward.w2.weight";
    tensor_map[pre + "layers.{i}.mlp.up_proj.weight"]       = "dec.{i}.feed_forward.w3.weight";
    tensor_map[pre + "layers.{i}.feed_forward.w1.weight"]   = "dec.{i}.feed_forward.w1.weight";
    tensor_map[pre + "layers.{i}.feed_forward.w2.weight"]   = "dec.{i}.feed_forward.w2.weight";
    tensor_map[pre + "layers.{i}.feed_forward.w3.weight"]   = "dec.{i}.feed_forward.w3.weight";
    tensor_map[pre + "{i}.ffn_gate.weight"]     = "dec.{i}.feed_forward.w1.weight";
    tensor_map[pre + "{i}.ffn_down.weight"]     = "dec.{i}.feed_forward.w2.weight";
    tensor_map[pre + "{i}.ffn_up.weight"]       = "dec.{i}.feed_forward.w3.weight";

    tensor_map[pre + "layers.{i}.input_layernorm.weight"]   = "dec.{i}.self_attn.pre_norm.weight";
    tensor_map[pre + "layers.{i}.attention_norm.weight"]    = "dec.{i}.self_attn.pre_norm.weight";
    tensor_map[pre + "{i}.attn_norm.weight"]                = "dec.{i}.self_attn.pre_norm.weight";
    tensor_map[pre + "layers.{i}.post_attention_layernorm.weight"]  = "dec.{i}.feed_forward.pre_norm.weight";
    tensor_map[pre + "layers.{i}.ffn_norm.weight"]                  = "dec.{i}.feed_forward.pre_norm.weight";
    tensor_map[pre + "{i}.ffn_norm.weight"]                         = "dec.{i}.feed_forward.pre_norm.weight";
}

void NetworkStructure::BuildTensorNameMap_Bloom(map<string, string> &tensor_map,
    const string &tensor_name_prefix)
{
    const string &pre = tensor_name_prefix;
    tensor_map[pre + "word_embeddings.weight"] = "token_embeddings.weight";
    tensor_map[pre + "tok_embeddings.weight"] = "token_embeddings.weight";
    tensor_map[pre + "word_embeddings_layernorm.weight"] = "dec.input_norm.weight";
    tensor_map[pre + "norm.weight"] = "dec.input_norm.weight";
    tensor_map[pre + "word_embeddings_layernorm.bias"] = "dec.input_norm.bias";
    tensor_map[pre + "norm.bias"] = "dec.input_norm.bias";

    tensor_map[pre + "lm_head.weight"] = "output.weight";
    tensor_map["lm_head.weight"] = "output.weight";
    tensor_map[pre + "output.weight"] = "output.weight";

    tensor_map[pre + "ln_f.weight"] = "dec.output_norm.weight";
    tensor_map[pre + "output_norm.weight"] = "dec.output_norm.weight";
    tensor_map[pre + "ln_f.bias"] = "dec.output_norm.bias";
    tensor_map[pre + "output_norm.bias"] = "dec.output_norm.bias";

    tensor_map[pre + "h.{i}.self_attention.query_key_value.weight"]
        = "dec.{i}.self_attn.qkv.weight";
    tensor_map[pre + "h.{i}.self_attention.query_key_value.bias"] = "dec.{i}.self_attn.qkv.bias";
    tensor_map[pre + "h.{i}.self_attention.dense.weight"] = "dec.{i}.self_attn.wo.weight";
    tensor_map[pre + "h.{i}.self_attention.dense.bias"] = "dec.{i}.self_attn.wo.bias";
    tensor_map[pre + "h.{i}.mlp.dense_h_to_4h.weight"] = "dec.{i}.feed_forward.w1.weight";
    tensor_map[pre + "h.{i}.mlp.dense_h_to_4h.bias"] = "dec.{i}.feed_forward.w1.bias";
    tensor_map[pre + "h.{i}.mlp.dense_4h_to_h.weight"] = "dec.{i}.feed_forward.w2.weight";
    tensor_map[pre + "h.{i}.mlp.dense_4h_to_h.bias"] = "dec.{i}.feed_forward.w2.bias";

    tensor_map[pre + "h.{i}.input_layernorm.weight"] = "dec.{i}.self_attn.pre_norm.weight";
    tensor_map[pre + "h.{i}.input_layernorm.bias"] = "dec.{i}.self_attn.pre_norm.bias";
    tensor_map[pre + "h.{i}.post_attention_layernorm.weight"] = "dec.{i}.feed_forward.pre_norm.weight";
    tensor_map[pre + "h.{i}.post_attention_layernorm.bias"] = "dec.{i}.feed_forward.pre_norm.bias";

    tensor_map[pre + "layers.{i}.attention.query_key_value.weight"]
        = "dec.{i}.self_attn.qkv.weight";
    tensor_map[pre + "layers.{i}.attention.query_key_value.bias"] = "dec.{i}.self_attn.qkv.bias";
    tensor_map[pre + "layers.{i}.attention.wo.weight"] = "dec.{i}.self_attn.wo.weight";
    tensor_map[pre + "layers.{i}.attention.wo.bias"] = "dec.{i}.self_attn.wo.bias";
    tensor_map[pre + "layers.{i}.feed_forward.w1.weight"] = "dec.{i}.feed_forward.w1.weight";
    tensor_map[pre + "layers.{i}.feed_forward.w1.bias"] = "dec.{i}.feed_forward.w1.bias";
    tensor_map[pre + "layers.{i}.feed_forward.w2.weight"] = "dec.{i}.feed_forward.w2.weight";
    tensor_map[pre + "layers.{i}.feed_forward.w2.bias"] = "dec.{i}.feed_forward.w2.bias";

    tensor_map[pre + "layers.{i}.attention_norm.weight"] = "dec.{i}.self_attn.pre_norm.weight";
    tensor_map[pre + "layers.{i}.attention_norm.bias"] = "dec.{i}.self_attn.pre_norm.bias";
    tensor_map[pre + "layers.{i}.ffn_norm.weight"] = "dec.{i}.feed_forward.pre_norm.weight";
    tensor_map[pre + "layers.{i}.ffn_norm.bias"] = "dec.{i}.feed_forward.pre_norm.bias";
}

TRANSFORMER_END
INFER_FLOW_END
