#include "model_reader.h"
#include "sslib/path.h"
#include "sslib/log.h"
#include "sslib/stream_helper.h"
#include "tensor/tensor_util.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

void ModelReader::Init(NetworkBuilder *network_builder)
{
    InitTokenByteMap();
    network_builder_ = network_builder;
}

bool ModelReader::Load(TransformerModel &model, TransformerContext &ctx,
    const ModelSpec &spec, bool is_study_mode)
{
    bool ret = true;
    JsonParser jparser;
    jparser.Init();

    bool has_embedded_hparams = false;
    bool has_embedded_tokenizer_data = false;
    bool is_llama2_c = false;
    switch (spec.model_file_format)
    {
    case ModelFileFormat::Std:
    case ModelFileFormat::GGML:
    case ModelFileFormat::GGUF:
        has_embedded_hparams = true;
        has_embedded_tokenizer_data = true;
        break;
    case ModelFileFormat::LLAMA2_C:
        has_embedded_hparams = true;
        has_embedded_tokenizer_data = true;
        is_llama2_c = true;
        break;
    default: break;
    }

    string file_path;
    TokenizerConfig tok_config;
    if (!spec.config_file.empty())
    {
        file_path = spec.dir + spec.config_file;
        ret = LoadConfigJson(model, tok_config, spec, file_path, jparser, has_embedded_hparams);
        if (!ret)
        {
            LogError("Failed to load the model config from %s", file_path.c_str());
            return false;
        }
    }
    else if (!has_embedded_hparams)
    {
        LogError("config_file should NOT be empty.");
        return false;
    }

    const auto &hparams = model.spec.hyper_params;
    if (!has_embedded_hparams)
    {
        network_builder_->InitNetworkStructure(model.std_network, model.spec);
    }

    if (!has_embedded_hparams || is_llama2_c)
    {
        model.network.Init(model.spec.network_structure, hparams.encoder_layers,
            hparams.decoder_layers, hparams.experts, model.spec.tensor_name_prefix,
            &model.spec.tensor_name_map);
    }

    model.spec.is_eager_device_building = false;
    NetworkType net_type = model.spec.network_structure;
    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(net_type);
    if (is_decoder_only && model.spec.multi_gpu_strategy == MultiGpuStrategy::BY_LAYER
        //&& (int)model.spec.device_groups.size() == 1
        && !has_embedded_hparams)
    {
        model.spec.is_eager_device_building = true;
    }

    ret = TransformerModel::CheckModelSpec(model.spec);
    Macro_RetFalseIf(!ret);

    if (!spec.generation_config_file.empty())
    {
        file_path = spec.dir + spec.generation_config_file;
        ret = LoadGeneratinConfig(model, file_path, jparser);
        if (!ret)
        {
            LogError("Failed to load the generation config from %s", file_path.c_str());
            return false;
        }
    }

    if (!has_embedded_tokenizer_data)
    {
        ret = LoadTokenizer(model, spec, jparser);
        Macro_RetFalseIf(!ret);
    }

    vector<string> model_file_list;
    string index_json = "index.json";
    string model_file = spec.model_files.empty() ? "" : spec.model_files[0];
    auto offset = model_file.rfind(index_json);
    if (offset != string::npos && offset + index_json.size() == model_file.size())
    {
        file_path = spec.dir + model_file;
        map<string, string> weight_map;
        ret = LoadWeightMap(weight_map, file_path, jparser);
        Macro_RetxFalseIf(!ret, LogError("Failed to load the weight map"));

        GetFileSetFromWeightMap(model_file_list, weight_map);
    }

    ModelSpec spec_for_net = spec;
    switch (spec.model_file_format)
    {
    case ModelFileFormat::Pickle:
        ret = LoadModel_Pickle(model, ctx, spec, model_file_list, is_study_mode);
        break;
    case ModelFileFormat::Safetensors:
        ret = LoadModel_Safetensors(model, ctx, spec, model_file_list,
            jparser, is_study_mode);
        break;
    case ModelFileFormat::GGML:
        model.is_cpu_tensor_row_major = false;
        ret = LoadModel_GGML(model, ctx, spec);
        break;
    case ModelFileFormat::GGUF:
        model.is_cpu_tensor_row_major = false;
        ret = LoadModel_GGUF(model, ctx, spec);
        break;
    case ModelFileFormat::LLAMA2_C:
        ret = LoadModel_Llama2DotC(model, ctx, spec, is_study_mode);
        break;
    case ModelFileFormat::Std:
    default:
        ret = LoadModel_Std(model, ctx, spec, is_study_mode);
        break;
    }

    if (has_embedded_hparams && !is_llama2_c)
    {
        network_builder_->InitNetworkStructure(model.std_network, model.spec);
        model.network.Init(model.spec.network_structure, hparams.encoder_layers,
            hparams.decoder_layers, hparams.experts, model.spec.tensor_name_prefix,
            &model.spec.tensor_name_map);
    }

    model.network.UpdateTensorSpecTable(model.tensor_spec_table);

    auto &vocab = model.vocabulary;
    if (!tok_config.unk_str.empty()) {
        vocab.SetUnk(tok_config.unk_str);
    }
    if (!tok_config.bos_str.empty()) {
        vocab.SetBos(tok_config.bos_str);
    }
    if (!tok_config.eos_str.empty()) {
        vocab.SetEos(tok_config.eos_str);
    }

    if (!spec.unk_token.empty()) {
        vocab.SetUnk(spec.unk_token);
    }
    if (!spec.pad_token.empty()) {
        model.pad_token_id = vocab.StrToId(spec.pad_token);
    }
    if (!spec.bos_token.empty()) {
        vocab.SetBos(spec.bos_token);
    }
    if (!spec.eos_token.empty()) {
        vocab.SetEos(spec.eos_token);
    }
    model.mask_token_id = vocab.StrToId(spec.mask_token.empty() ? "[MASK]" : spec.mask_token);

    //const auto &hparams = model.hyper_params;
    //model.network.Init(spec.network_structure, hparams.layers,
    //    model.tensor_spec_table, &model.spec.tensor_name_map);

    //LogKeyInfo("Building the network (%d)...", (int)spec.network_structure);
    //ret = BuildNetwork(model, ctx, spec_for_net);
    //Macro_RetxFalseIf(!ret, LogError("Failed to build the network"));

    return ret;
}

//static
bool ModelReader::LoadModelSpecJson(ModelSpec &model_spec, const string &file_path,
    const JsonParser &jparser)
{
    auto &hparams = model_spec.hyper_params;

    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to open the model-spec file."));

    wstring json_str;
    wstring line_str;
    while (reader.GetLine(line_str))
    {
        if (json_str.empty())
        {
            WString::Trim(line_str);
            if (line_str.empty() || line_str[0] == L'#') {
                continue;
            }
        }

        if (!line_str.empty())
        {
            if (!json_str.empty()) {
                json_str += L'\n';
            }
            json_str += line_str;
        }
    }

    JsonDoc jdoc;
    ret = jparser.Parse(jdoc, json_str);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    ModelFileFormatMap model_file_format_map;
    NetworkStructureMap network_structure_map;
    //MultiGpuStrategyMap multi_gpu_strategy_map;
    TransformerModel::InitModelFileFormatMap(model_file_format_map);
    TransformerModel::InitNetworkStructureMap(network_structure_map);
    //TransformerModel::InitMultiGpuStrategyMap(multi_gpu_strategy_map);

    TokenizationAlgMap tok_alg_map;
    TextTokenizer::InitAlgorithmMap(tok_alg_map);

    TensorNormAlgMap tensor_norm_map;
    ActivationFnMap activation_fn_map;
    PositionEmbeddingAlgMap pos_embedding_alg_map;
    HostTensorOpr::InitTensorNormAlgMap(tensor_norm_map);
    HostTensorOpr::InitActivationFnMap(activation_fn_map);
    HostTensorOpr::InitPositionEmbeddingAlgMap(pos_embedding_alg_map);

    JsonObject jobj = jdoc.GetJObject();
    JsonArray jarray;
    string str;

    ret = jobj.GetFieldValue(model_spec.config_file, L"config_file", jdoc);
    Macro_RetxFalseIf(!ret, LogError("Cannot find the config_file field"));

    ret = jobj.GetFieldValue(jarray, L"model_files", jdoc);
    for (uint32_t file_idx = 0; ret && file_idx < jarray.size; file_idx++)
    {
        ret = jarray.items[file_idx].GetString(str);
        model_spec.model_files.push_back(str);
    }
    Macro_RetxFalseIf(!ret, LogError("Invalid model_files value"));

    ret = jobj.GetFieldValue(str, L"model_file_format", jdoc);
    Macro_RetxFalseIf(!ret, LogError("Cannot find the model_file_format field"));

    auto file_format_iter = model_file_format_map.find(str);
    if (file_format_iter == model_file_format_map.end())
    {
        LogError("Invalid model file format: %s", str.c_str());
        return false;
    }
    model_spec.model_file_format = file_format_iter->second;

    jobj.GetFieldValue(hparams.vocab_size, L"vocab_size", jdoc, true);
    jobj.GetFieldValue(hparams.padded_vocab_size, L"padded_vocab_size", jdoc, true);
    jobj.GetFieldValue(hparams.vocab_size, L"input_vocab_size", jdoc, true);
    jobj.GetFieldValue(hparams.output_vocab_size, L"output_vocab_size", jdoc, true);

    jobj.GetFieldValue(model_spec.unk_token, L"unk_token", jdoc);
    jobj.GetFieldValue(model_spec.pad_token, L"pad_token", jdoc);
    jobj.GetFieldValue(model_spec.bos_token, L"bos_token", jdoc);
    jobj.GetFieldValue(model_spec.eos_token, L"eos_token", jdoc);

    if (jobj.GetFieldValue(jarray, L"tokenizer_files", jdoc))
    {
        for (uint32_t file_idx = 0; ret && file_idx < jarray.size; file_idx++)
        {
            ret = jarray.items[file_idx].GetString(str);
            model_spec.tokenizer_files.push_back(str);
        }
        Macro_RetxFalseIf(!ret, LogError("Invalid tokenizer_files value"));
    }
    else if (jobj.GetFieldValue(str, L"tokenizer_file", jdoc))
    {
        model_spec.tokenizer_files.push_back(str);
    }

    jobj.GetFieldValue(model_spec.token_bytes_mapping, L"token_bytes_mapping", jdoc);
    jobj.GetFieldValue(model_spec.token_remap_file, L"token_remap_file", jdoc);

    if (jobj.GetFieldValue(str, L"tokenization_algorithm", jdoc))
    {
        auto iter_tok_alg = tok_alg_map.find(str);
        if (iter_tok_alg == tok_alg_map.end())
        {
            LogError("Invalid tokenization algorithm: %s", str.c_str());
            return false;
        }

        model_spec.tokenization_algorithm = iter_tok_alg->second;
    }

    jobj.GetFieldValue(model_spec.generation_config_file, L"generation_config", jdoc);

    JsonObject net_struc_obj;
    ret = jobj.GetFieldValue(net_struc_obj, L"network_structure", jdoc);
    Macro_RetxFalseIf(!ret, LogError("Invalid network_structure value"));

    ret = net_struc_obj.GetFieldValue(str, L"type", jdoc);
    Macro_RetxFalseIf(!ret, LogError("Invalid network_structure.type value"));
    auto net_type_iter = network_structure_map.find(str);
    if (net_type_iter == network_structure_map.end())
    {
        LogError("Invalid network structure type: %s", str.c_str());
        return false;
    }

    model_spec.network_structure = net_type_iter->second;
    switch (model_spec.network_structure)
    {
    case NetworkType::LLAMA:
        model_spec.norm_alg = TensorNormAlg::RMS;
        model_spec.pos_embedding_alg = PositionEmbeddingAlg::ROPE;
        model_spec.activation_fn = ActivationFn::SILU;
        break;
    case NetworkType::BLOOM:
        model_spec.norm_alg = TensorNormAlg::STD;
        model_spec.pos_embedding_alg = PositionEmbeddingAlg::ALIBI;
        model_spec.activation_fn = ActivationFn::GELU;
        break;
    default: break;
    }

    if (net_struc_obj.GetFieldValue(str, L"normalization_function", jdoc))
    {
        auto iter = tensor_norm_map.find(str);
        if (iter == tensor_norm_map.end())
        {
            LogError("Invalid normalization function: %s", str.c_str());
            return false;
        }
        model_spec.norm_alg = iter->second;
    }

    if (net_struc_obj.GetFieldValue(str, L"activation_function", jdoc))
    {
        auto iter = activation_fn_map.find(str);
        if (iter == activation_fn_map.end())
        {
            LogError("Invalid activation function: %s", str.c_str());
            return false;
        }
        model_spec.activation_fn = iter->second;
    }

    if (net_struc_obj.GetFieldValue(str, L"position_embedding", jdoc))
    {
        auto iter = pos_embedding_alg_map.find(str);
        if (iter == pos_embedding_alg_map.end())
        {
            LogError("Invalid position embedding algorithm: %s", str.c_str());
            return false;
        }
        model_spec.pos_embedding_alg = iter->second;
    }

    net_struc_obj.GetFieldValue(model_spec.has_embedding_linear_norm,
        L"has_embedding_linear_norm", jdoc);
    net_struc_obj.GetFieldValue(model_spec.embedding_linear_scale,
        L"embedding_linear_scale", jdoc);
    net_struc_obj.GetFieldValue(model_spec.has_linear_norm_before_sinusoidal,
        L"linear_norm_before_sinusoidal", jdoc);

    net_struc_obj.GetFieldValue(model_spec.rope_theta, L"rope_theta", jdoc);
    net_struc_obj.GetFieldValue(model_spec.partial_rotary_factor, L"partial_rotary_factor", jdoc);

    net_struc_obj.GetFieldValue(model_spec.pos_embedding_offset, L"pos_embedding_offset", jdoc);

    net_struc_obj.GetFieldValue(model_spec.qk_column_order, L"qk_column_order", jdoc);
    net_struc_obj.GetFieldValue(model_spec.qkv_format, L"qkv_format", jdoc);
    net_struc_obj.GetFieldValue(model_spec.kq_scale, L"kq_scale", jdoc);
    net_struc_obj.GetFieldValue(model_spec.normalize_lm_head, L"normalize_lm_head", jdoc);
    net_struc_obj.GetFieldValue(model_spec.is_attn_post_as_residual, L"is_attn_post_as_residual", jdoc);
    net_struc_obj.GetFieldValue(model_spec.is_parallel_attn, L"is_parallel_attn", jdoc);
    net_struc_obj.GetFieldValue(model_spec.mlp_attn_share_input, L"mlp_attn_share_input", jdoc);

    net_struc_obj.GetFieldValue(model_spec.use_self_attn_pre_norm, L"use_self_attn_pre_norm", jdoc);
    net_struc_obj.GetFieldValue(model_spec.attn_pre_norm_base, L"attn_pre_norm_base", jdoc);
    net_struc_obj.GetFieldValue(model_spec.ffn_pre_norm_base, L"ffn_pre_norm_base", jdoc);
    net_struc_obj.GetFieldValue(model_spec.output_norm_base, L"output_norm_base", jdoc);

    net_struc_obj.GetFieldValue(model_spec.attn_out_scale, L"attn_out_scale", jdoc);
    net_struc_obj.GetFieldValue(model_spec.ffn_out_scale, L"ffn_out_scale", jdoc);
    net_struc_obj.GetFieldValue(model_spec.out_scale, L"out_scale", jdoc);

    net_struc_obj.GetFieldValue(hparams.experts, L"expert_count", jdoc);
    net_struc_obj.GetFieldValue(hparams.in_use_experts, L"using_expert_count", jdoc);
    net_struc_obj.GetFieldValue(hparams.moe_top_k, L"moe_top_k", jdoc);
    net_struc_obj.GetFieldValue(hparams.moe_norm_top_k_prob, L"moe_norm_top_k_prob", jdoc);
    net_struc_obj.GetFieldValue(hparams.moe_layer_start, L"moe_layer_start", jdoc);
    net_struc_obj.GetFieldValue(hparams.moe_layer_end, L"moe_layer_end", jdoc);
    net_struc_obj.GetFieldValue(hparams.has_shared_expert, L"has_shared_expert", jdoc);
    hparams.in_use_experts = min(hparams.in_use_experts, hparams.experts);

    if (model_spec.model_file_format == ModelFileFormat::GGUF
        && model_spec.tensor_name_prefix.empty())
    {
        model_spec.tensor_name_prefix = "blk.";
    }
    net_struc_obj.GetFieldValue(model_spec.tensor_name_prefix, L"tensor_name_prefix", jdoc);

    JsonObject tensor_name_map_obj;
    if (net_struc_obj.GetFieldValue(tensor_name_map_obj, L"tensor_name_mapping", jdoc))
    {
        string source_name, target_name;
        for (uint32_t field_idx = 0; field_idx < tensor_name_map_obj.size; field_idx++)
        {
            const auto &fld = tensor_name_map_obj.items[field_idx];
            fld.name.ToString(source_name);
            fld.value.GetString(target_name);
            model_spec.tensor_name_map[source_name] = target_name;
        }
    }

    if (net_struc_obj.GetFieldValue(tensor_name_map_obj, L"tensor_name_pre_mapping", jdoc))
    {
        string source_name, target_name;
        for (uint32_t field_idx = 0; field_idx < tensor_name_map_obj.size; field_idx++)
        {
            const auto &fld = tensor_name_map_obj.items[field_idx];
            fld.name.ToString(source_name);
            fld.value.GetString(target_name);
            model_spec.tensor_name_pre_map[source_name] = target_name;
        }
    }

    return ret;
}

//static
bool ModelReader::LoadConfigJson(TransformerModel &model, TokenizerConfig &tok_config,
    const ModelSpec &spec, const string &file_path, const JsonParser &jparser,
    bool has_embedded_hparams)
{
    NetworkType net_type = spec.network_structure;
    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(net_type);
    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(net_type);
    //bool is_encoder_decoder = NetworkStructure::IsEncoderDecoderTransformer(net_type);
    //bool is_unknown = !is_decoder_only && !is_encoder_only && !is_encoder_decoder;

    wstring content;
    bool ret = Path::GetFileContent_Text(content, file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to get the file content."));

    JsonDoc jdoc;
    ret = jparser.Parse(jdoc, content);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    JsonObject jobj = jdoc.GetJObject();

    wstring model_type;
    auto &hparams = model.spec.hyper_params;
    //ret = ret && jobj.GetFieldValue(model_type, L"model_type", jdoc);
    //model.model_type = StringUtil::ToUtf8(model_type);

    //jobj.GetFieldValue(hparams.layer_norm_epsilon, L"layer_norm_epsilon", jdoc);

    if (!has_embedded_hparams)
    {
        if (hparams.vocab_size <= 0)
        {
            ret = jobj.GetFieldValue(hparams.vocab_size, L"vocab_size", jdoc);
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.vocab_size, L"padded_vocab_size", jdoc);
            }

            if (!ret)
            {
                LogError("Failed to get the value of vocab_size or padded_vocab_size");
                return false;
            }
        }

        ret = jobj.GetFieldValue(hparams.embd_dims, L"d_model", jdoc);
        if (!ret) {
            ret = jobj.GetFieldValue(hparams.embd_dims, L"n_embed", jdoc);
        }
        if (!ret) {
            ret = jobj.GetFieldValue(hparams.embd_dims, L"n_embd", jdoc);
        }
        if (!ret) {
            ret = jobj.GetFieldValue(hparams.embd_dims, L"hidden_size", jdoc);
        }
        if (!ret)
        {
            LogError("Failed to get the value of d_model, n_embed or hidden_size");
            return false;
        }

        //encoder layers
        if (!is_decoder_only)
        {
            ret = jobj.GetFieldValue(hparams.encoder_layers, L"encoder_layers", jdoc);
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.encoder_layers, L"num_encoder_layers", jdoc);
            }

            if (!ret && is_encoder_only)
            {
                ret = jobj.GetFieldValue(hparams.encoder_layers, L"n_layer", jdoc);
                if (!ret) {
                    ret = jobj.GetFieldValue(hparams.encoder_layers, L"num_hidden_layers", jdoc);
                }
                if (!ret)
                {
                    LogError("Failed to get the value of %s, %s, %s, or %s", "encoder_layers",
                        "num_encoder_layers", "num_hidden_layers", "n_layer");
                    return false;
                }
            }
            else if (!ret && !is_encoder_only)
            {
                LogError("Failed to get the value of encoder_layers");
                return false;
            }

            ret = jobj.GetFieldValue(hparams.encoder_heads, L"encoder_attention_heads", jdoc);
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.encoder_heads, L"num_heads", jdoc);
            }
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.encoder_heads, L"n_head", jdoc);
            }
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.encoder_heads, L"num_attention_heads", jdoc);
            }

            if (!ret)
            {
                LogError("Failed to get the value of %s, %s, %s, or %s", "encoder_attention_heads",
                    "num_attention_heads", "num_heads", "n_head");
                return false;
            }
        }

        if (!is_encoder_only)
        {
            ret = jobj.GetFieldValue(hparams.decoder_layers, L"decoder_layers", jdoc);
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.decoder_layers, L"num_decoder_layers", jdoc);
            }

            if (!ret && is_decoder_only)
            {
                ret = jobj.GetFieldValue(hparams.decoder_layers, L"n_layer", jdoc);
                if (!ret) {
                    ret = jobj.GetFieldValue(hparams.decoder_layers, L"num_hidden_layers", jdoc);
                }
                if (!ret) {
                    ret = jobj.GetFieldValue(hparams.decoder_layers, L"num_layers", jdoc);
                }

                if (!ret)
                {
                    LogError("Failed to get the value of %s, %s, %s, or %s", "decoder_layers",
                        "num_hidden_layers", "num_layers", "n_layer");
                    return false;
                }
            }
            else if (!ret && !is_decoder_only)
            {
                LogError("Failed to get the value of decoder_layers");
                return false;
            }

            ret = jobj.GetFieldValue(hparams.decoder_intermediate_size, L"intermediate_size", jdoc);
        }

        if (!is_encoder_only)
        {
            ret = jobj.GetFieldValue(hparams.decoder_heads, L"decoder_attention_heads", jdoc);
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.decoder_heads, L"num_attention_heads", jdoc);
            }
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.decoder_heads, L"num_heads", jdoc);
            }
            if (!ret) {
                ret = jobj.GetFieldValue(hparams.decoder_heads, L"n_head", jdoc);
            }

            if (!ret)
            {
                LogError("Failed to get the value of %s, %s, %s, or %s", "decoder_attention_heads",
                    "num_attention_heads", "num_heads", "n_head");
                return false;
            }

            bool is_succ = jobj.GetFieldValue(hparams.decoder_kv_heads, L"num_key_value_heads", jdoc);
            if (!is_succ) {
                is_succ = jobj.GetFieldValue(hparams.decoder_kv_heads, L"num_kv_heads", jdoc);
            }
            if (!is_succ) {
                is_succ = jobj.GetFieldValue(hparams.decoder_kv_heads, L"multi_query_group_num", jdoc);
            }

            if (!is_succ) {
                hparams.decoder_kv_heads = hparams.decoder_heads;
            }
        }

        if (hparams.encoder_kv_heads <= 0) {
            hparams.encoder_kv_heads = hparams.encoder_heads;
        }

        if (is_decoder_only)
        {
            LogKeyInfo("vocab_size: %d, embd_dims: %d, %s: %d, %s: %d, %s: %d",
                hparams.vocab_size, hparams.embd_dims, "decoder layers", hparams.decoder_layers,
                "decoder heads", hparams.decoder_heads, "decoder kv heads", hparams.decoder_kv_heads);
        }
        else if (is_encoder_only)
        {
            LogKeyInfo("vocab_size: %d, embd_dims: %d, %s: %d, %s: %d, %s: %d",
                hparams.vocab_size, hparams.embd_dims, "encoder layers", hparams.encoder_layers,
                "encoder heads", hparams.encoder_heads, "encoder kv heads", hparams.encoder_kv_heads);
        }
        else
        {
            LogKeyInfo("vocab_size: %d, embd_dims: %d, %s: (%d, %d, %d), %s: (%d, %d, %d)",
                hparams.vocab_size, hparams.embd_dims, "encoder layers, heads, and kv heads",
                hparams.encoder_layers, hparams.encoder_heads, hparams.encoder_kv_heads,
                "decoder layers, heads, and kv heads", hparams.decoder_layers,
                hparams.decoder_heads, hparams.decoder_kv_heads);
        }
    }

    jobj.GetFieldValue(model.spec.rope_theta, L"rope_theta", jdoc);
    jobj.GetFieldValue(model.spec.partial_rotary_factor, L"partial_rotary_factor", jdoc);

    bool is_multi_query_attention = false;
    jobj.GetFieldValue(is_multi_query_attention, L"multi_query", jdoc);
    if (is_multi_query_attention) {
        hparams.decoder_kv_heads = 1;
    }

    //jobj.GetFieldValue(model.spec.is_parallel_attn, L"parallel_attn", jdoc);

    tok_config.Clear();
    wstring str;
    if (jobj.GetFieldValue(str, L"unk_token", jdoc)) {
        tok_config.unk_str = StringUtil::ToUtf8(str);
    }
    if (jobj.GetFieldValue(str, L"bos_token", jdoc)) {
        tok_config.bos_str = StringUtil::ToUtf8(str);
    }
    if (jobj.GetFieldValue(str, L"eos_token", jdoc)) {
        tok_config.eos_str = StringUtil::ToUtf8(str);
    }

    GetSpecialTokenIds(model, jobj, jdoc);
    return ret;
}

//static
bool ModelReader::LoadGeneratinConfig(TransformerModel &model,
    const string &file_path, const JsonParser &jparser)
{
    GenerationConfig &config = model.generation_config;
    //auto &vocab = model.vocabulary;

    wstring content;
    bool ret = Path::GetFileContent_Text(content, file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to get the file content."));

    JsonDoc jdoc;
    ret = jparser.Parse(jdoc, content);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    JsonObject jobj = jdoc.GetJObject();

    jobj.GetFieldValue(config.user_token_id, L"user_token_id", jdoc);
    jobj.GetFieldValue(config.assistant_token_id, L"assistant_token_id", jdoc);

    jobj.GetFieldValue(config.max_new_tokens, L"max_new_tokens", jdoc);

    jobj.GetFieldValue(config.temperature, L"temperature", jdoc);
    jobj.GetFieldValue(config.top_k, L"top_k", jdoc);
    jobj.GetFieldValue(config.top_p, L"top_p", jdoc);
    jobj.GetFieldValue(config.repetition_penalty, L"repetition_penalty", jdoc);

    GetSpecialTokenIds(model, jobj, jdoc);
    return ret;
}

//static
void ModelReader::GetSpecialTokenIds(TransformerModel &model,
    const JsonObject &jobj, const JsonDoc &jdoc)
{
    auto &vocab = model.vocabulary;

    int token_id = 0;
    if (jobj.GetFieldValue(token_id, L"pad_token_id", jdoc)) {
        model.pad_token_id = token_id;
    }
    if (jobj.GetFieldValue(token_id, L"unk_token_id", jdoc)) {
        vocab.SetUnk(token_id);
    }
    if (jobj.GetFieldValue(token_id, L"bos_token_id", jdoc)) {
        vocab.SetBos(token_id);
    }
    if (jobj.GetFieldValue(token_id, L"eos_token_id", jdoc)) {
        vocab.SetEos(token_id);
    }

    JsonArray jarray;
    if (jobj.GetFieldValue(jarray, L"eos_token_id", jdoc))
    {
        vector<int> eos_ids;
        for (uint32_t idx = 0; idx < jarray.size; idx++)
        {
            token_id = jarray.items[idx].GetIntValue();
            eos_ids.push_back(token_id);
        }

        vocab.SetEos(eos_ids);
    }

    //if (jobj.GetFieldValue(token_id, L"decoder_start_token_id", jdoc)) {
    //    model.decoder_start_token_id = token_id;
    //}
    if (jobj.GetFieldValue(token_id, L"mask_token_id", jdoc)) {
        model.mask_token_id = token_id;
    }
}

bool ModelReader::LoadTokenizer(TransformerModel &model, const ModelSpec &spec,
    const JsonParser &jparser)
{
    bool ret = true;
    string file_path;
    if (spec.tokenizer_files.empty() || spec.tokenizer_files[0].empty()) {
        LogError("tokenizer_file should NOT be empty.");
        return false;
    }

    int base_file_num = spec.hyper_params.output_vocab_size > 0 ? 2 : 1;

    SpecialTokens special_tokens;
    if ((int)spec.tokenizer_files.size() > base_file_num
        && !spec.tokenizer_files[base_file_num].empty())
    {
        file_path = spec.dir + spec.tokenizer_files[base_file_num];
        ret = LoadSpecialTokens(special_tokens, file_path, jparser);
        Macro_RetxFalseIf(!ret, LogError("Error occurred in loading special tokens."));
    }

    for (int file_idx = 0; file_idx < base_file_num; file_idx++)
    {
        bool is_output_vocab = file_idx > 0;
        auto &vocab = is_output_vocab ? model.output_vocabulary : model.vocabulary;

        file_path = spec.dir + spec.tokenizer_files[file_idx];
        Path path(file_path.c_str());
        string token_remap_file = spec.token_remap_file.empty()
            ? "" : spec.dir + spec.token_remap_file;
        string path_ext = path.GetExt();
        bool is_json = String::CaseCmp(path_ext, ".json") == 0;
        bool is_txt = String::CaseCmp(path_ext, ".txt") == 0;

        LogKeyInfo("Loading the vocabulary from %s...", spec.tokenizer_files[file_idx].c_str());
        if (is_txt)
        {
            ret = LoadTokenizer_Txt(vocab, model.spec, file_path, is_output_vocab);
        }
        else if (is_json)
        {
            ret = LoadTokenizer_Json(vocab, model.spec, file_path, jparser,
                special_tokens, spec.token_bytes_mapping, token_remap_file);
        }
        else
        {
            ret = LoadTokenizer_Bin(vocab, model.spec, file_path,
                special_tokens, spec.token_bytes_mapping);
        }

        if (!ret)
        {
            LogError("Failed to load the tokenizer data from %s", file_path.c_str());
            return false;
        }
    }

    if (!special_tokens.unk.empty()) {
        model.vocabulary.SetUnk(special_tokens.unk);
    }
    if (!special_tokens.bos.empty()) {
        model.vocabulary.SetBos(special_tokens.bos);
    }
    if (!special_tokens.eos.empty()) {
        model.vocabulary.SetEos(special_tokens.eos);
    }
    if (!special_tokens.pad.empty()) {
        model.pad_token_id = model.vocabulary.StrToId(special_tokens.pad);
    }
    if (!special_tokens.sep.empty()) {
        model.sep_token_id = model.vocabulary.StrToId(special_tokens.sep);
    }
    if (!special_tokens.mask.empty()) {
        model.mask_token_id = model.vocabulary.StrToId(special_tokens.mask);
    }

    return ret;
}

bool ModelReader::LoadSpecialTokens(SpecialTokens &special_tokens,
    const string &file_path, const JsonParser &jparser)
{
    special_tokens.Clear();

    string content;
    bool ret = Path::GetFileContent_Text(content, file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to get the file content."));

    JsonDoc jdoc;
    ret = jparser.ParseUtf8(jdoc, content);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    JsonObject jobj = jdoc.GetJObject();
    jobj.GetFieldValue(special_tokens.bos, L"bos_token", jdoc);
    jobj.GetFieldValue(special_tokens.eos, L"eos_token", jdoc);
    jobj.GetFieldValue(special_tokens.unk, L"unk_token", jdoc);
    jobj.GetFieldValue(special_tokens.sep, L"sep_token", jdoc);
    jobj.GetFieldValue(special_tokens.pad, L"pad_token", jdoc);

    string str;
    JsonArray jarray;
    jobj.GetFieldValue(jarray, L"additional_special_tokens", jdoc);
    for (uint32_t idx = 0; idx < jarray.size; idx++)
    {
        jarray.items[idx].GetString(str);
        special_tokens.additional.push_back(str);
    }

    return ret;
}

bool ModelReader::LoadVocabJson(StdVocabulary &vocab, const ModelSpec &spec,
    const JsonObject &vocab_obj, const map<int, int> &token_map,
    int token_bytes_mapping)
{
    const auto &hparams = spec.hyper_params;
    int vocab_size = hparams.vocab_size;
    vocab.token_array.resize(vocab_size);

    string key1 = StringUtil::ToUtf8(L"Ġ");
    string key2 = StringUtil::ToUtf8(L"▁");
    string str;
    wstring wstr;
    for (uint32_t idx = 0; idx < vocab_obj.size; idx++)
    {
        const auto &fld = vocab_obj.items[idx];
        if (fld.value.GetType() != JsonValueType::Number) {
            return false;
        }

        int token_id = fld.value.GetIntValue();
        //int original_token_id = token_id;
        if (!token_map.empty())
        {
            auto id_iter = token_map.find(token_id);
            token_id = id_iter == token_map.end() ? -1 : id_iter->second;
        }

        if (token_id >= 0 && token_id < vocab_size)
        {
            auto &token = vocab.token_array[token_id];
            token.id = token_id;
            str.clear();
            vocab_obj.items[idx].name.ToString(str);

            if (token_bytes_mapping != 0)
            {
                StringUtil::Utf8ToWideStr(wstr, str);
                bool is_succ = DecodeTokenStr(str, wstr, token_bytes_mapping);
                if (!is_succ) {
                    LogError("Invalid token %d", token.id);
                    return false;
                }
                //if (token.id == 603) {
                //    LogKeyInfo("%d: %s --> %s", token.id, StringUtil::ToUtf8(wstr).c_str(), str.c_str());
                //}
            }
            //String::ReplaceAll(str, key1, " ");
            String::ReplaceAll(str, key2, " ");

            token.str = str;
            token.score = 1.0f;
            if (token.str.size() == 3 && (uint8_t)token.str[0] == 0xEF
                && (uint8_t)token.str[1] == 0xBF && (uint8_t)token.str[2] == 0xBD)
            {
                //token.score = -1;
                token.type = (int)TokenType::Invalid;
            }

            vocab.str_to_id[token.str] = token.id;
        }
    }

    return true;
}

bool ModelReader::LoadTokenMerges(StdVocabulary &vocab, const ModelSpec &spec,
    const JsonArray &merge_array, int token_bytes_mapping)
{
    const auto &hparams = spec.hyper_params;
    int vocab_size = hparams.vocab_size;
    vocab.token_array.resize(vocab_size);

    string key1 = StringUtil::ToUtf8(L"Ġ");
    string key2 = StringUtil::ToUtf8(L"▁");
    string str, left_str, right_str;
    wstring wstr;
    for (uint32_t idx = 0; idx < merge_array.size; idx++)
    {
        merge_array.items[idx].GetString(str);
        auto pos = str.find(' ');
        if (pos == string::npos) {
            continue;
        }

        left_str = str.substr(0, pos);
        right_str = str.substr(pos + 1);

        if (token_bytes_mapping != 0)
        {
            StringUtil::Utf8ToWideStr(wstr, left_str);
            bool is_succ = DecodeTokenStr(left_str, wstr, token_bytes_mapping);
            if (!is_succ) {
                LogError("Invalid left token in item %d of the merges data", idx);
                return false;
            }

            StringUtil::Utf8ToWideStr(wstr, right_str);
            is_succ = DecodeTokenStr(right_str, wstr, token_bytes_mapping);
            if (!is_succ) {
                LogError("Invalid right token in item %d of the merges data", idx);
                return false;
            }
        }
        //String::ReplaceAll(left_str, key1, " ");
        String::ReplaceAll(left_str, key2, " ");
        //String::ReplaceAll(right_str, key1, " ");
        String::ReplaceAll(right_str, key2, " ");

        vocab.merge_map[make_pair(left_str, right_str)] = idx;
    }

    return true;
}

//static
bool ModelReader::LoadTokenizer_Json(StdVocabulary &vocab, const ModelSpec &spec,
    const string &file_path, const JsonParser &jparser, const SpecialTokens &special_tokens,
    int token_bytes_mapping, const string &token_remap_file)
{
    bool ret = true;
    const auto &hparams = spec.hyper_params;
    vocab.Clear();

    map<int, int> token_map;
    if (!token_remap_file.empty())
    {
        bool is_reverse = false;
        ret = LoadTokenRemapData(token_map, token_remap_file, is_reverse);
        if (!ret)
        {
            LogError("Failed to load the token remap data from %s", token_remap_file.c_str());
            return false;
        }

        int token_map_size = (int)token_map.size();
        if (hparams.vocab_size != token_map_size)
        {
            LogError("The token map should contain %u elements (now %d)",
                hparams.vocab_size, token_map_size);
            return false;
        }
    }

    string content;
    ret = Path::GetFileContent_Text(content, file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to get the file content."));

    JsonDoc jdoc;
    ret = jparser.ParseUtf8(jdoc, content);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    JsonObject jobj = jdoc.GetJObject();

    map<int, string> added_tokens;
    JsonArray jarray;
    bool has_added_tokens = jobj.GetFieldValue(jarray, L"added_tokens", jdoc);
    for (int idx = 0; has_added_tokens && idx < (int)jarray.size; idx++)
    {
        const auto &token_obj = jarray.items[idx].GetJObject();
        int token_id = 0; 
        wstring token_str;
        token_obj.GetFieldValue(token_id, L"id", jdoc);
        token_obj.GetFieldValue(token_str, L"content", jdoc);
        added_tokens[token_id] = StringUtil::ToUtf8(token_str);
    }

    JsonObject model_obj, vocab_obj;
    bool has_model_obj = jobj.GetFieldValue(model_obj, L"model", jdoc);
    bool has_vocab_obj = model_obj.GetFieldValue(vocab_obj, L"vocab", jdoc) && has_model_obj;
    if (!has_vocab_obj && has_added_tokens) {
        LogError("Invalid tokenizer data");
        return false;
    }

    int additional_num = (int)special_tokens.additional.size();
    JsonObject *vocab_obj_ptr = has_vocab_obj ? &vocab_obj : &jobj;
    int vocab_obj_size = (int)vocab_obj_ptr->size;
    int token_num = vocab_obj_size < hparams.vocab_size
        ? vocab_obj_size + additional_num : vocab_obj_size;
    if (vocab_obj_size < (int)(hparams.vocab_size * 0.9f))
    {
        LogWarning("Inconsistent vocabulary size: %d (hparam) vs. (%d + %d)",
            hparams.vocab_size, vocab_obj_size, additional_num);
        return false;
    }

    if (hparams.vocab_size != token_num && token_map.empty())
    {
        LogWarning("Inconsistent vocabulary size: %d (hparam) vs. %d",
            hparams.vocab_size, token_num);
    }

    LoadVocabJson(vocab, spec, *vocab_obj_ptr, token_map, token_bytes_mapping);

    JsonArray merge_array;
    bool has_merges_obj = model_obj.GetFieldValue(merge_array, L"merges", jdoc);
    if (has_merges_obj)
    {
        LoadTokenMerges(vocab, spec, merge_array, token_bytes_mapping);
    }

    if (vocab_obj_size < hparams.vocab_size)
    {
        if (vocab_obj_size + additional_num > vocab.Size()) {
            vocab.token_array.resize(vocab_obj_size + additional_num);
        }

        int new_token_id = vocab_obj_size;
        for (const string &token_str : special_tokens.additional)
        {
            auto &token = vocab.token_array[new_token_id];
            token.id = new_token_id;
            token.str = token_str;
            token.score = 1.0f;
            token.type = (int)TokenType::Control;
            vocab.str_to_id[token.str] = token.id;
            new_token_id++;
        }
    }

    for (auto iter = added_tokens.begin(); iter != added_tokens.end(); iter++)
    {
        int token_id = iter->first;
        if (!token_map.empty())
        {
            auto id_iter = token_map.find(token_id);
            token_id = id_iter == token_map.end() ? -1 : id_iter->second;
        }

        if (token_id >= 0 && token_id < hparams.vocab_size)
        {
            auto &token = vocab.token_array[token_id];
            token.id = token_id;
            token.str = iter->second;
            token.score = 1.0f;
            vocab.str_to_id[token.str] = token.id;
        }
    }

    return ret;
}

bool ModelReader::LoadTokenizer_Txt(StdVocabulary &vocab, const ModelSpec &spec,
    const string &file_path, bool is_output_vocab)
{
    const auto &hparams = spec.hyper_params;
    int vocab_size = is_output_vocab ? hparams.output_vocab_size : hparams.vocab_size;
    vocab.Clear();

    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to open file %s", file_path.c_str()));

    vocab.token_array.resize(vocab_size);

    int token_id = 0;
    string line_str;
    while (reader.GetLine(line_str))
    {
        if (token_id >= vocab_size) {
            break;
        }

        auto &token = vocab.token_array[token_id];
        token.type = (int)TokenType::Normal;
        token.id = token_id;
        token.str = line_str;
        token.score = 1.0f;

        vocab.str_to_id[token.str] = token_id;
        token_id++;
    }

    if (token_id != vocab_size)
    {
        LogWarning("Inconsistent vocabulary size: %d (hparam) vs. (%d)",
            vocab_size, token_id);
        return false;
    }

    return ret;
}

bool ModelReader::LoadTokenizer_Bin(StdVocabulary &vocab, const ModelSpec &spec,
    const string &file_path, const SpecialTokens &special_tokens,
    int token_bytes_mapping) const
{
    (void)special_tokens;
    vocab.Clear();
    const auto &hparams = spec.hyper_params;

    BinaryFileStream file_stream;
    bool ret = file_stream.OpenForRead(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to open file %s", file_path.c_str()));

    IBinaryStream &strm = file_stream;
    vocab.token_array.resize(hparams.vocab_size);

    StringBlockedHeap str_heap;
    char *buf = str_heap.New(4096);
    ret = strm.Read(buf, 1); //read the first byte (0x0A)
    if (!ret) {
        LogError("Invalid vocabulary format");
        return false;
    }

    string key1 = StringUtil::ToUtf8(L"Ġ");
    string key2 = StringUtil::ToUtf8(L"▁");
    int invalid_count = 0;
    wstring wstr;
    for (int token_id = 0; ret && token_id < hparams.vocab_size; token_id++)
    {
        auto &tok = vocab.token_array[token_id];
        tok.id = token_id;

        uint32_t start_offset = (uint32_t)strm.TellRd();
        ret = strm.Read(buf, 2);
        int token_info_len = (int)(uint8_t)buf[0];
        if (buf[1] != 0x0A)
        {
            int nh = (int)(uint8_t)buf[1];
            token_info_len = (token_info_len & 0x7F) | (nh << 7);
            ret = strm.Read(buf, 1);
            if (buf[0] != 0x0A)
            {
                LogError("Invalid vocabulary format for token %d (start_offset: %u)",
                    token_id, start_offset);
                return false;
            }
        }

        ret = ret && strm.Read(buf, token_info_len);
        bool is_line_feed = (uint8_t)buf[token_info_len - 1] == 0x0A; //LF
        if (!ret || (token_id + 1 < hparams.vocab_size && !is_line_feed))
        {
            LogError("Invalid vocabulary format for token %d (start_offset: %u)",
                token_id, start_offset);
            return false;
        }

        int token_str_start = 1;
        int token_len = (uint8_t)buf[0];
        if ((token_len & 0x80) != 0)
        {
            int nh = (uint8_t)buf[1];
            token_len = (token_len & 0x7F) | (nh << 7);
            token_str_start = 2;
        }

        //token_str_len (1 or 2), token_str, score_fld (1), score (4)
        if (token_info_len < token_str_start + token_len + 1 + 4)
        {
            LogError("Invalid token line format (token_id: %d, len: %d, line_len: %d)",
                token_id, token_len, token_info_len);
            return false;
        }

        tok.str.assign(buf + token_str_start, token_len);

        int offset = token_str_start + token_len;
        while (offset + 1 < token_info_len)
        {
            uint8_t field_ch = (uint8_t)buf[offset];
            switch (field_ch)
            {
            case 0x15:
                if (offset + 4 < token_info_len)
                { //4: sizeof(float)
                    memcpy(&tok.score, buf + offset + 1, 4);
                    offset += (1 + 4);
                }
                else
                {
                    LogError("Invalid token line format (token_id: %d, offset: %d)",
                        token_id, offset);
                    return false;
                }
                break;
            case 0x18:
                if (offset + 1 < token_info_len)
                {
                    tok.type = 0;
                    memcpy(&tok.type, buf + offset + 1, 1);
                    offset += (1 + 1);
                }
                else
                {
                    LogError("Invalid token line format (token_id: %d, offset: %d)",
                        token_id, offset);
                    return false;
                }
                break;
            default:
                LogError("Unknown token field (token_id: %d, start_offset: %u, field: 0x%x)",
                    token_id, start_offset, field_ch);
                return false;
            }
        }

        if (token_bytes_mapping != 0)
        {
            StringUtil::Utf8ToWideStr(wstr, tok.str);
            bool is_succ = DecodeTokenStr(tok.str, wstr, token_bytes_mapping);
            if (!is_succ) {
                LogError("Invalid token %d", tok.id);
                return false;
            }
        }
        //String::ReplaceAll(tok.str, key1, " ");
        String::ReplaceAll(tok.str, key2, " ");

        if (tok.str.size() == 3 && (uint8_t)tok.str[0] == 0xEF
            && (uint8_t)tok.str[1] == 0xBF && (uint8_t)tok.str[2] == 0xBD)
        {
            //tok.score = -1;
            tok.type = (int)TokenType::Invalid;
            invalid_count++;
        }

        vocab.str_to_id[tok.str] = token_id;
    }

    if (invalid_count > 0) {
        LogWarning("Number of invalid tokens: %d", invalid_count);
    }

    return ret;
}

bool ModelReader::ReadVocabulary_Std(TransformerModel &model,
    IBinaryStream &strm, bool has_score) const
{
    bool ret = true;
    model.vocabulary.Clear();
    const auto &hparams = model.spec.hyper_params;

    model.vocabulary.token_array.resize(hparams.vocab_size);

    const uint32_t max_token_len = 4095;
    uint32_t token_len = 0;
    int invalid_count = 0;
    char *token_buf = new char[max_token_len + 1];
    for (int token_id = 0; ret && token_id < hparams.vocab_size; token_id++)
    {
        ret = strm.Read(token_len);
        if (token_len > max_token_len)
        {
            LogError("Token %d is too long: %d", token_id, token_len);
            delete[] token_buf;
            return false;
        }

        ret = ret && strm.Read(token_buf, token_len);
        //token_buf[token_len] = '\0';
        string token_str(token_buf, token_len);

        int token_type = 0;
        float score = 0.0f;
        if (ret && has_score)
        {
            ret = strm.Read(score);
            //if (score < -0.001f) {
            //    score = 0.0f;
            //}
        }

        if (token_str.size() == 3 && (uint8_t)token_str[0] == 0xEF
            && (uint8_t)token_str[1] == 0xBF && (uint8_t)token_str[2] == 0xBD)
        {
            //score = -1;
            token_type = (int)TokenType::Invalid;
            invalid_count++;
        }

        model.vocabulary.str_to_id[token_str] = token_id;
        auto &tok = model.vocabulary.token_array[token_id];
        tok.id = token_id;
        tok.str = std::move(token_str);
        tok.score = score;
        tok.type = token_type;

        //const auto *token_ptr = model.vocabulary.Find(token_str.c_str());
        //if (token_ptr != nullptr)
        //{
        //    LogError("Duplicate token: \"%s\" (length: %d, prev: %s)",
        //        token_str.c_str(), token_len, token_ptr->str);
        //}
        //model.vocabulary.AddItem(token_str, score);
    }
    delete[] token_buf;

    if (invalid_count > 0) {
        LogWarning("Number of tokens with invalid scores: %d", invalid_count);
    }

    int vocab_dict_size = (int)model.vocabulary.token_array.size();
    if (ret && hparams.vocab_size != vocab_dict_size)
    {
        LogError("Duplicate tokens are detected (%d vs. %d)",
            hparams.vocab_size, vocab_dict_size);
        return false;
    }

    return ret;
}

bool ModelReader::ReadVocabulary_Format2(TransformerModel &model, IBinaryStream &strm) const
{
    model.vocabulary.Clear();
    const auto &hparams = model.spec.hyper_params;

    model.vocabulary.token_array.resize(hparams.vocab_size);

    uint32_t max_token_len = 4095;
    bool ret = strm.Read(max_token_len);
    Macro_RetFalseIf(!ret);

    uint32_t token_len = 0;
    int invalid_count = 0;
    char *token_buf = new char[max_token_len + 1];
    for (int token_id = 0; ret && token_id < hparams.vocab_size; token_id++)
    {
        auto &tok = model.vocabulary.token_array[token_id];
        tok.id = token_id;

        float score = 0;
        ret = ret && strm.Read(score);
        //if (score < -0.001f) {
        //    score = 0.0f;
        //}

        ret = ret && strm.Read(token_len);
        if (token_len > max_token_len)
        {
            LogError("Token %d is too long: %d", token_id, token_len);
            delete[] token_buf;
            return false;
        }

        ret = ret && strm.Read(token_buf, token_len);
        //token_buf[token_len] = '\0';
        tok.str.assign(token_buf, token_len);

        if (tok.str.size() == 3 && (uint8_t)tok.str[0] == 0xEF
            && (uint8_t)tok.str[1] == 0xBF && (uint8_t)tok.str[2] == 0xBD)
        {
            //score = -1;
            tok.type = (int)TokenType::Invalid;
            invalid_count++;
        }

        tok.score = score;
        model.vocabulary.str_to_id[tok.str] = tok.id;
    }
    delete[] token_buf;

    if (invalid_count > 0) {
        LogWarning("Number of tokens with invalid scores: %d", invalid_count);
    }

    return ret;
}

//static
bool ModelReader::LoadTokenRemapData(map<int, int> &token_map,
    const string &file_path, bool is_reverse)
{
    BinaryFileStream strm;
    bool ret = strm.OpenForRead(file_path);
    Macro_RetFalseIf(!ret);

    PickleReader reader;
    reader.Init();

    strm.SeekRd(64);
    vector<PickleOperation> opr_list;
    ret = reader.Read(opr_list, strm);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the pickle operations"));

    int int_count = 0, num = 0;
    int opr_num = (int)opr_list.size();
    for (int idx = 0; idx < opr_num; idx++)
    {
        const auto &opr = opr_list[idx];
        //const PickleOperation *prev_opr = idx > 0 ? &opr_list[idx - 1] : nullptr;

        if (opr.cat == PickleOpcat::PushInt)
        {
            int_count++;
            if (int_count % 2 == 1)
            {
                num = opr.arg1.nv;
            }
            else
            {
                if (is_reverse) {
                    token_map[opr.arg1.nv] = num;
                }
                else {
                    token_map[num] = opr.arg1.nv;
                }
            }
        }
    }

    strm.Close();
    return ret;
}

//static
bool ModelReader::LoadWeightMap(map<string, string> &weight_map,
    const string &file_path, JsonParser &jparser)
{
    wstring content;
    bool ret = Path::GetFileContent_Text(content, file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to get the file content."));

    JsonDoc jdoc;
    ret = jparser.Parse(jdoc, content);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    JsonObject jobj = jdoc.GetJObject();
    JsonObject weight_map_obj;
    ret = jobj.GetFieldValue(weight_map_obj, L"weight_map", jdoc);
    Macro_RetxFalseIf(!ret, LogError("Failed to get the weight_map object"));

    wstring str, value_str;
    for (int field_idx = 0; field_idx < (int)weight_map_obj.size; field_idx++)
    {
        const auto &field = weight_map_obj.items[field_idx];
        field.name.ToString(str);
        field.value.GetString(value_str);
        weight_map[StringUtil::ToUtf8(str)] = StringUtil::ToUtf8(value_str);
    }

    return ret;
}

//static
void ModelReader::GetFileSetFromWeightMap(vector<string> &file_list,
    const map<string, string> &weight_map)
{
    file_list.clear();

    set<string> file_set;
    for (auto iter = weight_map.begin(); iter != weight_map.end(); iter++)
    {
        file_set.insert(iter->second);
    }

    for (const string &file_name : file_set) {
        file_list.push_back(file_name);
    }
}

void ModelReader::InitTokenByteMap()
{
    auto &char_to_byte = token_char_to_byte_map_;

    //bytes: [0x21, 0x7E]
    for (int cid = 0x21; cid <= 0x7E; cid++) {
        char_to_byte[cid] = (uint8_t)cid;
    }

    //bytes: [0xA1, 0xFF]
    char_to_byte[0x143] = 0xAD;
    for (int cid = 0xA1; cid <= 0xFF; cid++)
    {
        if (cid != 0xAD) {
            char_to_byte[cid] = (uint8_t)cid;
        }
    }

    //bytes: [0x00, 0x20]
    for (int cid = 0x100; cid <= 0x120; cid++) {
        char_to_byte[cid] = (uint8_t)(cid & 0xFF);
    }

    //bytes: [0x7F, 0xA0]
    for (int cid = 0x121; cid <= 0x142; cid++) {
        char_to_byte[cid] = (uint8_t)(cid - 0xA2);
    }
}

bool ModelReader::DecodeTokenStr(string &target, const wstring &src, int alg) const
{
    (void)alg;
    target.clear();
    const auto &char_to_byte = token_char_to_byte_map_;

    for (wchar_t ch : src)
    {
        auto iter = char_to_byte.find(ch);
        if (iter == char_to_byte.end()) {
            return false;
        }

        target += (char)iter->second;
    }

    return true;
}

//static
HostTensor* ModelReader::ReadTensor(ElementType data_type, int cx, int cy,
    IBinaryStream &strm, TransformerContext &ctx)
{
    HostTensor *tensor = ctx.host_tensor_heap.New(1);
    tensor->New(data_type, cx, cy);
    int element_num = cy < 1 ? cx : cx * cy;
    /*ggml_tensor *tensor = nullptr;
    ggml_type ggml_data_type = ToGgmlType(data_type);

    int element_num = 0;
    if (cy < 1)
    {
        tensor = ggml_new_tensor_1d(ctx.ggml_ctx, ggml_data_type, cx);
        element_num = cx;
    }
    else
    {
        tensor = ggml_new_tensor_2d(ctx.ggml_ctx, ggml_data_type, cx, cy);
        element_num = cx * cy;
    }*/

    if (tensor == nullptr) {
        LogError("Error occurred in creating a host tensor (cx: %d, cy: %d)", cx, cy);
        return nullptr;
    }

    int byte_num = (int)TensorCommon::ByteCount(data_type, element_num);
    //tensor->data = ctx.byte_heap.New(byte_num);
    bool ret = strm.Read((char*)tensor->data, byte_num);
    if (!ret) {
        LogError("Failed to read tensor data (%d, %d)", cx, cy);
    }

    return ret ? tensor : nullptr;
}

//static
bool ModelReader::ReadAndTransTensor(HostTensor &tensor, HostTensor *mem_tensor,
    TensorSpec &spec, TransformerContext &ctx, IBinaryStream &strm)
{
    bool ret = true;
    int byte_count = (int)TensorCommon::ByteCount((ElementType)spec.data_type, spec.size);

    ElementType src_data_type = (ElementType)spec.data_type;
    ElementType target_data_type = TensorCommon::IsQuantType(src_data_type)
        ? src_data_type : ElementType::F16;

    spec.data_type = (int)target_data_type;
    tensor.New(target_data_type, spec.dims, spec.ne);

    int data_source_size = mem_tensor == nullptr ? 0 : (int)mem_tensor->size;
    int target_byte_count = (int)TensorCommon::ByteCount(target_data_type, tensor.size);
    bool use_mem_tensor = tensor.size < spec.size && mem_tensor != nullptr;

    if (src_data_type == target_data_type)
    {
        if (use_mem_tensor)
        {
            if (mem_tensor->size > 0)
            {
                const uint8_t *src_data = (const uint8_t*)mem_tensor->data;
                memcpy(tensor.data, src_data + spec.offset_in_data_source, target_byte_count);
            }
            else
            {
                inferflow_fp16 *data_array = (inferflow_fp16*)ctx.fp16_heap.New(spec.size);
                strm.Read((char*)data_array, byte_count);
                memcpy(tensor.data, ((const char*)data_array) + spec.offset_in_data_source,
                    target_byte_count);

                mem_tensor->Set(src_data_type, data_array, spec.size, 0, 0, false);
            }
        }
        else
        {
            ret = strm.Read((char*)tensor.data, byte_count);
        }

        return ret;
    }

    //transform bf16 to fp16
    if (src_data_type == ElementType::BF16)
    {
        ctx.fp16_heap.Clear(1);

        uint16_t *data_array = nullptr;
        if (data_source_size > 0)
        {
            data_array = (uint16_t*)mem_tensor->data;
        }
        else
        {
            data_array = (uint16_t*)ctx.fp16_heap.New(spec.size);
            strm.Read((char*)data_array, byte_count);

            if (use_mem_tensor) {
                mem_tensor->Set(src_data_type, data_array, spec.size, 0, 0, false);
            }
        }

        inferflow_fp16 *target_array = tensor.data_f16();
        const uint16_t *src_array = data_array + spec.offset_in_data_source;
        uint32_t tensor_size = min((uint32_t)tensor.size, spec.size);
        for (uint32_t idx = 0; idx < tensor_size; idx++)
        {
#ifdef __GNUC__
#           pragma GCC diagnostic push
#           pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

            uint32_t u32_value = (uint32_t)src_array[idx];
            u32_value <<= 16;
            inferflow_fp16 f16_value = (inferflow_fp16)*(const float*)&u32_value;
            target_array[idx] = f16_value;

#ifdef __GNUC__
#           pragma GCC diagnostic pop
#endif
        }
    }
    else if (src_data_type == ElementType::F32)
    {
        ctx.float_heap.Clear(1);

        float *data_array = nullptr;
        if (data_source_size > 0)
        {
            data_array = (float*)mem_tensor->data;
        }
        else
        {
            data_array = ctx.float_heap.New(spec.size);
            strm.Read((char*)data_array, byte_count);

            if (use_mem_tensor) {
                mem_tensor->Set(src_data_type, data_array, spec.size, 0, 0, false);
            }
        }

        const float *src_array = data_array + spec.offset_in_data_source;
        inferflow_fp16 *target_array = tensor.data_f16();
        uint32_t tensor_size = min((uint32_t)tensor.size, spec.size);
        for (uint32_t idx = 0; idx < tensor_size; idx++)
        {
            inferflow_fp16 f16_value = (inferflow_fp16)src_array[idx];
            target_array[idx] = f16_value;
        }
    }
    else
    {
        LogError("Data type %d is not supported so far.", spec.data_type);
        ret = false;
    }

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Std format

//static
bool ModelReader::LoadModel_Std(TransformerModel &model, TransformerContext &ctx,
    const ModelSpec &spec, bool is_study_mode)
{
    (void)model; (void)ctx; (void)is_study_mode;
    string model_file = spec.model_files.empty() ? "" : spec.model_files[0];
    string file_path = spec.dir + model_file;
    BinaryFileStream strm;
    bool ret = strm.OpenForRead(file_path);
    if (!ret) {
        LogError("Failed to open file %s", file_path.c_str());
        return false;
    }

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Pickle format
////////////////////////////////////////////////////////////////////////////////

bool ModelReader::LoadModel_Pickle(TransformerModel &model, TransformerContext &ctx,
    const ModelSpec &spec, const vector<string> &model_file_list, bool is_study_mode)
{
    PickleReader reader;
    bool ret = reader.Init();
    Macro_RetxFalseIf(!ret, LogError("Failed to initialize the pickle reader"));

    string model_file = spec.model_files.empty() ? "" : spec.model_files[0];
    int file_num = (int)model_file_list.size();
    if (file_num > 1) {
        LogKeyInfo("Loading tensors from %d files...", file_num);
    }
    else {
        LogKeyInfo("Loading tensors...");
    }

    ModelPartition model_partition;
    NetworkBuilder::GetDeviceAssignments(model_partition.decoder_assignments,
        model.spec, false, model.spec.decoder_cpu_layer_count);

    int builder_count = (int)model_partition.decoder_assignments.size();
    PtrVector<NetworkBuilder> builder_list;
    for (int builder_id = 0; builder_id < builder_count; builder_id++)
    {
        auto *builder = new NetworkBuilder;
        builder->Init(*network_builder_);
        builder_list.push_back(builder);
    }

#if defined(USE_CUDA)
    int default_device_id = CudaUtil::GetDevice();
#endif //USE_CUDA

    int start_idx = file_num == 0 ? -1 : 0;
    uint32_t proc_num = 0;
    TaskMonitor tm(10);
    for (int file_idx = start_idx; ret && file_idx < file_num; file_idx++)
    {
        const string &file_name = file_idx >= 0 ? model_file_list[file_idx]
            : model_file;
        string file_path = spec.dir + file_name;

        BinaryFileStream strm;
        strm.SetRdBufferSize(4096);
        ret = strm.OpenForRead(file_path);
        Macro_RetxFalseIf(!ret, LogError("Failed to open the model file: %s", file_name.c_str()));

        reader.Clear();
        map<string, StrAndCount> section_to_tensor_name_map;
        //int base_idx = model.tensor_spec_table.Size();
        ret = Pickle_ReadHeader(model, section_to_tensor_name_map, strm, reader, file_idx);
        Macro_RetxFalseIf(!ret, LogError("Failed to read the header"));

        for (auto pre_iter = model.spec.tensor_name_pre_map.begin();
            pre_iter != model.spec.tensor_name_pre_map.end(); pre_iter++)
        {
            int src_tensor_idx = model.tensor_spec_table.GetIndex(pre_iter->first);
            if (src_tensor_idx >= 0 && src_tensor_idx < model.tensor_spec_table.Size())
            {
                const string &new_tensor_name = pre_iter->second;
                model.tensor_spec_table.tensor_array[src_tensor_idx].name = new_tensor_name;
            }
        }

        const int max_skip_count = 6;
        int skip_count = 0, local_proc_num = 0;
        //auto &tensor_array = model.tensor_spec_table.tensor_array;
        int section_num = (int)section_to_tensor_name_map.size();
        //LogKeyInfo("Section names: %d", section_num);
        //int tensor_num = (int)tensor_array.size();
        //for (int idx = 0; idx < tensor_num - base_idx; idx++)
        for (int idx = 0; ret && local_proc_num < section_num; idx++)
        {
            int read_count = Pickle_ReadTensor(model, ctx, strm, reader, file_idx,
                section_to_tensor_name_map, model_partition, builder_list,
                is_study_mode);
            if (read_count <= -2)
            {
                ret = idx >= section_num;
                break;
            }

            if (read_count <= 0)
            {
                skip_count++;
                if (file_num > 1) {
                    LogWarning("In file %d: Skip block %d of %d", file_idx, idx, section_num);
                }
                else {
                    LogWarning("Skip block %d of %d", idx, section_num);
                }

                if (skip_count > max_skip_count)
                {
                    LogError("Error occurred in reading a tensor from section %d", idx);
                    ret = false;
                }
                continue;
            }

            local_proc_num++;
            proc_num++;
            tm.Progress(proc_num);
        }
    }
    tm.End();

    //vector<string> tensors_with_missing_data;
    //int tensor_num = model.tensor_spec_table.Size();
    //for (int tensor_idx = 0; tensor_idx < tensor_num; tensor_idx++)
    //{
    //    const auto &tensor = model.tensor_spec_table.tensor_array[tensor_idx];
    //    if (tensor.data_source == -1 && tensor.host_tensor == nullptr) {
    //        tensors_with_missing_data.push_back(tensor.name);
    //    }
    //}

    //if (!tensors_with_missing_data.empty())
    //{
    //    int missing_num = (int)tensors_with_missing_data.size();
    //    int top_k = min(5, missing_num);
    //    LogKeyInfo("%d of %d tensor(s) with missing data:", top_k, missing_num);
    //    for (int tensor_idx = 0; tensor_idx < top_k; tensor_idx++)
    //    {
    //        LogKeyInfo("    %s", tensors_with_missing_data[tensor_idx].c_str());
    //    }
    //}

#if defined(USE_CUDA)
    CudaUtil::SetDevice(default_device_id);
#endif //USE_CUDA
    return ret;
}

//static
bool ModelReader::Pickle_ReadHeader(TransformerModel &model,
    map<string, StrAndCount> &section_to_tensor_name_map,
    IBinaryStream &strm, PickleReader &reader, int file_idx)
{
    section_to_tensor_name_map.clear();

    string section_name;
    bool ret = reader.ReadSectionHeader(section_name, strm, false);
    Macro_RetxFalseIf(!ret, LogError("File %d: Invalid section header", file_idx));

    vector<PickleOperation> opr_list;
    ret = reader.Read(opr_list, strm, file_idx);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the pickle operations")); 

    ElementTypeMap data_type_map;
    data_type_map["FloatStorage"] = ElementType::F32;
    data_type_map["HalfStorage"] = ElementType::F16;
    data_type_map["BFloat16Storage"] = ElementType::BF16;

    int warning_count = 0;
    TensorSpec tensor;
    string storage, tensor_section_name;
    ElementType data_type = ElementType::F16;
    ElementType local_data_type = ElementType::F16;
    map<int, string> section_map; //key-id to section sid
    map<int, ElementType> pushed_data_types;
    bool has_collections = false, need_storage = false;
    int put_get_count_after_marks = 0;
    int opr_num = (int)opr_list.size();
    for (int idx = 0; idx < opr_num; idx++)
    {
        const auto &opr = opr_list[idx];
        const PickleOperation *next_opr = idx + 1 < opr_num ? &opr_list[idx + 1] : nullptr;
        const PickleOperation *prev_opr = idx > 0 ? &opr_list[idx - 1] : nullptr;

        if (opr.cat == PickleOpcat::Mark && prev_opr != nullptr
            && prev_opr->cat == PickleOpcat::Mark)
        {
            put_get_count_after_marks = 0;
        }

        if (opr.cat == PickleOpcat::Put || opr.cat == PickleOpcat::Get) {
            put_get_count_after_marks++;
        }

        //if (has_collections && opr.cat == PickleOpcat::PushString)
        if (opr.cat == PickleOpcat::PushString)
        {
            if (String::CaseCmp(opr.arg1.str, "_metadata") == 0)
            {
                break;
            }

            if (tensor.name.empty())
            {
                tensor.name = opr.arg1.str;
            }
            else if (strlen(opr.arg1.str) > 0 && opr.arg1.str[0] >= '0' && opr.arg1.str[0] <= '9')
            {
                tensor_section_name = opr.arg1.str;

                if (put_get_count_after_marks == 2 && next_opr != nullptr
                    && next_opr->cat == PickleOpcat::Put)
                {
                    section_map[next_opr->arg1.nv] = opr.arg1.str;
                }
            }

            if (need_storage) {
                storage = opr.arg1.str;
            }

            need_storage = String::CaseCmp(opr.arg1.str, "storage") == 0;
        }

        if (opr.code == PickleOpcode::Global && opr.arg_num == 2)
        {
            if (String::CaseCmp(opr.arg1.str, "collections") == 0)
            {
                has_collections = true;
            }
            else if (String::CaseCmp(opr.arg1.str, "torch") == 0)
            {
                auto iter = data_type_map.find(opr.arg2.str);
                if (iter != data_type_map.end())
                {
                    local_data_type = iter->second;
                    if (need_storage) {
                        data_type = iter->second;
                    }
                    //LogKeyInfo("tensor: %s, data_type: %d, local_data_type: %d",
                    //    tensor.name.c_str(), data_type, local_data_type);

                    if (next_opr != nullptr && next_opr->cat == PickleOpcat::Put) {
                        pushed_data_types[next_opr->arg1.nv] = local_data_type;
                    }
                }
            }

            need_storage = false;
        }

        if (opr.cat == PickleOpcat::Get)
        {
            auto iter_find = pushed_data_types.find(opr.arg1.nv);
            if (iter_find != pushed_data_types.end()) {
                local_data_type = iter_find->second;
            }

            if (put_get_count_after_marks == 3 && opr.cat == PickleOpcat::Get)
            {
                auto section_iter = section_map.find(opr.arg1.nv);
                if (section_iter != section_map.end()) {
                    tensor_section_name = section_iter->second;
                }
            }
        }

        if (opr.cat == PickleOpcat::TupleX && prev_opr != nullptr
            && prev_opr->cat == PickleOpcat::PushInt)
        {
            if (tensor.size == 0)
            {
                tensor.size = prev_opr->arg1.nv;
            }
            else if (tensor.dims == 0)
            {
                tensor.dims = 1;
                tensor.ne[0] = prev_opr->arg1.nv;
                if (idx >= 2 && opr.code == PickleOpcode::Tuple2)
                {
                    const PickleOperation &prev2 = opr_list[idx - 2];
                    tensor.ne[1] = prev2.arg1.nv;
                    tensor.dims++;
                    if (idx >= 3)
                    {
                        const PickleOperation &prev3 = opr_list[idx - 3];
                        if (prev3.cat == PickleOpcat::PushInt && prev3.arg_num >= 1)
                        {
                            tensor.offset_in_data_source = prev3.arg1.nv;
                            tensor.has_start_offset = true;
                        }
                    }
                }
            }
        }

        if (opr.code == PickleOpcode::EmptyTuple || opr.code == PickleOpcode::NewFalse)
        {
            if (tensor.size > 0)
            {
                tensor.data_type = (int)local_data_type;

                uint32_t tensor_size = 1;
                for (int dim_idx = 0; dim_idx < tensor.dims; dim_idx++) {
                    tensor_size *= tensor.ne[dim_idx];
                }

                if (tensor_size != tensor.size && !tensor.has_start_offset
                    || tensor_size > tensor.size)
                {
                    //LogKeyInfo("The size of %s is adjusted from %d to %d",
                    //    tensor.name.c_str(), tensor.size, tensor_size);
                    //tensor.size = tensor_size;
                    warning_count++;
                    if (warning_count <= 3)
                    {
                        LogWarning("Inconsistent tensor size (%s): %u vs. %u",
                            tensor.name.c_str(), tensor.size, tensor_size);
                    }
                }

                int new_tensor_idx = model.tensor_spec_table.Add(tensor);
                if (!tensor_section_name.empty() && new_tensor_idx >= 0)
                {
                    auto &new_tensor = model.tensor_spec_table.tensor_array[new_tensor_idx];
                    Pickle_HandleSectionName(section_to_tensor_name_map, new_tensor,
                        model, tensor_section_name);
                    tensor_section_name.clear();
                }
            }

            //LogKeyInfo("tensor: %s, local_data_type: %d",
            //    tensor.name.c_str(), local_data_type);
            tensor.Clear();
            local_data_type = data_type;
        }
    }

    if (warning_count > 0) {
        LogKeyInfo("Number of warnings in this file: %d", warning_count);
    }
    (void)has_collections;
    return ret;
}

//static
void ModelReader::Pickle_HandleSectionName(map<string, StrAndCount> &section_to_tensor_name_map,
    TensorSpec &tensor, const TransformerModel &model, const string &section_name)
{
    int tensor_id = model.tensor_spec_table.GetIndex(tensor.name);
    auto section_iter = section_to_tensor_name_map.find(section_name);
    if (section_iter == section_to_tensor_name_map.end())
    {
        StrAndCount snc(tensor.name, 1);
        snc.tensor_id_arr[0] = tensor_id;
        section_to_tensor_name_map[section_name] = snc;
    }
    else
    {
        auto &snc = section_iter->second;
        snc.tensor_id_arr[snc.count] = tensor_id;
        snc.count++;
        tensor.data_source = model.tensor_spec_table.GetIndex(snc.str);
    }
}

int ModelReader::Pickle_ReadTensor(TransformerModel &model, TransformerContext &ctx,
    IBinaryStream &strm, PickleReader &reader, int file_idx,
    const map<string, StrAndCount> &section_to_tensor_name_map,
    const ModelPartition &model_partition, const PtrVector<NetworkBuilder> &builder_list,
    bool is_study_mode)
{
    (void)model_partition; (void)builder_list;
    string section_name;
    HostTensorMap &tensor_map = model.std_network.tensor_map;
    TensorSpecTable &tensor_table = model.tensor_spec_table;
    bool ret = reader.ReadSectionHeader(section_name, strm, true);
    if (!ret)
    {
        LogWarning("Failed to read the section header");
        return -2;
    }

    auto pos = section_name.rfind('/');
    if (pos == string::npos || pos + 1 >= section_name.size())
    {
        LogWarning("Invalid section name: %s", section_name.c_str());
        return -1;
    }

    string section_sid = section_name.substr(pos + 1);

    auto iter_section = section_to_tensor_name_map.find(section_sid);
    if (iter_section == section_to_tensor_name_map.end())
    {
        if (section_sid.find_first_of("_.") != string::npos)
        {
            LogWarning("Canot find section %s in the section_to_tensor_name map",
                section_sid.c_str());
        }
        return -1;
    }

    int tensor_count = iter_section->second.count;
    const auto &tensor_id_arr = iter_section->second.tensor_id_arr;
    const string &tensor_name = iter_section->second.str;
    int tensor_idx = tensor_table.GetIndex(tensor_name);
    if (tensor_idx < 0)
    {
        LogWarning("Cannot find tensor %s", tensor_name.c_str());
        return -1;
    }

    TensorSpec &tensor_spec = tensor_table.tensor_array[tensor_idx];
    //auto &spec = tensor_specs[tensor_idx + base_idx];
    if (is_study_mode)
    {
        //LogKeyInfo("Tensor index: %d, tensor: %s, data_type: %d, size: %d",
        //    tensor_idx, spec.name.c_str(), spec.data_type, spec.size);
    }

    ElementType src_data_type = (ElementType)tensor_spec.data_type;
    int byte_count = (int)TensorCommon::ByteCount(src_data_type, tensor_spec.size);

    int tensor_size = 1;
    for (int idx = 0; idx < tensor_spec.dims; idx++) {
        tensor_size *= tensor_spec.ne[idx];
    }

    HostTensor mem_tensor(false);
    HostTensor *mem_tensor_ptr = tensor_size < (int)tensor_spec.size ? &mem_tensor : nullptr;
    if (mem_tensor_ptr == nullptr && tensor_count > 1) {
        tensor_count = 1;
    }

    for (int target_idx = 0; target_idx < tensor_count; target_idx++)
    {
        int target_tensor_id = tensor_id_arr[target_idx];
        TensorSpec &target_tensor_spec = tensor_table.tensor_array[target_tensor_id];

        bool has_device_tensor = false;
#if defined(USE_CUDA)
        if (model.spec.is_eager_device_building)
        {
            string norm_tensor_name;
            model.network.TransTensorName(norm_tensor_name, target_tensor_spec.name);
            TensorNameInfo tni;
            bool is_succ = false;
            if (!norm_tensor_name.empty())
            {
                is_succ = NetworkBuilder::ParseTensorName(tni, norm_tensor_name,
                    model.std_network.device_net, model.spec);
            }

            if (is_succ && tni.layer_id >= model.spec.decoder_cpu_layer_count)
            {
                if (tni.expert_id < model.spec.hyper_params.in_use_experts)
                {
                    HostTensor cpu_tensor;
                    ret = ReadAndTransTensor(cpu_tensor, mem_tensor_ptr,
                        target_tensor_spec, ctx, strm);
                    Macro_RetFalseIf(!ret);

                    int device_id = NetworkBuilder::GetDeviceByLayer(model_partition,
                        tni.layer_id, true);
                    CudaUtil::SetDevice(device_id);

                    int la_idx = NetworkBuilder::GetDeviceGroupIndex(model_partition,
                        tni.layer_id, true);
                    auto *net_builder = builder_list[la_idx];
                    net_builder->BuildDeviceTensor(model.std_network.device_net,
                        cpu_tensor, tni, model.spec);
                }
                has_device_tensor = true;
            }
        }
#endif //USE_CUDA

        if (!has_device_tensor)
        {
            HostTensor *host_tensor = ctx.host_tensor_heap.New(1);
            ret = ReadAndTransTensor(*host_tensor, mem_tensor_ptr,
                target_tensor_spec, ctx, strm);
            Macro_RetIf(-1, !ret);

            tensor_map[tensor_idx] = host_tensor;

            //if (is_study_mode && spec.data_type == (int)ElementType::F16 && spec.size >= 2)
            //{
            //    const half *half_array = (half*)spec.host_tensor->data;
            //    float v1 = (float)half_array[0];
            //    float v2 = (float)half_array[1];
            //    LogKeyInfo("name: %s, v1: %f, v2: %f", spec.name.c_str(), v1, v2);
            //}
        }
    }

    char key2[] = { (char)0x50, (char)0x4B, (char)0x07, (char)0x08 };
    int key2_len = 4;

    uint32_t signature = 0;
    uint64_t signature_offset = strm.TellRd();
    ret = strm.Read(signature);
    if (!ret || memcmp(&signature, key2, key2_len) != 0)
    {
        LogWarning("Invalid section-end signature: %x (file: %d, offset: %I64u, section name: %s, tensor_name: %s, tensor_size: %u)",
            signature, file_idx, signature_offset, section_name.c_str(), tensor_spec.name.c_str(), tensor_spec.size);
        return -1;
        //ret = reader.SeekByKey(strm, key2, key2_len);
        //if (!ret)
        //{
        //    LogError("Failed to find the section end (file: %d, offset: %I64u, section name: %s)",
        //        file_idx, signature_offset, section_name_buf);
        //}
    }

    if (ret)
    {
        uint32_t num1 = 0, num2 = 0;
        ret = strm.SeekRd(strm.TellRd() + 4);
        ret = ret && strm.Read(num1);
        ret = ret && strm.Read(num2);
        if (num1 != (uint32_t)byte_count && num2 != (uint32_t)byte_count)
        {
            //LogError("%s: %u vs. %u vs. %u. Tensor size: %d (%d, %d, %d), data_type = %d",
            //    "Inconsistent byte count", num1, num2, (uint32_t)byte_count, tensor_spec.size,
            //    tensor_spec.ne[0], tensor_spec.ne[1], tensor_spec.ne[2],
            //    tensor_spec.data_type);
            //return false;
        }

        if (is_study_mode) {
            //LogKeyInfo("byte_num: %u", num1);
        }
        tensor_spec.flag++;
    }

    return ret ? tensor_count : -1;
}

////////////////////////////////////////////////////////////////////////////////
// Safetensors format
////////////////////////////////////////////////////////////////////////////////

bool ModelReader::LoadModel_Safetensors(TransformerModel &model, TransformerContext &ctx,
    const ModelSpec &spec, const vector<string> &model_file_list, JsonParser &jparser,
    bool is_study_mode)
{
    (void)is_study_mode;
    bool ret = true;

    string model_file = spec.model_files.empty() ? "" : spec.model_files[0];
    int file_num = (int)model_file_list.size();
    if (file_num > 1) {
        LogKeyInfo("Loading tensors from %d files...", file_num);
    }
    else {
        LogKeyInfo("Loading tensors...");
    }

    TaskMonitor tm(10);
    int start_idx = file_num == 0 ? -1 : 0;
    for (int file_idx = start_idx; file_idx < file_num; file_idx++)
    {
        const string &file_name = file_idx >= 0 ? model_file_list[file_idx]
            : model_file;
        string file_path = spec.dir + file_name;
        //LogKeyInfo("Loading tensors from %s...", file_name.c_str());

        BinaryFileStream strm;
        ret = strm.OpenForRead(file_path);
        if (!ret) {
            LogError("Failed to open the model file: %s", file_name.c_str());
            return false;
        }

        int base_idx = model.tensor_spec_table.Size();
        ret = Safetensors_ReadHeader(model, strm, jparser);
        Macro_RetxFalseIf(!ret, LogError("Failed to read the header"));

        ret = Safetensors_ReadTensors(model, ctx, strm, base_idx, tm);
        Macro_RetxFalseIf(!ret, LogError("Failed to read tensors"));
    }
    tm.End();

    return ret;
}

//static
bool ModelReader::Safetensors_ReadHeader(TransformerModel &model,
    IBinaryStream &strm, JsonParser &jparser)
{
    uint64_t header_len = 0;
    bool ret = strm.Read(header_len);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the header length"));

    if (header_len >= 64 * 1000 * 1000) {
        LogError("Invalid header length: %I64u", header_len);
    }

    char *buf = new char[header_len + 1];
    ret = strm.Read(buf, header_len);
    if (!ret)
    {
        delete[] buf;
        LogError("Failed to read the header data");
        return false;
    }

    wstring json_str = StringUtil::Utf8ToWideStr(buf, (uint32_t)header_len);
    delete[] buf;

    JsonDoc jdoc;
    ret = jparser.Parse(jdoc, json_str);
    Macro_RetxFalseIf(!ret, LogError("Invalid JSON format"));

    map<wstring, ElementType, WStrLessNoCase> data_type_map;
    data_type_map[L"F32"] = ElementType::F32;
    data_type_map[L"F16"] = ElementType::F16;
    data_type_map[L"BF16"] = ElementType::BF16;

    JsonObject jobj = jdoc.GetJObject();
    JsonObject fld_obj;
    JsonArray jarray;
    wstring field_name, data_type_str;
    for (int field_idx = 0; ret && field_idx < (int)jobj.size; field_idx++)
    {
        const auto &fld = jobj.items[field_idx];
        fld.name.ToString(field_name);
        bool is_obj = fld.value.GetJObject(fld_obj);
        if (!field_name.empty() && field_name[0] != L'_' && is_obj)
        {
            TensorSpec spec;
            spec.name = StringUtil::ToUtf8(field_name);

            fld_obj.GetFieldValue(data_type_str, L"dtype", jdoc);
            auto dtype_iter = data_type_map.find(data_type_str);
            spec.data_type = dtype_iter != data_type_map.end()
                ? (int)dtype_iter->second : (int)ElementType::F16;

            ret = ret && fld_obj.GetFieldValue(jarray, L"shape", jdoc);
            spec.dims = min((int)jarray.size, TensorSpec::MaxDimCount);
            spec.size = 1;
            for (int dim_idx = 0; dim_idx < spec.dims; dim_idx++)
            {
                int target_dim_idx = spec.dims - dim_idx - 1;
                spec.ne[target_dim_idx] = jarray.items[dim_idx].GetIntValue();
                spec.size *= spec.ne[target_dim_idx];
            }

            ret = ret && fld_obj.GetFieldValue(jarray, L"data_offsets", jdoc);
            ret = ret && jarray.size >= 1;
            if (ret)
            {
                spec.offset_in_file = jarray.items[0].GetUInt64Value();
            }

            model.tensor_spec_table.Add(spec);
        }
    }

    return ret;
}

bool ModelReader::Safetensors_ReadTensors(TransformerModel &model,
    TransformerContext &ctx, IBinaryStream &strm, int base_idx,
    TaskMonitor &tm)
{
    bool ret = true;
    bool is_study_mode = false;
    uint64_t base_offset = strm.TellRd();

    ModelPartition model_partition;
    NetworkBuilder::GetDeviceAssignments(model_partition.decoder_assignments,
        model.spec, false, model.spec.decoder_cpu_layer_count);

    int builder_count = (int)model_partition.decoder_assignments.size();
    PtrVector<NetworkBuilder> builder_list;
    for (int builder_id = 0; builder_id < builder_count; builder_id++)
    {
        auto *builder = new NetworkBuilder;
        builder->Init(*network_builder_);
        builder_list.push_back(builder);
    }

    auto &tensor_array = model.tensor_spec_table.tensor_array;
    int tensor_count = (int)tensor_array.size();

#if defined(USE_CUDA)
    int default_device_id = CudaUtil::GetDevice();
#endif //USE_CUDA
    for (int tensor_idx = base_idx; tensor_idx < tensor_count; tensor_idx++)
    {
        auto &tensor_spec = tensor_array[tensor_idx];

        //int byte_count = (int)TensorCommon::ByteCount((ElementType)spec.data_type, spec.size);
        //LogKeyInfo("data_type: %d, size: %d, bytes: %d, offset_in_file: %I64u",
        //    spec.data_type, spec.size, byte_count, spec.offset_in_file);

        strm.SeekRd(base_offset + tensor_spec.offset_in_file);

        //if (tensor_spec.name.find("shared_experts") != string::npos)
        //{
        //    LogKeyInfo("shared-experts");
        //}

        bool has_device_tensor = false;
#if defined(USE_CUDA)
        if (model.spec.is_eager_device_building)
        {
            string norm_tensor_name;
            model.network.TransTensorName(norm_tensor_name, tensor_spec.name);
            //if (String::CaseCmp(tensor_spec.name, "word_embeddings.weight") == 0) {
            //    LogKeyInfo("hi");
            //}
            if (!norm_tensor_name.empty())
            {
                TensorNameInfo tni;
                bool is_succ = NetworkBuilder::ParseTensorName(tni, norm_tensor_name,
                    model.std_network.device_net, model.spec);
                if (is_succ && tni.layer_id >= model.spec.decoder_cpu_layer_count)
                {
                    if (tni.expert_id < model.spec.hyper_params.in_use_experts)
                    {
                        HostTensor cpu_tensor;
                        ret = ReadAndTransTensor(cpu_tensor, nullptr, tensor_spec, ctx, strm);
                        Macro_RetFalseIf(!ret);

                        int device_id = NetworkBuilder::GetDeviceByLayer(model_partition,
                            tni.layer_id, true);
                        CudaUtil::SetDevice(device_id);

                        int la_idx = NetworkBuilder::GetDeviceGroupIndex(model_partition,
                            tni.layer_id, true);
                        auto *net_builder = builder_list[la_idx];
                        net_builder->BuildDeviceTensor(model.std_network.device_net,
                            cpu_tensor, tni, model.spec);
                    }
                    has_device_tensor = true;
                }
            }
        }
#endif //USE_CUDA

        if (!has_device_tensor)
        {
            HostTensor *host_tensor = ctx.host_tensor_heap.New(1);
            ret = ReadAndTransTensor(*host_tensor, nullptr, tensor_spec, ctx, strm);
            Macro_RetFalseIf(!ret);

            model.std_network.tensor_map[tensor_idx] = host_tensor;

            if (is_study_mode && tensor_spec.data_type == (int)ElementType::F16)
            {
                const inferflow_fp16 *half_array = (inferflow_fp16*)host_tensor->data;
                float v1 = (float)half_array[0];
                float v2 = (float)half_array[1];
                LogKeyInfo("name: %s, v1: %f, v2: %f", tensor_spec.name.c_str(), v1, v2);
            }
        }

        //strm.SeekRd(spec.offset_in_file + spec.size);

        tm.Progress(1 + tensor_idx);
    }

#if defined(USE_CUDA)
    CudaUtil::SetDevice(default_device_id);
#endif //USE_CUDA
    return strm.IsGood();
}

////////////////////////////////////////////////////////////////////////////////
// lamma.cpp GGML format (https://github.com/ggerganov/llama.cpp)
////////////////////////////////////////////////////////////////////////////////

struct LlamaCppMagic
{
    static const uint32_t GGJT = 0x67676a74; // 'ggjt'
    static const uint32_t GGLA = 0x67676c61; // 'ggla'
    static const uint32_t GGMF = 0x67676d66; // 'ggmf'
    static const uint32_t GGML = 0x67676d6c; // 'ggml'
    static const uint32_t GGSN = 0x6767736e; // 'ggsn'
    static const uint32_t GGUF = 0x46554747; // "GGUF"
};

enum class LlamaCppFileVersion
{
    GGML,
    GGMF_V1, // added version field and scores in vocab
    GGJT_V1, // added padding
    GGJT_V2, // changed quantization format
    GGJT_V3, // changed Q4 and Q8 quantization format
};

bool ModelReader::LoadModel_GGML(TransformerModel &model, TransformerContext &ctx,
    const ModelSpec &spec, bool is_study_mode)
{
    (void)is_study_mode;
    string model_file = spec.model_files.empty() ? "" : spec.model_files[0];
    string file_path = spec.dir + model_file;
    uint64_t file_length = 0;
    bool ret = Path::GetFileLength(file_path, file_length);
    if (!ret) {
        LogError("Cannot get file length: %s", file_path.c_str());
        return false;
    }

    BinaryFileStream strm;
    ret = strm.OpenForRead(file_path);
    if (!ret) {
        LogError("Failed to open file %s", file_path.c_str());
        return false;
    }

    int file_version = 0;
    ret = ret && LlamaCpp_ReadMagic(file_version, strm);
    ret = ret && LlamaCpp_ReadHyperParams(model, strm);

    model.spec.hyper_params.decoder_kv_heads = model.spec.hyper_params.decoder_heads;

    bool has_score = file_version >= (int)LlamaCppFileVersion::GGMF_V1;
    ret = ret && ReadVocabulary_Std(model, strm, has_score);
    ret = ret && LlamaCpp_ReadTensors(model, ctx, strm, file_version, file_length);
    Macro_RetFalseIf(!ret);

    return ret;
}

//static
bool ModelReader::LlamaCpp_ReadMagic(int &file_version, IBinaryStream &strm)
{
    uint32_t magic = 0, version = 0;
    bool ret = strm.Read(magic);
    if (magic == LlamaCppMagic::GGML)
    {
        file_version = (int)LlamaCppFileVersion::GGML;
        return true;
    }

    ret = strm.Read(version);
    if (!ret) {
        return false;
    }

    switch (magic)
    {
    case LlamaCppMagic::GGMF:
        switch (version)
        {
        case 1: file_version = (int)LlamaCppFileVersion::GGMF_V1; return true;
        }
        break;
    case LlamaCppMagic::GGJT:
        switch (version)
        {
        case 1: file_version = (int)LlamaCppFileVersion::GGJT_V1; return true;
        case 2: file_version = (int)LlamaCppFileVersion::GGJT_V2; return true;
        case 3: file_version = (int)LlamaCppFileVersion::GGJT_V3; return true;
        }
    default: break;
    }

    LogError("Unknown (magic, version) combination: %08x, %08x", magic, version);
    return false;
}

//static
bool ModelReader::LlamaCpp_ReadHyperParams(TransformerModel &model, IBinaryStream &strm)
{
    auto &hparams = model.spec.hyper_params;
    uint32_t mult = 0, rot = 0, ftype = 0;

    bool ret = strm.Read(hparams.vocab_size);
    ret = ret && strm.Read(hparams.embd_dims);
    ret = ret && strm.Read(mult);
    ret = ret && strm.Read(hparams.decoder_heads);
    ret = ret && strm.Read(hparams.decoder_layers);
    ret = ret && strm.Read(rot);
    ret = ret && strm.Read(ftype);

    LogKeyInfo("vocab_size: %d, embd_dims: %d, heads: %d, layers: %d, rot: %d, ftype: %d",
        hparams.vocab_size, hparams.embd_dims, hparams.decoder_heads, hparams.decoder_layers,
        rot, ftype);
    return ret;
}

//static
bool ModelReader::LlamaCpp_ReadTensors(TransformerModel &model,
    TransformerContext &ctx, IBinaryStream &strm, int file_version,
    uint64_t file_length)
{
    uint32_t dims = 0, name_len = 0, data_type = 0;
    const int max_tensor_name_len = 255;
    char buf[max_tensor_name_len + 1];
    int tensor_id = 0;

    TaskMonitor tm(20);
    uint32_t tensor_num = 0;
    while (strm.TellRd() < file_length)
    {
        TensorSpec spec;
        strm.Read(dims);
        strm.Read(name_len);
        strm.Read(data_type);

        if (dims < 1 || dims > 2) {
            LogError("Tensors in this file should NOT be %u-dimensional", dims);
            return false;
        }

        if (name_len > max_tensor_name_len) {
            LogError("The tensor name is too long: %d", name_len);
            return false;
        }

        spec.dims = dims;
        spec.data_type = data_type;
        strm.Read((char*)spec.ne, dims * sizeof(spec.ne[0]));
        strm.Read(buf, name_len);

        spec.id = tensor_id++;
        spec.name.assign(buf, name_len);
        //LogKeyInfo("dims: %d, name_len: %d, name: %s", dims, name_len, spec.name.c_str());

        switch (spec.data_type)
        {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        default:
            LogError("Unrecognized tensor type %u", spec.data_type);
            return false;
        }

        if (file_version >= (int)LlamaCppFileVersion::GGJT_V1)
        {
            // skip to the next multiple of 32 bytes
            uint64_t offset = strm.TellRd();
            uint64_t r = offset % 32;
            if (r != 0) {
                strm.SeekRd(offset + (32 - r));
            }
            //LogKeyInfo("offset: %I64u, r: %d", offset, (int)r);
        }

        spec.file_idx = 0;
        spec.offset_in_file = strm.TellRd();
        spec.size = LlamaCpp_CalcTensorSize(spec, file_version);
        //LogKeyInfo("spec.offset_in_file: %I64u, spec.size: %d", spec.offset_in_file, spec.size);

        ElementType element_type = TensorUtil::ToElementType((ggml_type)spec.data_type);
        HostTensor *host_tensor = ReadTensor(element_type, spec.ne[0], spec.ne[1], strm, ctx);
        /*struct ggml_tensor *host_tensor = nullptr;
        if (spec.dims == 1) {
            host_tensor = ggml_new_tensor_1d(ctx.ggml_ctx, (ggml_type)spec.data_type, spec.ne[0]);
        }
        else if (spec.dims == 2) {
            host_tensor = ggml_new_tensor_2d(ctx.ggml_ctx, (ggml_type)spec.data_type, spec.ne[0], spec.ne[1]);
        }*/

        if (host_tensor == nullptr) {
            LogError("Null host tensor");
            return false;
        }

        //host_tensor->data = ctx.byte_heap.New(spec.size);
        //strm.Read((char*)host_tensor->data, spec.size);*/

        strm.SeekRd(spec.offset_in_file + spec.size);

        //add to the tensor spec table
        int ret_idx = model.tensor_spec_table.Add(spec);
        model.std_network.tensor_map[ret_idx] = host_tensor;

        tm.Progress(++tensor_num);
    }
    tm.End();

    return strm.IsGood();
}

//static
uint32_t ModelReader::LlamaCpp_CalcTensorSize(const TensorSpec &spec, int file_version)
{
    uint64_t size = 1;
    for (int dim_idx = 0; dim_idx < spec.dims; dim_idx++)
    {
        int dim = spec.ne[dim_idx];
        size *= dim;
    }

    auto data_type = (ggml_type)spec.data_type;
    int block_size = ggml_blck_size((ggml_type)spec.data_type);
    int type_size = (int)ggml_type_size((ggml_type)spec.data_type);
    if (file_version <= 0)
    {
        if (data_type == GGML_TYPE_Q4_0) {
            type_size += 2;
        }
        else if (data_type == GGML_TYPE_Q4_1) {
            type_size += 4;
        }
    }

    uint32_t tensor_bytes = (uint32_t)(size * type_size / block_size);
    //LogKeyInfo("data_type: %d, size: %I64u, type_size: %d, block_size: %d, tensor_bytes: %u",
    //    spec.data_type, size, type_size, block_size, tensor_bytes);
    return tensor_bytes;
}

////////////////////////////////////////////////////////////////////////////////
// lamma.cpp GGUF format (https://github.com/ggerganov/llama.cpp)
////////////////////////////////////////////////////////////////////////////////

bool ModelReader::LoadModel_GGUF(TransformerModel &model, TransformerContext &ctx,
    const ModelSpec &spec, bool is_study_mode)
{
    (void)model; (void)ctx; (void)is_study_mode;
    string model_file = spec.model_files.empty() ? "" : spec.model_files[0];
    string file_path = spec.dir + model_file;
    uint64_t file_length = 0;
    bool ret = Path::GetFileLength(file_path, file_length);
    if (!ret) {
        LogError("Cannot get file length: %s", file_path.c_str());
        return false;
    }

    BinaryFileStream strm;
    ret = strm.OpenForRead(file_path);
    if (!ret) {
        LogError("Failed to open file %s", file_path.c_str());
        return false;
    }

    GgufHeader header;
    ret = GGUF_ReadHeader(header, strm);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the header"));
     
    map<string, GgufAttr> attr_map;
    ret = GGUF_ReadAttributes(attr_map, strm, header);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the attributes"));

    ret = GGUF_ReadHyperParams(model, attr_map);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the hyper-parameters"));

    int token_bytes_mapping = 0;
    ret = GGUF_ReadVocabulary(model, attr_map, token_bytes_mapping);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the vocabulary"));

    ret = GGUF_ReadTensorSpecTable(model.tensor_spec_table, strm, header);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the tensor spec table"));

    int alignment = 32;
    auto attr_iter = attr_map.find("general.alignment");
    if (attr_iter != attr_map.end()) {
        alignment = attr_iter->second.value.u32;
    }

    uint64_t offset = strm.TellRd();
    uint64_t offset_pad = offset % alignment;
    if (offset_pad != 0)
    {
        offset += (alignment - offset_pad);
        strm.SeekRd(offset);
    }

    int tensor_num = model.tensor_spec_table.Size();
    for (int tensor_idx = 0; ret && tensor_idx < tensor_num; tensor_idx++)
    {
        auto &tensor_spec = model.tensor_spec_table.tensor_array[tensor_idx];
        ElementType element_type = (ElementType)tensor_spec.data_type;
        auto *tensor_ptr = ReadTensor(element_type, tensor_spec.ne[0], tensor_spec.ne[1], strm, ctx);
        model.std_network.tensor_map[tensor_idx] = tensor_ptr;
        if (tensor_ptr == nullptr)
        {
            LogError("Failed to load tensor %s", tensor_spec.name.c_str());
            return false;
        }
    }

    return ret;
}

bool ModelReader::GGUF_ReadHeader(GgufHeader &header, IBinaryStream &strm)
{
    bool ret = strm.Read(header.magic);
    if (header.magic != LlamaCppMagic::GGUF)
    {
        LogError("Invalid magic number (%08x) in reading the GGUF file.", header.magic);
        return false;
    }

    ret = strm.Read(header.version);
    if (!ret) {
        return false;
    }

    if (header.version == 1)
    {
        uint32_t tensor_num_u32 = 0;
        uint32_t attr_num_u32 = 0;
        ret = ret && strm.Read(tensor_num_u32);
        ret = ret && strm.Read(attr_num_u32);
        header.tensor_num = tensor_num_u32;
        header.attr_num = attr_num_u32;
    }
    else
    {
        ret = ret && strm.Read(header.tensor_num);
        ret = ret && strm.Read(header.attr_num);
    }

    return ret;
}

bool ModelReader::GGUF_ReadAttributes(map<string, GgufAttr> &attr_map,
    IBinaryStream &strm, const GgufHeader &header)
{
    bool ret = true; 
    attr_map.clear();

    int buf_len = 1024;
    char *buf = new char[buf_len];
    int value_type = 0;
    string attr_name;

    for (int attr_idx = 0; ret && attr_idx < (int)header.attr_num; attr_idx++)
    {
        GgufAttr attr;
        ret = GGUF_ReadString(attr_name, strm, header.version, buf, buf_len);
        ret = ret && strm.Read(value_type);
        attr.value_type = (GgufValueType)value_type;
        ret = ret && GGUF_ReadAttrValue(attr.value, strm, attr.value_type, header.version);
        attr_map[attr_name] = attr;
    }

    delete[] buf;
    return ret;
}

bool ModelReader::GGUF_ReadTensorSpecTable(TensorSpecTable &tensor_spec_table,
    IBinaryStream &strm, const GgufHeader &header)
{
    bool ret = true;
    int buf_len = 1024;
    char *buf = new char[buf_len];
    uint64_t ne[HostTensor::MaxDimCount];

    for (int tensor_idx = 0; ret && tensor_idx < (int)header.tensor_num; tensor_idx++)
    {
        TensorSpec tensor_spec;
        ret = GGUF_ReadString(tensor_spec.name, strm, header.version, buf, buf_len);
        ret = ret && strm.Read(tensor_spec.dims);
        if (!ret) {
            LogError("Error occurred in reading the name or dims of tensor %d", tensor_idx);
            break;
        }
        if (tensor_spec.dims <= 0 && tensor_spec.dims > 2) {
            LogError("Invalid tensor dims: %d (should be 1 or 2)", tensor_spec.dims);
            break;
        }

        for (int dim = 0; dim < tensor_spec.dims; dim++)
        {
            GGUF_ReadSize(ne[dim], strm, header.version);
            tensor_spec.ne[dim] = (uint32_t)ne[dim];
        }

        int ggml_data_type = 0;
        ret = ret && strm.Read(ggml_data_type);
        ElementType element_type = TensorUtil::ToElementType((ggml_type)ggml_data_type);
        if (element_type == ElementType::Invalid)
        {
            LogError("ggml data type %d is not supported so far.", ggml_data_type);
            ret = false;
            break;
        }

        tensor_spec.data_type = (int)element_type;
        ret = ret && strm.Read(tensor_spec.offset_in_file);

        tensor_spec_table.Add(tensor_spec);
    }

    delete[] buf;
    return ret;
}

bool ModelReader::GGUF_ReadHyperParams(TransformerModel &model,
    const map<string, GgufAttr> &attr_map)
{
    bool ret = true;
    auto &hparams = model.spec.hyper_params;

    string model_arch = GGUF_GetAttrValue_Str(attr_map, "general.architecture");
    string model_name = GGUF_GetAttrValue_Str(attr_map, "general.name");

    const auto *array_ptr = GGUF_GetAttrValue_Arr(attr_map, "tokenizer.ggml.tokens");
    if (array_ptr != nullptr) {
        hparams.vocab_size = (int)array_ptr->size;
    }

    string attr_name = GGUF_BuildAttrName("%s.context_length", model_arch);
    hparams.training_context_len = GGUF_GetAttrValue_I32(attr_map, attr_name);

    attr_name = GGUF_BuildAttrName("%s.embedding_length", model_arch);
    hparams.embd_dims = GGUF_GetAttrValue_I32(attr_map, attr_name);

    attr_name = GGUF_BuildAttrName("%s.feed_forward_length", model_arch);
    hparams.hidden_dim = GGUF_GetAttrValue_I32(attr_map, attr_name);

    attr_name = GGUF_BuildAttrName("%s.attention.head_count", model_arch);
    int kv_heads = GGUF_GetAttrValue_I32(attr_map, attr_name);
    hparams.decoder_kv_heads = kv_heads > 0 ? kv_heads : hparams.decoder_heads;

    attr_name = GGUF_BuildAttrName("%s.attention.head_count_kv", model_arch);
    hparams.decoder_heads = GGUF_GetAttrValue_I32(attr_map, attr_name);

    attr_name = GGUF_BuildAttrName("%s.block_count", model_arch);
    hparams.decoder_layers = GGUF_GetAttrValue_I32(attr_map, attr_name);

    return ret;
}

bool ModelReader::GGUF_ReadVocabulary(TransformerModel &model,
    const map<string, GgufAttr> &attr_map, int token_bytes_mapping)
{
    bool ret = true;
    //const auto &hparams = model.hyper_params;
    auto &vocab = model.vocabulary;
    vocab.Clear();

    const auto *token_array = GGUF_GetAttrValue_Arr(attr_map, "tokenizer.ggml.tokens");
    const auto *score_array = GGUF_GetAttrValue_Arr(attr_map, "tokenizer.ggml.scores");
    const auto *token_type_array = GGUF_GetAttrValue_Arr(attr_map, "tokenizer.ggml.token_type");
    if (token_array == nullptr) {
        LogError("Cannot find the token array");
        return false;
    }

    int vocab_size = (int)token_array->size;
    vocab.token_array.resize(vocab_size);

    string key1 = StringUtil::ToUtf8(L"Ġ");
    string key2 = StringUtil::ToUtf8(L"▁");
    wstring wstr;
    for (int token_id = 0; token_id < vocab_size; token_id++)
    {
        auto &token = model.vocabulary.token_array[token_id];
        token.id = token_id;
        GgufStr *str_array = (GgufStr*)token_array->data;
        token.str.assign(str_array[token_id].data, str_array[token_id].size);

        if (token_bytes_mapping != 0)
        {
            StringUtil::Utf8ToWideStr(wstr, token.str);
            bool is_succ = DecodeTokenStr(token.str, wstr, token_bytes_mapping);
            if (!is_succ) {
                LogError("Invalid token %d", token.id);
                return false;
            }
        }

        token.score = 0.0f;
        if (score_array != nullptr) {
            token.score = ((float*)score_array->data)[token_id];
        }

        //String::ReplaceAll(token.str, key1, " ");
        String::ReplaceAll(token.str, key2, " ");

        if (token.str.size() == 3 && (uint8_t)token.str[0] == 0xEF
            && (uint8_t)token.str[1] == 0xBF && (uint8_t)token.str[2] == 0xBD)
        {
            //token.score = -1;
            token.type = (int)TokenType::Invalid;
        }

        token.type = 0;
        if (token_type_array != nullptr) {
            token.type = ((int*)token_type_array->data)[token_id];
        }

        vocab.str_to_id[token.str] = token.id;
    }

    return ret;
} 

//static
bool ModelReader::GGUF_ReadAttrValue(GgufAttrValue &value, IBinaryStream &strm,
    GgufValueType value_type, int version)
{
    bool ret = true;
    int i32_value = 0;
    int type_size = -1;
    GgufStr *str_array = nullptr;

    switch (value_type)
    {
    case GgufValueType::T_UINT8: ret = strm.Read(value.u8); break;
    case GgufValueType::T_INT8: ret = strm.Read(value.i8); break;
    case GgufValueType::T_UINT16: ret = strm.Read(value.u16); break;
    case GgufValueType::T_INT16: ret = strm.Read(value.i16); break;
    case GgufValueType::T_UINT32: ret = strm.Read(value.u32); break;
    case GgufValueType::T_INT32: ret = strm.Read(value.i32); break;
    case GgufValueType::T_FLOAT32: ret = strm.Read(value.f32); break;
    case GgufValueType::T_UINT64: ret = strm.Read(value.u64); break;
    case GgufValueType::T_INT64: ret = strm.Read(value.i64); break;
    case GgufValueType::T_FLOAT64: ret = strm.Read(value.f64); break;
    case GgufValueType::T_BOOL: ret = strm.Read(value.bval); break;
    case GgufValueType::T_STRING: ret = GGUF_ReadString(value.str, strm, version); break;
        break;
    case GgufValueType::T_ARRAY:
        ret = strm.Read(i32_value);
        value.arr.value_type = (GgufValueType)i32_value;
        ret = ret && GGUF_ReadSize(value.arr.size, strm, version);

        type_size = -1;
        switch (value.arr.value_type)
        {
        case GgufValueType::T_UINT8:
        case GgufValueType::T_INT8:
            type_size = 1;
            break;
        case GgufValueType::T_UINT16:
        case GgufValueType::T_INT16:
            type_size = 2;
            break;
        case GgufValueType::T_UINT32:
        case GgufValueType::T_INT32:
        case GgufValueType::T_FLOAT32:
            type_size = 4;
            break;
        case GgufValueType::T_UINT64:
        case GgufValueType::T_INT64:
        case GgufValueType::T_FLOAT64:
            type_size = 8;
            break;
        case GgufValueType::T_BOOL:
            type_size = sizeof(bool);
            break;
        case GgufValueType::T_STRING:
            str_array = new GgufStr[value.arr.size];
            for (uint32_t item_idx = 0; item_idx < value.arr.size; item_idx++) {
                ret = ret && GGUF_ReadString(str_array[item_idx], strm, version);
            }
            value.arr.data = str_array;
            break;
        default:
            LogError("Invalid array type: %d", value.arr.value_type);
            break;
        }

        if (type_size > 0)
        {
            value.arr.data = new char[value.arr.size * type_size];
            ret = ret && strm.Read((char*)value.arr.data, value.arr.size * type_size);
        }
        break;
    default:
        LogError("Invalid type: %d", value_type);
        break;
    }

    return ret;
}

//static
bool ModelReader::GGUF_ReadString(GgufStr &str, IBinaryStream &strm, int version)
{
    bool ret = GGUF_ReadSize(str.size, strm, version);
    if (ret)
    {
        str.data = new char[str.size];
        strm.Read(str.data, str.size);
    }

    return ret;
}

//static
bool ModelReader::GGUF_ReadString(string &str, IBinaryStream &strm, int version,
    char *buf, uint64_t buf_len)
{
    str.clear();
    uint64_t str_len = 0;
    bool ret = GGUF_ReadSize(str_len, strm, version);
    if (ret)
    {
        if (str_len > buf_len)
        {
            LogError("The String length is too large: %d", str_len);
            return false;
        }

        strm.Read(buf, str_len);
        str.assign(buf, str_len);
    }

    return ret;
}

//static
bool ModelReader::GGUF_ReadSize(uint64_t &size, IBinaryStream &strm, int version)
{
    bool ret = true;
    if (version == 1)
    {
        uint32_t u32 = 0;
        ret = strm.Read(u32);
        size = u32;
    }
    else
    {
        ret = strm.Read(size);
    }
    return ret;
}

//static
uint32_t ModelReader::GGUF_GetAttrValue_U32(const map<string, GgufAttr> &attr_map,
    const string &attr_name)
{
    auto iter = attr_map.find(attr_name);
    if (iter == attr_map.end()) {
        return UINT32_MAX;
    }

    const GgufAttr &attr = iter->second;
    switch (attr.value_type)
    {
    case GgufValueType::T_UINT32:
        return attr.value.u32;
    case GgufValueType::T_INT32:
        return (uint32_t)attr.value.i32;
    case GgufValueType::T_UINT16:
        return (uint32_t)attr.value.u16;
    case GgufValueType::T_INT16:
        return (uint32_t)attr.value.i16;
    default:
        break;
    }

    return UINT32_MAX;
}

//static
int ModelReader::GGUF_GetAttrValue_I32(const map<string, GgufAttr> &attr_map,
    const string &attr_name)
{
    auto iter = attr_map.find(attr_name);
    if (iter == attr_map.end()) {
        return UINT32_MAX;
    }

    const GgufAttr &attr = iter->second;
    switch (attr.value_type)
    {
    case GgufValueType::T_UINT32:
        return (int)attr.value.u32;
    case GgufValueType::T_INT32:
        return attr.value.i32;
    case GgufValueType::T_UINT16:
        return (int)attr.value.u16;
    case GgufValueType::T_INT16:
        return (int)attr.value.i16;
    default:
        break;
    }

    return -1;
}

//static
string ModelReader::GGUF_GetAttrValue_Str(const map<string, GgufAttr> &attr_map,
    const string &attr_name)
{
    auto iter = attr_map.find(attr_name);
    string str;
    const GgufAttr *attr_ptr = iter != attr_map.end() ? &iter->second: nullptr;
    if (attr_ptr != nullptr && attr_ptr->value_type == GgufValueType::T_STRING) {
        str.assign(attr_ptr->value.str.data, attr_ptr->value.str.size);
    }

    return str;
}

//static
const ModelReader::GgufArr* ModelReader::GGUF_GetAttrValue_Arr(
    const map<string, GgufAttr> &attr_map, const string &attr_name)
{
    auto iter = attr_map.find(attr_name);
    const GgufAttr *attr_ptr = iter != attr_map.end() ? &iter->second : nullptr;
    if (attr_ptr != nullptr && attr_ptr->value_type == GgufValueType::T_ARRAY) {
        return &attr_ptr->value.arr;
    }

    return nullptr;
}

//static
string ModelReader::GGUF_BuildAttrName(const string &name_template, const string &model_name)
{
    string attr_name = name_template;
    String::ReplaceAll(attr_name, "%s", model_name);
    return attr_name;
}


////////////////////////////////////////////////////////////////////////////////
// llama2.c format (https://github.com/karpathy/llama2.c)
////////////////////////////////////////////////////////////////////////////////

bool ModelReader::LoadModel_Llama2DotC(TransformerModel &model,
    TransformerContext &ctx, const ModelSpec &spec, bool is_study_mode)
{
    (void)is_study_mode;
    string model_file = spec.model_files.empty() ? "" : spec.model_files[0];
    string file_path = spec.dir + model_file;
    uint64_t file_length = 0;
    bool ret = Path::GetFileLength(file_path, file_length);
    if (!ret) {
        LogError("Cannot get file length: %s", file_path.c_str());
        return false;
    }

    BinaryFileStream file_stream;
    ret = file_stream.OpenForRead(file_path);
    if (!ret) {
        LogError("Failed to open file %s", file_path.c_str());
        return false;
    }

    uint32_t magic = 0, version = 0;
    IBinaryStream &strm = file_stream;
    ret = strm.Read(magic);
    Macro_RetxFalseIf(!ret, LogError("Failed to read the magic number"));

    if (magic == 0x616b3432) //"ak42" in ASCII
    {
        ret = strm.Read(version);
        Macro_RetxFalseIf(!ret, LogError("Failed to read the version number"));

        if (version != 1) {
            LogError("Version %d is not supported so far.", version);
            return false;
        }
    }
    else
    {
        strm.SeekRd(0);
    }

    auto &hparams = model.spec.hyper_params;
    ret = ret && strm.Read(hparams.embd_dims);
    ret = ret && strm.Read(hparams.hidden_dim);
    ret = ret && strm.Read(hparams.decoder_layers);
    ret = ret && strm.Read(hparams.decoder_heads);
    ret = ret && strm.Read(hparams.decoder_kv_heads);
    ret = ret && strm.Read(hparams.vocab_size);
    ret = ret && strm.Read(hparams.training_context_len);

    bool is_shared_classifier = hparams.vocab_size >= 0;
    hparams.vocab_size = abs(hparams.vocab_size);

    if (version == 1)
    {
        uint8_t v8 = (uint8_t)0;
        ret = ret && strm.Read(v8);
        is_shared_classifier = v8 != 0;
        ret = ret && strm.SeekRd(256);
    }

    Macro_RetxFalseIf(!ret, LogError("Failed to read the hyper-paramters"));
    LogKeyInfo("vocab_size: %d, embd_dim: %d, hidden_dim: %d, heads: %d, kv_heads: %d, layers: %d",
        hparams.vocab_size, hparams.embd_dims, hparams.hidden_dim,
        hparams.decoder_heads, hparams.decoder_kv_heads, hparams.decoder_layers);

    ret = ret && Llama2DotC_ReadTensors(model, ctx, strm,
        version, file_length, is_shared_classifier);
    Macro_RetFalseIf(!ret);

    if (spec.tokenizer_files.empty() || spec.tokenizer_files[0].empty())
    {
        LogError("Empty tokenizer file");
        return false;
    }

    /// vocabulary
    file_path = spec.dir + spec.tokenizer_files[0];
    file_stream.Close();
    ret = file_stream.OpenForRead(file_path);
    if (!ret) {
        LogError("Failed to open file %s", file_path.c_str());
        return false;
    }

    ret = ReadVocabulary_Format2(model, file_stream);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the vocabulary"));

    return ret;
}

//static
bool ModelReader::Llama2DotC_ReadTensors(TransformerModel &model,
    TransformerContext &ctx, IBinaryStream &strm, int version,
    uint64_t file_length, bool is_shared_classifier)
{
    (void)file_length; (void)version;
    const auto &hparams = model.spec.hyper_params;
    int embd_dim = hparams.embd_dims;
    int head_size = hparams.embd_dims / hparams.decoder_heads;
    auto &host_net = model.std_network.host_net;
    ElementType data_type = ElementType::F32;

    TaskMonitor tm;
    model.decoder_embeddings = ReadTensor(data_type, embd_dim, hparams.vocab_size, strm, ctx);
    tm.Progress(1);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto *new_layer = new StdHostNetwork::DecoderLayer;
        new_layer->self_attn.pre_norm = ReadTensor(data_type, embd_dim, 0, strm, ctx);
        host_net.decoder_layers.push_back(new_layer);
    }
    tm.Progress(2);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->self_attn.wq = ReadTensor(data_type, embd_dim,
            hparams.decoder_heads * head_size, strm, ctx);
    }
    tm.Progress(3);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->self_attn.wk = ReadTensor(data_type, embd_dim, hparams.decoder_kv_heads * head_size, strm, ctx);
    }
    tm.Progress(4);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->self_attn.wv = ReadTensor(data_type, embd_dim, hparams.decoder_kv_heads * head_size, strm, ctx);
    }
    tm.Progress(5);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->self_attn.wo = ReadTensor(data_type, hparams.decoder_heads * head_size, embd_dim, strm, ctx);
    }
    tm.Progress(6);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->ffn.pre_norm = ReadTensor(data_type, embd_dim, 0, strm, ctx);
    }
    tm.Progress(7);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->ffn.w1 = ReadTensor(data_type, embd_dim, hparams.hidden_dim, strm, ctx);
    }
    tm.Progress(8);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->ffn.w2 = ReadTensor(data_type, hparams.hidden_dim, embd_dim, strm, ctx);
    }
    tm.Progress(9);

    for (int layer_id = 0; layer_id < hparams.decoder_layers; layer_id++)
    {
        auto layer = host_net.decoder_layers[layer_id];
        layer->ffn.w3 = ReadTensor(data_type, embd_dim, hparams.hidden_dim, strm, ctx);
    }
    tm.Progress(10);

    host_net.decoder_output_norm = ReadTensor(data_type, embd_dim, 0, strm, ctx);
    uint64_t offset = strm.TellRd();
    strm.SeekRd(offset + hparams.training_context_len * head_size);

    host_net.output = model.decoder_embeddings;
    if (!is_shared_classifier) {
        host_net.output = ReadTensor(data_type, embd_dim, hparams.vocab_size, strm, ctx);
    }

    tm.End();
    return strm.IsGood();
}

TRANSFORMER_END
INFER_FLOW_END
