#include "inference_engine.h"
#include "sslib/path.h"
#include "tensor/tensor_util.h"
#include "common/quantization.h"
#include "model_writer.h"
#if defined(USE_CUDA)
#   include "tensor/device_tensor_util.h"
#   include "tensor/device_tensor_builder.h"
#endif //USE_CUDA

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

InferenceEngine::InferenceEngine()
{
    version_ = "0.1.0";
}

InferenceEngine::~InferenceEngine()
{
    decoding_strategies_.Clear();

    if (config_.debug.is_study_mode && config_.debug.show_tensors) {
        tensor_writer_.close();
    }

    if (cpu_worker_ != nullptr)
    {
        delete cpu_worker_;
        cpu_worker_ = nullptr;
    }

    if (cpu_ctx_ != nullptr)
    {
        ggml_free(cpu_ctx_);
        cpu_ctx_ = nullptr;
    }
}

bool InferenceEngine::Init(const InferenceConfig &cfg)
{
    bool ret = true;
    TensorCommon::InitElementTypeMap(element_type_map_);
    TransformerModel::InitModelFileFormatMap(model_file_format_map_);
    TransformerModel::InitNetworkStructureMap(network_structure_map_);

    config_ = cfg;
    config_.max_concurrent_queries = min(cfg.max_concurrent_queries,
        1 + QueryState::MAX_PROC_ID);
    config_ex_.is_gpu_tensor_row_major = false;
    //config_ex_.matrix_mul_alg = MatrixMulAlg::Bruce;
    config_ex_.gemv_alg = VectorMatrixMulAlg::Alg3;
    config_ex_.enable_full_quant_gemv = true;

    if (config_.debug.is_study_mode && config_.debug.show_tensors)
    {
        string tensor_writer_path = config_.data_dir + "tensor_dump.txt";
        tensor_writer_.open(tensor_writer_path);
    }

    InitKeyMap();

    json_parser_.Init();
    decoding_strategies_.Init();

    const ModelSpec *model_spec = config_.models.empty() ? nullptr : &config_.models[0];
    if (model_spec == nullptr) {
        LogError("No model is available");
        return false;
    }

    //std::sort(config_.devices.begin(), config_.devices.end(), std::greater<int>());
    //default_device_id_ = *config_.devices.rbegin();
    default_device_id_ = GetDefaultDevice(*model_spec);

    model_reader_.Init(&network_builder_);
    network_builder_.Init(config_, config_ex_, context_);

#if defined(USE_CUDA)
    local_device_heap_.Init(64 * 1000 * 1000); //64 million
#endif //USE_CUDA

    //ret = cublas_engine_.Init();
    //Macro_RetxFalseIf(!ret, LogError("Failed to initialize the cublas_engine"));

    struct ggml_init_params ggml_params;
    ggml_params.mem_size = 200 * 1024 * 1024; //200MB
    ggml_params.mem_buffer = nullptr;
    ggml_params.no_alloc = true;
    context_.ggml_ctx = ggml_init(ggml_params);

    LogKeyInfo("Loading model %s...", model_spec->sid.c_str());
    model_.spec = *model_spec;
    ret = model_reader_.Load(model_, context_, *model_spec, config_.debug.is_study_mode);
    model_spec = &model_.spec;
    if (!ret) {
        LogError("Failed to load the model");
        return false;
    }

    const auto &hparams = model_.spec.hyper_params;
    if (model_.spec.max_context_len <= 0 && hparams.training_context_len > 0) {
        model_.spec.max_context_len = hparams.training_context_len;
    }
    else if (model_.spec.max_context_len <= 16) {
        model_.spec.max_context_len = ModelSpec::DEFAULT_MAX_CONTEXT_LEN;
    }

    const auto &vocab = model_.vocabulary;
    wstring token_str = StringUtil::Utf8ToWideStr(TokenIdToStr(vocab.unk()));
    LogKeyInfo(L"unk\t%d\t%ls", vocab.unk(), token_str.c_str());
    token_str = StringUtil::Utf8ToWideStr(TokenIdToStr(vocab.bos()));
    LogKeyInfo(L"bos\t%d\t%ls", vocab.bos(), token_str.c_str());
    token_str = StringUtil::Utf8ToWideStr(TokenIdToStr(vocab.eos()));
    LogKeyInfo(L"eos\t%d\t%ls", vocab.eos(), token_str.c_str());

    LogKeyInfo("Initialize the tokenizer...");
    ret = tokenizer_.Init(model_.vocabulary);
    Macro_RetxFalseIf(!ret, LogError("Failed to initialize the tokenizer"));

    if (model_spec->model_file_format == ModelFileFormat::LLAMA2_C)
    {
        network_builder_.BuildHostTensorMap(model_.std_network.host_net);
    }
    else
    {
        LogKeyInfo("Building the host network (type: %d)...",
            (int)model_spec->network_structure);
        ret = network_builder_.BuildHostNetwork(model_, *model_spec);
        Macro_RetxFalseIf(!ret, LogError("Failed to build the host network"));
    }

    LogKeyInfo("Checking the on-host model...");
    ret = network_builder_.CheckHostModel(model_);
    Macro_RetxFalseIf(!ret, LogError("Something is wrong with the model"));

    bool enable_printing = false;
    if (enable_printing)
    {
        LogKeyInfo("Printing the model to JSON files...");
        string model_dump_dir = config_.data_dir + "model_dump/";
        Path::Mkdir(model_dump_dir);
        ret = ModelWriter::Print(model_, model_dump_dir, "model", false, true);
        if (!ret) {
            LogError("Failed to print the model");
        }
    }

    is_cpu_only_ = true;
#if defined(USE_CUDA)
    is_cpu_only_ = model_spec->decoder_cpu_layer_count >= hparams.decoder_layers
        && model_spec->encoder_cpu_layer_count >= hparams.encoder_layers;
    if (!is_cpu_only_)
    {
        ModelPartition model_partition;
        NetworkBuilder::GetDeviceAssignments(model_partition.encoder_assignments,
            *model_spec, true, model_spec->encoder_cpu_layer_count);
        NetworkBuilder::GetDeviceAssignments(model_partition.decoder_assignments,
            *model_spec, false, model_spec->decoder_cpu_layer_count);

        LogKeyInfo("Building GPU layers...");
        ret = BuildGpuLayers(*model_spec, model_partition);
        Macro_RetxFalseIf(!ret, LogError("Failed to build the GPU layers"));

        LogKeyInfo("Checking the on-device model...");
        ret = network_builder_.CheckDeviceModel(model_);
        Macro_RetxFalseIf(!ret, LogError("Something is wrong with the model"));

        network_builder_.ClearDeviceMemory();

        LogKeyInfo("Creating workers...");
        CreateGpuWorkers(*model_spec, model_partition);
    }

    int device_count = CudaUtil::DeviceCount();
    for (int device_id = 0; device_id < device_count; device_id++)
    {
        CudaUtil::LogDeviceMemoryInfo(device_id);
    }

    CudaUtil::SetDevice(default_device_id_);
#endif //USE_CUDA

    int encoder_cpu_layer_count = is_cpu_only_ ? hparams.encoder_layers
        : model_spec->encoder_cpu_layer_count;
    int decoder_cpu_layer_count = is_cpu_only_ ? hparams.decoder_layers
        : model_spec->decoder_cpu_layer_count;
    is_cpu_involved_ = decoder_cpu_layer_count > 0 || encoder_cpu_layer_count > 0;
    if (is_cpu_involved_)
    {
        size_t mem_size = 2u * 1024 * 1024 * 1024; //2G
        cpu_ctx_ = CpuInferenceWorker::CreateContext(mem_size);

        // CPU batch max 3
        config_.max_concurrent_queries = min(3, config_.max_concurrent_queries);

        if (!NetworkStructure::IsDecoderOnlyTransformer(model_spec->network_structure))
        {
            LogError("CPU inference only supports decoder-only transformers so far");
            return false;
        }

        LogKeyInfo("Building CPU layers");
        ret = BuildCpuLayers(*model_spec, encoder_cpu_layer_count,
            decoder_cpu_layer_count);

        struct ggml_init_params params_for_kv_cache;
        params_for_kv_cache.mem_size = (uint64_t)6u * 1024 * 1024 * 1024; // Todo: estimated 
        params_for_kv_cache.mem_buffer = nullptr;
        params_for_kv_cache.no_alloc = false;

        LogKeyInfo("CPU KV cache init");
        struct ggml_context *ctx_kv_cache = ggml_init(params_for_kv_cache);

        cpu_worker_ = new CpuInferenceWorker;
        int cpu_layers = min(decoder_cpu_layer_count, hparams.decoder_layers);
        ret = cpu_worker_->Init(config_, config_ex_, model_, 0, cpu_layers, ctx_kv_cache);
    }

    network_builder_.Clear();
    return ret;
}

bool InferenceEngine::Tokenize(std::vector<int> &tokens, const string &text,
    bool add_bos, TokenizationAlg alg)
{
    if (alg == TokenizationAlg::Auto) {
        alg = model_.spec.tokenization_algorithm;
    }

    std::vector<int> temp_tokens;
    string key1 = "{user_token}";
    string key2 = "{assistant_token}";
    string sub_text;
    size_t start_pos = 0;
    while (start_pos < text.size())
    {
        auto pos1 = text.find(key1, start_pos);
        auto pos2 = text.find(key2, start_pos);
        if (pos1 == string::npos && pos2 == string::npos)
        {
            sub_text = text.substr(start_pos);
            tokenizer_.Tokenize(temp_tokens, sub_text, add_bos && start_pos == 0, alg);
            tokens.insert(tokens.end(), temp_tokens.begin(), temp_tokens.end());
            break;
        }

        size_t pos = pos1 < pos2 ? pos1 : pos2;
        int key_len = pos1 < pos2 ? (int)key1.size() : (int)key2.size();
        int token_id = pos1 < pos2 ? model_.generation_config.user_token_id
            : model_.generation_config.assistant_token_id;

        if (pos > start_pos)
        {
            sub_text = text.substr(start_pos, pos - start_pos);
            tokenizer_.Tokenize(temp_tokens, sub_text, add_bos && start_pos == 0, alg);
            tokens.insert(tokens.end(), temp_tokens.begin(), temp_tokens.end());
        }

        if (token_id >= 0) {
            tokens.push_back(token_id);
        }

        start_pos = pos + key_len;
    }

    return true;
}

//int InferenceEngine::AddQuery(const string &text,
//    const SamplingStrategy::QueryOptions &query_options,
//    bool add_bos, TokenizationAlg alg)
//{
//    vector<int> q_prefix, q_suffix;
//    return AddQuery(text, query_options, q_prefix, q_suffix, add_bos, alg);
//}

int InferenceEngine::AddQuery(const EncoderInput &encoder_input,
    const DecoderPrefix &decoder_prefix,
    const SamplingStrategy::QueryOptions &query_options,
    TokenizationAlg alg)
{
    vector<int> encoder_input_tokens, decoder_prefix_tokens;
    int query_id = AddQuery(encoder_input_tokens, decoder_prefix_tokens,
        encoder_input, decoder_prefix, query_options, alg);
    return query_id;
}

int InferenceEngine::AddQuery(vector<int> &encoder_input_tokens,
    vector<int> &decoder_prefix_tokens,
    const EncoderInput &encoder_input,
    const DecoderPrefix &decoder_prefix,
    const SamplingStrategy::QueryOptions &query_options,
    TokenizationAlg alg)
{
    bool ret = true;
    encoder_input_tokens.clear();
    decoder_prefix_tokens.clear();

    query_state_lock_.lock(); //lock
    int query_count = query_state_table_.Size();
    query_state_lock_.unlock(); //unlock

    int max_query_count = config_.max_concurrent_queries;
    if (query_count >= max_query_count) {
        return 0; //busy
    }

    //LogKeyInfo("decoder_prefix.prompt_text: %s", decoder_prefix.prompt_text.c_str());
    if (!encoder_input.core_text.empty())
    {
        ret = Tokenize(encoder_input_tokens, encoder_input.core_text, false, alg);
        Macro_RetxIf(-1, !ret, LogError("Failed to tokenize the encoder core text"));
    }

    if (!encoder_input.prefix_tokens.empty())
    {
        encoder_input_tokens.insert(encoder_input_tokens.begin(),
            encoder_input.prefix_tokens.begin(),
            encoder_input.prefix_tokens.end());
    }

    if (!encoder_input.suffix_tokens.empty())
    {
        encoder_input_tokens.insert(encoder_input_tokens.end(),
            encoder_input.suffix_tokens.begin(),
            encoder_input.suffix_tokens.end());
    }

    vector<int> tokens;
    for (const auto &section : decoder_prefix.sections)
    {
        if (section.is_text)
        {
            tokens.clear();
            ret = Tokenize(tokens, section.text, false, alg);
            Macro_RetxIf(-1, !ret, LogError("Failed to tokenize the decoder text"));

            decoder_prefix_tokens.insert(decoder_prefix_tokens.end(), tokens.begin(), tokens.end());
        }
        else
        {
            if (section.token_id >= 0) {
                decoder_prefix_tokens.push_back(section.token_id);
            }
        }
    }

    tokens.clear();
    ret = Tokenize(tokens, decoder_prefix.res_prefix, false, alg);
    Macro_RetxIf(-1, !ret, LogError("Failed to tokenize the result prefix text"));

    decoder_prefix_tokens.insert(decoder_prefix_tokens.end(), tokens.begin(), tokens.end());

    int query_id = AddQuery(encoder_input_tokens, decoder_prefix_tokens, query_options);
    return query_id;
}

int InferenceEngine::AddQuery(const vector<int> &tokens,
    const SamplingStrategy::QueryOptions &query_options)
{
    vector<int> encoder_input_tokens;
    int query_id = AddQuery(encoder_input_tokens, tokens, query_options);
    return query_id;
}

int InferenceEngine::AddQuery(const vector<int> &encoder_input_tokens,
    const vector<int> &decoder_prefix_tokens,
    const SamplingStrategy::QueryOptions &query_options)
{
    const auto &model_spec = model_.spec;
    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(model_spec.network_structure);
    auto strategy_id = query_options.strategy_id;
    const string &config_json = model_spec.decoding_strategy_config;
    int max_query_count = config_.max_concurrent_queries;
    if (decoder_prefix_tokens.empty() && !is_encoder_only)
    {
        LogWarning("Empty decoder_prefix_tokens");
        return -1; //invalid query text
    }

    query_state_lock_.lock(); //lock
    int query_count = query_state_table_.Size();
    int query_id = 0; //0: busy
    if (query_count < max_query_count)
    {
        query_id = query_state_table_.Add(encoder_input_tokens, decoder_prefix_tokens,
            model_spec, (int)strategy_id, query_options.max_output_tokens);

        SamplingStrategy *strategy_ptr = GetSamplingStrategy(strategy_id);
        if (strategy_ptr != nullptr)
        {
            strategy_ptr->BeginQuery(query_id, query_options, config_json, &json_parser_);
        }
    }
    query_state_lock_.unlock(); //unlock

    return query_id;
}

bool InferenceEngine::RemoveQuery(int query_id)
{
    query_state_lock_.lock(); //lock
    bool ret = query_state_table_.Remove(query_id);
    query_state_lock_.unlock(); //unlock
    return ret;
}

int InferenceEngine::QueryCount() const
{
    LockQueryStateTable();
    int query_count = query_state_table_.Size();
    UnLockQueryStateTable();
    return query_count;
}

string InferenceEngine::GetEncoderInputTemplate(const string &model) const
{
    if (model.empty()) {
        return model_.spec.encoder_input_template;
    }

    for (const auto &model_spec : config_.models)
    {
        if (String::CaseCmp(model_spec.sid, model) == 0) {
            return model_spec.encoder_input_template;
        }
    }

    return "";
}

string InferenceEngine::GetPromptTemplate(const string &model) const
{
    if (model.empty()) {
        return model_.spec.decoder_input_template;
    }

    for (const auto &model_spec : config_.models)
    {
        if (String::CaseCmp(model_spec.sid, model) == 0) {
            return model_spec.decoder_input_template;
        }
    }

    return "";
}

void InferenceEngine::BuildEncoderInput(EncoderInput &obj, const LlmQuery &query)
{
    obj.Clear();
    const auto &model = model_;
    const auto &vocab = model.vocabulary;
    if (model.spec.hyper_params.encoder_layers <= 0) {
        return;
    }

    string template_str = GetEncoderInputTemplate();
    if (!query.encoder_input_template.empty()) {
        template_str = StringUtil::ToUtf8(query.encoder_input_template);
    }

    bool is_prefix_phase = true;
    string key, suffix_str;
    size_t template_len = template_str.size();
    size_t start_pos = 0;
    while (start_pos < template_len)
    {
        char ch = template_str[start_pos];
        size_t pos1 = ch == '{' ? start_pos : template_str.find('{', start_pos);
        size_t pos2 = pos1 == string::npos ? string::npos : template_str.find('}', pos1 + 1);
        if (pos2 == string::npos)
        {
            obj.core_text += template_str.substr(start_pos);
            obj.core_text += suffix_str;
            obj.suffix_tokens.clear();
            break;
        }

        if (pos1 > start_pos)
        {
            obj.core_text += template_str.substr(start_pos, pos1 - start_pos);
            obj.core_text += suffix_str;
            obj.suffix_tokens.clear();
        }

        bool is_prefix = obj.core_text.empty();
        int token_id = -1;

        key = template_str.substr(pos1 + 1, pos2 - pos1 - 1);
        auto iter = key_map_.find(key);
        KeyId key_id = iter == key_map_.end() ? KeyId::UnknownKey : iter->second;

        bool is_as_token = false;
        switch (key_id)
        {
        case KeyId::SystemPrompt:
            obj.core_text += StringUtil::ToUtf8(query.system_prompt);
            is_prefix = false;
            break;
        case KeyId::QueryText:
            obj.core_text += StringUtil::ToUtf8(query.text);
            is_prefix = false;
            break;
        case KeyId::QueryTextWithPreSpace:
            obj.core_text += " ";
            obj.core_text += StringUtil::ToUtf8(query.text);
            is_prefix = false;
            break;
        case KeyId::ResPrefix:
            obj.core_text += StringUtil::ToUtf8(query.response_prefix);
            is_prefix = false;
            break;
        case KeyId::Bos:
            token_id = vocab.bos();
            is_as_token = true;
            break;
        case KeyId::Eos:
            token_id = vocab.eos();
            is_as_token = true;
            break;
        case KeyId::UserToken:
            token_id = model.generation_config.user_token_id;
            break;
        case KeyId::AssistantToken:
            token_id = model.generation_config.assistant_token_id;
            break;
        default:
            if (key.size() > 1 && key[0] == '#') {
                token_id = atoi(key.c_str() + 1);
            }
            else {
                token_id = vocab.StrToId(key);
            }
            is_as_token = true;
            break;
        }

        start_pos = pos2 + 1;
        if (!is_as_token || !obj.core_text.empty()) {
            is_prefix_phase = false;
        }

        if (!is_as_token)
        {
            obj.core_text += suffix_str;
            obj.suffix_tokens.clear();
            continue;
        }

        int max_token_id = vocab.Size() - 1;
        if (model.encoder_embeddings != nullptr) {
            max_token_id = max(model.encoder_embeddings->ne[1] - 1, max_token_id);
        }

        if (token_id < 0 || token_id > max_token_id)
        {
            LogWarning("Invalid token: %s", key.c_str());
            continue;
        }

        if (is_prefix)
        {
            obj.prefix_tokens.push_back(token_id);
        }
        else
        {
            suffix_str += key;
            obj.suffix_tokens.push_back(token_id);
        }
    }
}

void InferenceEngine::BuildDecoderInput(DecoderPrefix &obj, const LlmQuery &query)
{
    obj.Clear();
    const auto &model = model_;
    const auto &hparams = model.spec.hyper_params;
    const auto &vocab = model.vocabulary;
    if (hparams.decoder_layers <= 0) {
        return;
    }

    string template_str = GetPromptTemplate();
    if (!query.decoder_input_template.empty()) {
        template_str = StringUtil::ToUtf8(query.decoder_input_template);
    }

    obj.res_prefix = StringUtil::ToUtf8(query.response_prefix);

    string section_text;
    string key;
    size_t template_len = template_str.size();
    size_t start_pos = 0;
    while (start_pos < template_len)
    {
        char ch = template_str[start_pos];
        size_t pos1 = ch == '{' ? start_pos : template_str.find('{', start_pos);
        size_t pos2 = pos1 == string::npos ? string::npos : template_str.find('}', pos1 + 1);
        if (pos2 == string::npos)
        {
            section_text += template_str.substr(start_pos);
            break;
        }

        if (pos1 > start_pos)
        {
            section_text += template_str.substr(start_pos, pos1 - start_pos);
        }

        int token_id = -1;

        key = template_str.substr(pos1 + 1, pos2 - pos1 - 1);
        auto iter = key_map_.find(key);
        KeyId key_id = iter == key_map_.end() ? KeyId::UnknownKey : iter->second;

        if (key_id == KeyId::ResPrefix)
        {
            //obj.prompt_text += StringUtil::ToUtf8(query.response_prefix);
            break;
        }

        bool is_as_token = false;
        switch (key_id)
        {
        case KeyId::SystemPrompt:
            section_text += StringUtil::ToUtf8(query.system_prompt);
            break;
        case KeyId::QueryText:
            section_text += StringUtil::ToUtf8(query.text);
            break;
        case KeyId::QueryTextWithPreSpace:
            section_text += " ";
            section_text += StringUtil::ToUtf8(query.text);
            break;
        case KeyId::Bos:
            token_id = vocab.bos();
            is_as_token = true;
            break;
        case KeyId::Eos:
            token_id = vocab.eos();
            is_as_token = true;
            break;
        case KeyId::UserToken:
            token_id = model.generation_config.user_token_id;
            is_as_token = true;
            break;
        case KeyId::AssistantToken:
            token_id = model.generation_config.assistant_token_id;
            is_as_token = true;
            break;
        default:
            if (key.size() > 1 && key[0] == '#') {
                token_id = atoi(key.c_str() + 1);
            }
            else {
                token_id = vocab.StrToId(key);
            }
            is_as_token = true;
            break;
        }

        start_pos = pos2 + 1;

        if (is_as_token)
        {
            if (!section_text.empty())
            {
                DecoderPrefixSection section;
                section.is_text = true;
                section.text = section_text;
                obj.sections.push_back(section);
                section_text.clear();
            }

            int max_token_id = vocab.Size() - 1;
            if (model.decoder_embeddings != nullptr) {
                max_token_id = max(model.decoder_embeddings->ne[1] - 1, max_token_id);
            }

            if (token_id < 0 || token_id > max_token_id)
            {
                LogWarning("Invalid token: %s", key.c_str());
                continue;
            }

            DecoderPrefixSection section;
            section.is_text = false;
            section.token_id = token_id;
            section.text = key;
            obj.sections.push_back(section);
        }
    }

    if (!section_text.empty())
    {
        DecoderPrefixSection section;
        section.is_text = true;
        section.text = section_text;
        obj.sections.push_back(section);
    }
}

string InferenceEngine::BuildPrompt(const string &prompt_template, const LlmQuery &query)
{
    string question = StringUtil::ToUtf8(query.text);
    string res_prefix = StringUtil::ToUtf8(query.response_prefix);
    string system_prompt = StringUtil::ToUtf8(query.system_prompt);
    return BuildPrompt(prompt_template, question, res_prefix, system_prompt);
}

string InferenceEngine::BuildPrompt(const string &prompt_template, const string &question,
    const string &answer_prefix, const string &system_prompt)
{
    string utf8_prompt = prompt_template;
    string question_with_space = " " + question;
    string system_prompt_key = "{system_prompt}";
    string question_key1 = "{query}";
    string question_key2 = "{_query}";
    string bos_key = "{bos}";
    auto pos1 = utf8_prompt.find(question_key1);
    auto pos2 = utf8_prompt.find(question_key2);

    if (pos1 != string::npos || pos2 != string::npos)
    {
        if (pos1 != string::npos) {
            String::ReplaceAll(utf8_prompt, question_key1, question);
        }
        if (pos2 != string::npos) {
            String::ReplaceAll(utf8_prompt, question_key2, question_with_space);
        }

        auto pos = utf8_prompt.find(system_prompt_key);
        if (pos != string::npos) {
            String::ReplaceAll(utf8_prompt, system_prompt_key, system_prompt);
        }

        int bos_id = model_.vocabulary.bos();
        pos = utf8_prompt.find(bos_key);
        if (pos != string::npos && bos_id >= 0)
        {
            string bos_str = model_.vocabulary.IdToStr(bos_id);
            String::ReplaceAll(utf8_prompt, bos_key, bos_str);
        }

        pos = utf8_prompt.find("{res_prefix}");
        if (pos != string::npos) {
            utf8_prompt = utf8_prompt.substr(0, pos);
        }
        if (!answer_prefix.empty()) {
            utf8_prompt += answer_prefix;
        }
    }
    else
    {
        utf8_prompt = question;
        if (!answer_prefix.empty()) {
            utf8_prompt += answer_prefix;
        }
    }

    return utf8_prompt;
}

//static
bool InferenceEngine::ParseQueryJson(LlmQuery &query, const wstring &query_str,
    const JsonParser &json_parser)
{
    query.Clear();

    JsonDoc jdoc;
    bool ret = json_parser.Parse(jdoc, query_str);
    Macro_RetFalseIf(!ret);

    JsonObject jobj = jdoc.GetJObject();
    ret = ParseQueryJson(query, jobj, jdoc);
    return ret;
}

//static
bool InferenceEngine::ParseQueryJson(LlmQuery &query, const JsonObject &jobj,
    const JsonDoc &jdoc)
{
    query.Clear();

    jobj.GetFieldValue(query.system_prompt, L"system_prompt", jdoc);
    bool ret1 = jobj.GetFieldValue(query.text, L"text", jdoc);
    bool ret2 = jobj.GetFieldValue(query.response_prefix, L"res_prefix", jdoc);

    jobj.GetFieldValue(query.encoder_input_template, L"encoder_input_template", jdoc);
    jobj.GetFieldValue(query.decoder_input_template, L"decoder_input_template", jdoc);

    //query.q_prefix.clear();
    //string str;
    //JsonArray jarray;
    //jobj.GetFieldValue(jarray, L"q_prefix", jdoc);
    //for (uint32_t idx = 0; idx < jarray.size; idx++)
    //{
    //    jarray.items[idx].GetString(str);
    //    query.q_prefix.push_back(str);
    //}

    //query.q_suffix.clear();
    //jobj.GetFieldValue(jarray, L"q_suffix", jdoc);
    //for (uint32_t idx = 0; idx < jarray.size; idx++)
    //{
    //    jarray.items[idx].GetString(str);
    //    query.q_suffix.push_back(str);
    //}

    return ret1 || ret2;
}

bool InferenceEngine::Infer(InferenceResult &res)
{
    bool ret = true;
    if (is_cpu_only_)
    {
        ret = Infer_CpuOnly(res, config_.cpu_threads);
    }
    else
    {
#if defined(USE_CUDA)
        ret = Infer_Gpu(res);
#endif //USE_CUDA
    }

    return ret;
}

#if defined(USE_CUDA)

bool InferenceEngine::Infer_Gpu(InferenceResult &res)
{
    bool ret = true;
    auto net_type = model_.spec.network_structure;
    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(net_type);

    ClearResult(res);
    TaskMonitor tm_e2e;
    const auto &model = model_;

    LocalInput input;
    vector<int> saturated_queries;
    bool has_query = GetLocalInput(input, saturated_queries);
    if (!has_query) {
        return true;
    }

    local_device_heap_.Clear();

    DeviceTensor cpu_output_device_tensor;
    if (is_cpu_involved_ && !input.query_list.empty())
    {
        gf_ = {};
        gf_.n_threads = config_.cpu_threads;

        ostream *tensor_writer = config_.debug.is_study_mode && config_.debug.show_tensors
            ? &tensor_writer_ : nullptr;

        int token_num = (int)input.tokens.size();
        ggml_tensor *embd = ggml_new_tensor_1d(cpu_ctx_, GGML_TYPE_I32, token_num);
        memcpy(embd->data, input.tokens.data(), token_num * ggml_element_size(embd));
        ggml_tensor *input_tensor = ggml_get_rows(cpu_ctx_, model.std_network.ggml_net.decoder_embeddings, embd);
        ggml_tensor *output_tensor = cpu_worker_->Inference(input.query_list,
            input_tensor, gf_, cpu_ctx_, tensor_writer);

        HostTensor host_output(false);
        NetworkBuilder::ConvertGgmlToHost(host_output, output_tensor);

        vector<half> host_array;
        DeviceTensorUtil::GetFP16List(host_array, host_output);
        cpu_output_device_tensor.FromHost(host_array.data(), host_output.ne[0],
            host_output.ne[1], host_output.ne[2]);

        input.input_tensor = &cpu_output_device_tensor;

        ggml_clear_context(cpu_ctx_);
    }

    LocalOutput output;
    if (!input.query_list.empty())
    {
        BuildLocalOutput(output, res.perf_stat, input);

        if (ret && input.is_encoder && !is_encoder_only)
        {
            for (const auto &query : input.query_list)
            {
                query_state_table_.UpdateEncoderEnd(query.query_id, *output.layer_output,
                    query.start_row, query.token_num);
            }

            res.perf_stat.time_map[0] = tm_e2e.GetElapsedTime(false) / 1000.0f;
            tm_e2e.Start();
            if (config_.debug.is_study_mode && config_.debug.enable_perf_stat) {
                res.perf_stat_pre = res.perf_stat;
            }

            GetLocalInput(input, saturated_queries);
            BuildLocalOutput(output, res.perf_stat, input);
        }
    }
    Macro_RetFalseIf(!ret);

    DeviceTensor output_tensor(false);
    if (!input.query_list.empty())
    {
        output_tensor.data_type = output.layer_output->data_type;
        output_tensor.SetStructure(output.layer_output->dim, output.layer_output->ne);
        if (output.device_id == default_device_id_)
        {
            output_tensor.data = output.layer_output->data;
        }
        else
        {
            output_tensor.data = local_device_heap_.NewHalfArray(output.layer_output->size);
            output_tensor.CopyFromDevice(output.layer_output->data_f16(),
                output.layer_output->size * sizeof(half));
        }
    }

    for (int idx = 0; idx < (int)input.query_list.size(); idx++)
    {
        const QueryProcInput &query_input = input.query_list[idx];
        auto *query_res = new QueryInferenceResult;
        query_res->query_id = query_input.query_id;
        query_res->prefix_len = query_input.prefix_len;

        TaskMonitor tm;
        SamplingStrategyId strategy_id = (SamplingStrategyId)query_input.sampling_strategy;
        if (is_encoder_only)
        {
            int mask_token_id = model_.mask_token_id;
            int row_idx = query_input.start_row;
            const auto &input_tokens = query_input.state->encoder_input_tokens;
            for (int tok_idx = 0; tok_idx < (int)input_tokens.size(); tok_idx++)
            {
                if (input_tokens[tok_idx] == mask_token_id)
                {
                    row_idx = query_input.start_row + tok_idx;
                    break;
                }
            }

            EncoderSampleTokens(query_res->next_tokens, output_tensor, row_idx,
                query_input.query_id, strategy_id);
        }
        else
        {
            if (config_.return_output_tensors)
            {
                output_tensor.CopyToHost(query_res->output_tensor,
                    query_input.start_row, query_input.token_num);
            }

            int row_idx = query_input.start_row + query_input.token_num - 1;
            SampleTokens(query_res->next_tokens, output_tensor, row_idx,
                query_input.query_id, query_input.state->prefix_tokens,
                query_input.state->decoder_tokens, strategy_id);
            //CalculateNextCandidates(res.next_tokens, *out);
        }

        if (config_.debug.is_study_mode && config_.debug.enable_perf_stat)
        {
            float value = tm.GetElapsedTime(false) / 1000.0f;
            int key = 2000000;
            res.perf_stat.time_map[key] = value;
        }

        //LogKeyInfo("start_row: %d, row_idx: %d", query_input.start_row, row_idx);
        //output_tensor.Print(cout, 8, 32, 8, "logits:\n") << endl;

        res.items.push_back(query_res);
    }

    HandleSaturatedQueries(res, saturated_queries, model);

    if (config_.debug.enable_perf_stat) {
        res.perf_stat.time_map[0] = tm_e2e.GetElapsedTime(false) / 1000.0f;
    }

    return ret;
}

#endif //USE_CUDA

// CPU-only or CPU+GPU inference
bool InferenceEngine::Infer_CpuOnly(InferenceResult &res, int cpu_threads)
{
    bool ret = true;
    ClearResult(res);
    const auto &model = model_;

    LocalInput input;
    vector<int> saturated_queries;
    bool has_query = GetLocalInput(input, saturated_queries);
    if (!has_query) {
        return true;
    }

    gf_ = {};
    gf_.n_threads = cpu_threads;

    ostream *tensor_writer = config_.debug.is_study_mode && config_.debug.show_tensors
        ? &tensor_writer_ : nullptr;

    //const auto &hparams = model.spec.hyper_params;
    int token_num = (int)input.tokens.size();

    ggml_tensor *embd = ggml_new_tensor_1d(cpu_ctx_, GGML_TYPE_I32, token_num);
    memcpy(embd->data, input.tokens.data(), token_num * ggml_element_size(embd));
    ggml_tensor *input_tensor = ggml_get_rows(cpu_ctx_, model.std_network.ggml_net.decoder_embeddings, embd);
    ggml_tensor *output_tensor = cpu_worker_->Inference(input.query_list,
        input_tensor, gf_, cpu_ctx_, tensor_writer);

    for (int idx = 0; idx < (int)input.query_list.size(); idx++)
    {
        const QueryProcInput &query_input = input.query_list[idx];
        auto *query_res = new QueryInferenceResult;
        query_res->query_id = query_input.query_id;
        query_res->prefix_len = query_input.prefix_len;

        if (config_.return_output_tensors)
        {
            HostTensor host_output(false);
            NetworkBuilder::ConvertGgmlToHost(host_output, output_tensor);
            host_output.CopyTo(query_res->output_tensor, query_input.start_row,
                query_input.token_num);
        }

        int row_idx = query_input.start_row + query_input.token_num - 1;
        SamplingStrategyId strategy_id = (SamplingStrategyId)query_input.sampling_strategy;
        SampleTokensCpu(query_res->next_tokens, output_tensor, row_idx,
            query_input.query_id, query_input.state->prefix_tokens,
            query_input.state->decoder_tokens, strategy_id);
        res.items.push_back(query_res);
    }

    HandleSaturatedQueries(res, saturated_queries, model);

    //ggml_free(cpu_ctx_);
    ggml_clear_context(cpu_ctx_);
    return ret;
}

bool InferenceEngine::GetLocalInput(LocalInput &input, vector<int> &saturated_queries)
{
    input.Clear();
    const auto &model = model_;
    auto net_type = model.spec.network_structure;
    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(net_type);

    map<int, const QueryState*> query_map;
    int max_query_num = -1;
    int max_token_num = 256;
    query_state_lock_.lock(); //lock
    query_state_table_.Get(query_map, net_type, model.spec.max_context_len,
        max_query_num, max_token_num);
    query_state_lock_.unlock(); //unlock

    if (query_map.empty()) {
        return false;
    }

    for (auto iter = query_map.begin(); iter != query_map.end(); iter++)
    {
        const auto *state = iter->second;
        if (state->next_net == 0 && !is_decoder_only) {
            input.is_encoder = true;
        }

        const auto &state_tokens = input.is_encoder ? state->encoder_input_tokens
            : state->decoder_tokens;
        int total_len = state->prefix_len + (int)state_tokens.size();
        int output_len = state->prefix_len - state->initial_prefix_len;
        int max_output_len = state->max_output_tokens;
        if (state->prefix_len >= model.spec.max_context_len
            || (max_output_len >= 0 && output_len >= max_output_len))
        {
            saturated_queries.push_back(state->query_id);
        }
        else if (total_len > model.spec.max_context_len)
        {
            saturated_queries.push_back(state->query_id);
        }
        else
        {
            QueryProcInput query;
            query.query_id = state->query_id;
            query.proc_id = state->proc_id;
            query.sampling_strategy = state->sampling_strategy;
            query.prefix_len = state->prefix_len;
            query.token_num = (int)state_tokens.size();
            query.start_row = (int)input.tokens.size();
            query.state = state;
            input.query_list.push_back(query);
        }

        input.tokens.insert(input.tokens.end(), state_tokens.begin(), state_tokens.end());
    }

    if (config_.debug.is_study_mode)
    {
        LogKeyInfo("normal queries: %d, tokens: %d, saturated queries: %d",
            (int)input.query_list.size(), (int)input.tokens.size(),
            (int)saturated_queries.size());
        if (!input.query_list.empty())
        {
            const QueryProcInput &query = input.query_list[0];
            LogKeyInfo("First query: id = %d, proc_id = %d, prefix_len = %d, token_num = %d",
                query.query_id, query.proc_id, query.prefix_len, query.token_num);
        }
    }

    return true;
}

//static
void InferenceEngine::HandleSaturatedQueries(InferenceResult &res,
    const vector<int> &saturated_queries, const TransformerModel &model)
{
    int eos = model.vocabulary.eos();
    for (int query_id : saturated_queries)
    {
        auto *query_res = new QueryInferenceResult;
        query_res->query_id = query_id;
        query_res->prefix_len = model.spec.max_context_len;
        query_res->next_tokens.push_back(IdWeight<float>(eos, 1.0f));
        res.items.push_back(query_res);
    }
}

#if defined(USE_CUDA)

bool InferenceEngine::BuildLocalOutput(LocalOutput &output,
    InferencePerfStat &perf_stat, const LocalInput &input)
{
    bool ret = true;
    bool is_by_tensor = model_.spec.multi_gpu_strategy != MultiGpuStrategy::BY_LAYER;
    if (!input.query_list.empty())
    {
        if (is_by_tensor) {
            ret = Infer_TensorParallelism(output, perf_stat, input);
        }
        else {
            ret = Infer_Std(output, perf_stat, input);
        }
    }

    return ret;
}

bool InferenceEngine::Infer_Std(LocalOutput &output, InferencePerfStat &perf_stat,
    const LocalInput &input)
{
    bool ret = true;
    const auto &hparams = model_.spec.hyper_params;
    //int token_num = (int)input.tokens.size();

    TaskMonitor tm;
    DeviceTensor embd_tensor(false);
    if (input.input_tensor == nullptr) {
        GetEmbdTensor(embd_tensor, input, input.is_encoder);
    }

    if (config_.debug.enable_perf_stat) {
        perf_stat.time_map[1] = tm.GetElapsedTime(false) / 1000.0f;
    }
    //tm.ShowElapsedTime();

    //embd_tensor.Print(cout, 8, 8, 8, "embd_tensor:\n") << endl;

    int layer_num = input.is_encoder ? hparams.encoder_layers : hparams.decoder_layers;
    //int start_gpu_layer = 0;
    //int gpu_layer_num = input.is_encoder ? model_.spec.gpu_encoder_layer_count
    //    : model_.spec.gpu_decoder_layer_count;
    //int end_gpu_layer = min((int)layer_num, gpu_layer_num);
    const DeviceTensor *layer_input = input.input_tensor != nullptr
        ? input.input_tensor : &embd_tensor;
    const DeviceTensor *layer_output = nullptr;
    int worker_num = (int)gpu_worker_groups_.size();
    int end_layer = layer_num;
    int output_tensor_device = 0;
    for (int worker_id = 0; worker_id < worker_num; worker_id++)
    {
        //LogKeyInfo("##### Worker %d", worker_id);
        auto *worker_group = gpu_worker_groups_[worker_id];
        auto *worker_ptr = worker_group->workers[0];
        if (worker_ptr->StartLayer(input.is_encoder) >= end_layer) {
            break;
        }

        ostream *tensor_writer = config_.debug.is_study_mode && config_.debug.show_tensors
            ? &tensor_writer_ : nullptr;
        worker_ptr->SetInput(layer_input, perf_stat, tensor_writer, input.query_list,
            input.tokens, end_layer, input.is_encoder);
        worker_ptr->Create();
        worker_ptr->Join();

        layer_output = worker_ptr->GetOutput();
        layer_input = layer_output;
        if (layer_output == nullptr) {
            return false;
        }

        output_tensor_device = worker_ptr->device_id();
    }

    output.layer_output = layer_output;
    output.device_id = output_tensor_device;
    return ret;
}

bool InferenceEngine::Infer_TensorParallelism(LocalOutput &output,
    InferencePerfStat &perf_stat, const LocalInput &input)
{
    bool ret = true;
    int group_num = (int)gpu_worker_groups_.size();
    const auto &hparams = model_.spec.hyper_params;
    //int token_num = (int)input.tokens.size();

    TaskMonitor tm;
    DeviceTensor embd_tensor(false);
    if (input.input_tensor == nullptr) {
        GetEmbdTensor(embd_tensor, input, input.is_encoder);
    }

    if (config_.debug.enable_perf_stat) {
        perf_stat.time_map[1] = tm.GetElapsedTime(false) / 1000.0f;
    }
    //tm.ShowElapsedTime();
    //embd_tensor.Print(cout, 8, 8, 8, "embd_tensor:\n") << endl;

    ostream *tensor_writer = config_.debug.is_study_mode && config_.debug.show_tensors
        ? &tensor_writer_ : nullptr;

    const DeviceTensor *layer_input = input.input_tensor != nullptr
        ? input.input_tensor : &embd_tensor;
    const DeviceTensor *layer_output = nullptr;
    int output_tensor_device = 0;

    uint32_t layer_num = (uint32_t)hparams.decoder_layers;
    int end_layer = layer_num;
    for (int group_idx = 0; group_idx < group_num; group_idx++)
    {
        const auto &device_group = model_.spec.device_groups[group_idx];
        auto *worker_group = gpu_worker_groups_[group_idx];
        int worker_num = (int)worker_group->workers.size();

        worker_group->global_data.Clear();
        worker_group->global_data.Init(worker_num);

        for (int worker_id = 0; worker_id < worker_num; worker_id++)
        {
            //LogKeyInfo("##### Worker %d", worker_id);
            auto *worker_ptr = worker_group->workers[worker_id];
            //worker_ptr->SetInput(embd_tensors[worker_id], perf_stat,
            //    input.query_list, end_layer, is_by_layer);
            worker_ptr->SetInput(layer_input, perf_stat, tensor_writer,
                input.query_list, input.tokens, end_layer);
            worker_ptr->SetGlobalData(worker_group->global_data, device_group);
            worker_ptr->Create();
        }

        for (int worker_id = 0; worker_id < worker_num; worker_id++)
        {
            auto *worker_ptr = worker_group->workers[worker_id];
            worker_ptr->Join();

            if (worker_id + 1 == worker_num)
            {
                layer_output = worker_ptr->GetOutput();
                output_tensor_device = worker_ptr->device_id();
            }
        }

        layer_input = layer_output;
    }

    output.layer_output = layer_output;
    output.device_id = output_tensor_device;
    if (layer_output == nullptr) {
        ret = false;
    }

    //LogKeyInfo("End of Infer_TensorParallelism. Return %s", ret ? "true" : "false");
    return ret;
}

bool InferenceEngine::GetEmbdTensor(DeviceTensor &embd_tensor,
    const LocalInput &input, bool is_encoder)
{
    bool ret = true;
    const auto &hparams = model_.spec.hyper_params;
    int token_num = (int)input.tokens.size();
    const auto &net = model_.std_network;

    embd_tensor.data_type = ElementType::F16;
    embd_tensor.data = local_device_heap_.NewHalfArray(token_num * hparams.embd_dims);
    embd_tensor.SetStructure(hparams.embd_dims, token_num);

    vector<half> half_array;
    //LogKeyInfo("Building the embedding tensor for %d token(s)...", token_num);
    for (int idx = 0; idx < token_num; idx++)
    {
        int token_id = input.tokens[idx];
        //LogKeyInfo("token_id: %d", token_id);
        void *target_row = embd_tensor.RowData(idx);
        int byte_num = sizeof(half) * hparams.embd_dims;
        const auto *gpu_embeddings = is_encoder ? net.device_net.encoder_embeddings
            : net.device_net.decoder_embeddings;
        if (gpu_embeddings != nullptr)
        {	
            const void *src_row = gpu_embeddings->RowData(token_id);
            ret = CudaUtil::DeviceToDeviceMemcpy(target_row, src_row, byte_num);
            //LogKeyInfo("bytes_per_row: %d, %d, src_ptr: %I64u, target_ptr: %I64u",
            //    embd_tensor.bytes_per_row, model_.gpu_embeddings->bytes_per_row,
            //    src_row, target_row);
        }
        else
        {
            const HostHalfBuffer &cpu_embeddings = is_encoder ? net.host_net.encoder_embeddings
                : net.host_net.decoder_embeddings;
            const half *src_row = cpu_embeddings.data() + token_id * hparams.embd_dims;
            ret = CudaUtil::HostToDeviceMemcpy(target_row, src_row, byte_num);
            //DeviceTensorUtil::GetFP16List(half_array, *src_row);
            //LogKeyInfo("first embed weight: %.5f", (float)cpu_embeddings.data()[0]);
        }

        Macro_RetFalseIf(!ret);
    }

    return ret;
}

#endif //USE_CUDA

bool InferenceEngine::CommitInferenceResult(const map<int, QueryNextToken> &query_map)
{
    bool ret = true;
    query_state_lock_.lock(); //lock
    for (auto iter = query_map.begin(); iter != query_map.end(); iter++)
    {
        int query_id = iter->first;
        const auto &next_token = iter->second;

        if (next_token.is_end)
        {
            const auto *query_state = query_state_table_.GetQueryState(query_id);
            if (query_state != nullptr)
            {
                auto strategy_id = (SamplingStrategyId)query_state->sampling_strategy;
                SamplingStrategy *strategy_ptr = GetSamplingStrategy(strategy_id);
                if (strategy_ptr != nullptr) {
                    strategy_ptr->EndQuery(query_id);
                }
            }
        }

        bool is_succ = query_state_table_.Update(query_id, next_token.id, next_token.is_end);
        ret = ret && is_succ;
    }
    query_state_lock_.unlock(); //unlock

    return ret;
}

std::string InferenceEngine::TokenIdToStr(int id, bool enable_decoding) const
{
    return model_.vocabulary.IdToStr(id, enable_decoding);
}

int InferenceEngine::TokenStrToId(const std::string &str) const
{
    return model_.vocabulary.StrToId(str);
}

std::string InferenceEngine::OutputTokenIdToStr(int id, bool enable_decoding) const
{
    bool has_output_vocab = model_.output_vocabulary.Size() > 0;
    const auto &vocab = has_output_vocab ? model_.output_vocabulary : model_.vocabulary;
    return vocab.IdToStr(id, enable_decoding);
}

int InferenceEngine::OutputTokenStrToId(const std::string &str) const
{
    bool has_output_vocab = model_.output_vocabulary.Size() > 0;
    const auto &vocab = has_output_vocab ? model_.output_vocabulary : model_.vocabulary;
    return vocab.StrToId(str);
}

//static
bool InferenceEngine::LoadConfig(InferenceConfig &config, const string &config_path,
    const string &section, const string &data_root_dir)
{
    ElementTypeMap element_type_map;
    MultiGpuStrategyMap multi_gpu_strategy_map;
    DecodingStrategyMap decoding_strategy_map;
    TensorCommon::InitElementTypeMap(element_type_map);
    TransformerModel::InitMultiGpuStrategyMap(multi_gpu_strategy_map);
    DecodingStrategies::InitStrategyMap(decoding_strategy_map);

    JsonParser jparser;
    jparser.Init();

    ConfigData cfg_data;
    bool ret = cfg_data.Load(config_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the config-data"));

    if (!data_root_dir.empty()) {
        cfg_data.AddMacro("data_root_dir", data_root_dir);
    }

    //prompt templates
    ret = LoadPromptTemplates(config.prompt_templates, cfg_data, "prompt_templates");
    Macro_RetFalseIf(!ret);

    //devices
    ret = LoadDeviceGroups(config.device_groups, cfg_data, section, "devices");
    Macro_RetFalseIf(!ret);

    if (config.device_groups.empty())
    {
        vector<int> new_group;
        new_group.push_back(0);
        config.device_groups.push_back(new_group);
    }

    //string devices_str;
    //cfg_data.GetItem(section, "devices", devices_str, false);
    //vector<string> columns;
    //String::Split(devices_str, columns, ",;");
    //for (auto &col_str : columns)
    //{
    //    String::Trim(col_str);
    //    if (!col_str.empty()) {
    //        config.devices.push_back(atoi(col_str.c_str()));
    //    }
    //}

    int cpu_layer_count = 0;
    cfg_data.GetItem(section, "cpu_layer_count", cpu_layer_count, false);
    config.encoder_cpu_layer_count = cpu_layer_count;
    config.decoder_cpu_layer_count = cpu_layer_count;

    cfg_data.GetItem(section, "encoder_cpu_layer_count", config.encoder_cpu_layer_count, false);
    cfg_data.GetItem(section, "decoder_cpu_layer_count", config.decoder_cpu_layer_count, false);

    //models
    string str;
    LogKeyInfo("Loading model specifications...");
    ret = ret && cfg_data.GetItem(section, "models", str, true);

    vector<string> model_id_list;
    String::Split(str, model_id_list, ",;");
    for (const string &sid : model_id_list)
    {
        ModelSpec spec;
        spec.sid = sid;
        String::Trim(spec.sid);

        string model_section = "model." + spec.sid;
        ret = LoadModelSpec(spec, cfg_data, model_section, element_type_map,
            multi_gpu_strategy_map, decoding_strategy_map, jparser);
        if (!ret)
        {
            LogError("Failed to load the specification of model %s", spec.sid.c_str());
            return false;
        }

        if (spec.device_groups.empty())
        {
            //copy from config.device_groups
            for (const auto &group : config.device_groups)
            {
                vector<int> new_group;
                new_group.insert(new_group.end(), group.begin(), group.end());
                spec.device_groups.push_back(new_group);
            }
        }

        int group_num = (int)spec.device_groups.size();
        if (group_num > 0)
        {
            int group_size = (int)spec.device_groups[0].size();
            spec.multi_gpu_strategy = group_size <= 1 ? MultiGpuStrategy::BY_LAYER
                : (group_num == 1 ? MultiGpuStrategy::BY_TENSOR : MultiGpuStrategy::HYBRID);
        }

        if (spec.encoder_cpu_layer_count < 0) {
            spec.encoder_cpu_layer_count = config.encoder_cpu_layer_count;
        }
        if (spec.decoder_cpu_layer_count < 0) {
            spec.decoder_cpu_layer_count = config.decoder_cpu_layer_count;
        }

        //Do not trim, otherwise the ending {\n} will be trimmed
        //String::Trim(spec.decoder_input_template);
        if (!spec.decoder_input_template.empty()
            && spec.decoder_input_template.find('{') == string::npos)
        {
            auto iter = config.prompt_templates.find(spec.decoder_input_template);
            if (iter == config.prompt_templates.end())
            {
                LogWarning("Cannot find the prompt template of model %s", spec.sid.c_str());
                spec.decoder_input_template.clear();
            }
            else
            {
                spec.decoder_input_template = iter->second;
            }
        }

        config.models.push_back(spec);
    }

    //
    cfg_data.GetItem(section, "max_concurrent_queries", config.max_concurrent_queries, false);

    cfg_data.GetItem(section, "cpu_threads", config.cpu_threads, false);

    cfg_data.GetItem(section, "return_output_tensors", config.return_output_tensors, false);

    //debug
    cfg_data.GetItem(section, "is_study_mode", config.debug.is_study_mode, false);
    cfg_data.GetItem(section, "show_tensors", config.debug.show_tensors, false);

    return ret;
}

//static
bool InferenceEngine::LoadModelSpec(ModelSpec &spec, const ConfigData &cfg_data,
    const string &section, const ElementTypeMap &element_type_map,
    const MultiGpuStrategyMap &multi_gpu_strategy_map,
    const DecodingStrategyMap &decoding_strategy_map,
    const JsonParser &jparser)
{
    string str;
    bool ret = cfg_data.GetItem(section, "model_dir", spec.dir, true);
    Macro_RetFalseIf(!ret);
    if (spec.dir.empty())
    {
        LogError("The directory of model \"%s\" should not be empty", spec.sid.c_str());
        return false;
    }

    spec.spec_file.clear();
    ret = cfg_data.GetItem(section, "model_specification_file", spec.spec_file, false);
    if (!ret) {
        ret = cfg_data.GetItem(section, "model_spec_file", spec.spec_file, false);
    }
    if (spec.spec_file.empty())
    {
        LogError("The specification file of model \"%s\" should not be empty", spec.sid.c_str());
        return false;
    }

    Path path;
    path.SetDir(spec.dir, true, true);
    spec.dir = path.GetDir();

    cfg_data.GetItem(section, "decoding_strategy", spec.decoding_strategy, false);
    if (!spec.decoding_strategy.empty())
    {
        bool is_json = spec.decoding_strategy[0] == L'{';
        if (is_json)
        {
            spec.decoding_strategy_config = spec.decoding_strategy;
            spec.decoding_strategy.clear();

            JsonDoc jdoc;
            ret = jparser.ParseUtf8(jdoc, spec.decoding_strategy_config);
            if (!ret)
            {
                LogError("Invalid JSON format in the %s of model %s",
                    "decoding strategy configuration", spec.sid.c_str());
                return false;
            }

            JsonObject jobj = jdoc.GetJObject();
            ret = jobj.GetFieldValue(spec.decoding_strategy, L"name", jdoc);
            if (!ret)
            {
                LogError("The \"name\" field is missing in the %s of model %s",
                    "decoding strategy configuration", spec.sid.c_str());
                return false;
            }
        }
        else
        {
            auto iter = decoding_strategy_map.find(spec.decoding_strategy);
            if (iter == decoding_strategy_map.end())
            {
                LogError("Invalid decoding_strategy for model %s", spec.sid.c_str());
                return false;
            }
        }
    }

    cfg_data.GetItem(section, "encoder_input_template", spec.encoder_input_template, false);
    String::ReplaceAll(spec.encoder_input_template, "{\\n}", "\n");

    cfg_data.GetItem(section, "decoder_input_template", spec.decoder_input_template, false);
    cfg_data.GetItem(section, "prompt_template", spec.decoder_input_template, false);
    String::ReplaceAll(spec.decoder_input_template, "{\\n}", "\n");

    str.clear();
    cfg_data.GetItem(section, "host_weight_data_type", str, false);
    if (!str.empty())
    {
        auto iter = element_type_map.find(str);
        if (iter == element_type_map.end())
        {
            LogError("Invalid host_weight_data_type for model %s", spec.sid.c_str());
            return false;
        }
        spec.host_weight_data_type = iter->second;
    }

    str.clear();
    cfg_data.GetItem(section, "device_weight_data_type", str, false);
    if (!str.empty())
    {
        auto iter = element_type_map.find(str);
        if (iter == element_type_map.end())
        {
            LogError("Invalid device_weight_data_type for model %s", spec.sid.c_str());
            return false;
        }

        int element_size = TensorCommon::ElementSize(iter->second);
        spec.device_weight_data_type = element_size >= 2
            ? ElementType::F16 : iter->second;
    }

    str.clear();
    cfg_data.GetItem(section, "device_kv_cache_data_type", str, false);
    if (!str.empty())
    {
        auto iter = element_type_map.find(str);
        if (iter == element_type_map.end())
        {
            LogError("Invalid device_kv_cache_data_type for model %s", spec.sid.c_str());
            return false;
        }

        int element_size = TensorCommon::ElementSize(iter->second);
        spec.device_kv_cache_data_type = element_size >= 2
            ? ElementType::F16 : ElementType::Q8_B32T2;
    }

    cfg_data.GetItem(section, "tensor_quant_threshold", spec.tensor_quant_threshold, false);

    cfg_data.GetItem(section, "delta_tensor_ratio", spec.delta_tensor_ratio, false);
    spec.delta_tensor_ratio = min(0.05f, spec.delta_tensor_ratio);

    cfg_data.GetItem(section, "be_host_embeddings", spec.be_host_embeddings, false);
    cfg_data.GetItem(section, "host_kv_cache_percent", spec.host_kv_cache_percent, false);

    str.clear();
    (void)multi_gpu_strategy_map;
    //cfg_data.GetItem(section, "multi_gpu_strategy", str, false);
    //auto iter_multi_gpu = multi_gpu_strategy_map.find(str);
    //if (iter_multi_gpu != multi_gpu_strategy_map.end()) {
    //    spec.multi_gpu_strategy = iter_multi_gpu->second;
    //}

    ret = LoadDeviceGroups(spec.device_groups, cfg_data, section, "devices");
    Macro_RetFalseIf(!ret);

    cfg_data.GetItem(section, "max_context_len", spec.max_context_len, false);

    if (ret)
    {
        bool is_abs = Path::IsAbsolute(spec.spec_file);
        string spec_file_path = is_abs ? spec.spec_file : spec.dir + spec.spec_file;

        ret = ModelReader::LoadModelSpecJson(spec, spec_file_path, jparser);
    }
    return ret;
}

bool InferenceEngine::LoadDeviceGroups(vector<vector<int>> &device_groups,
    const ConfigData &cfg_data, const string &section, const string &key)
{
    device_groups.clear();

    string str;
    cfg_data.GetItem(section, key, str, false);
    vector<string> tokens, sub_tokens;
    String::Split(str, tokens, ",;");
    for (string &token_str : tokens)
    {
        String::Trim(token_str);
        if (token_str.empty()) {
            continue;
        }

        vector<int> sub_list;
        String::Split(token_str, sub_tokens, "&|");
        for (string &sub_token_str : sub_tokens)
        {
            String::Trim(sub_token_str);
            if (!sub_token_str.empty()) {
                sub_list.push_back(atoi(sub_token_str.c_str()));
            }
        }
        device_groups.push_back(sub_list);
    }

    int device_group_num = (int)device_groups.size();
    if (device_group_num > 1)
    {
        int group_size = (int)device_groups[0].size();
        for (int group_idx = 1; group_idx < device_group_num; group_idx++)
        {
            int num = (int)device_groups[group_idx].size();
            if (num != group_size)
            {
                LogError("All device groups should have the same size: %d vs. %d",
                    group_size, num);
                return false;
            }
        }
    }

    return true;
}

bool InferenceEngine::LoadPromptTemplates(map<string, string> &prompt_templates,
    const ConfigData &cfg_data, const string &section)
{
    int template_num = 0;
    vector<string> str_list;
    bool ret = cfg_data.GetItem(section, "prompt_template_count", template_num, true);
    ret = ret && cfg_data.GetItems(section, "prompt_template", 1, template_num, str_list, true);
    Macro_RetFalseIf(!ret);

    for (const string &template_name : str_list)
    {
        string template_section = "prompt_template." + template_name;
        string template_str;
        ret = LoadPromptTemplate(template_str, cfg_data, template_section);
        Macro_RetFalseIf(!ret);

        bool is_valid = template_str.find("{query}") != string::npos
            || template_str.find("{_query}") != string::npos;
        if (!is_valid) {
            LogError("Invalid prompt template: %s", template_name.c_str());
            return false;
        }

        prompt_templates[template_name] = template_str;
    }

    return ret;
}

bool InferenceEngine::LoadPromptTemplate(string &template_str,
    const ConfigData &cfg_data, const string &section)
{
    template_str.clear();

    int line_num = 0;
    vector<string> lines;
    bool ret = cfg_data.GetItem(section, "line_count", line_num, true);
    ret = ret && cfg_data.GetItems(section, "line", 1, line_num, lines, true);
    Macro_RetFalseIf(!ret);

    for (int idx = 0; idx < (int)lines.size(); idx++)
    {
        const string &line_str = lines[idx];
        if (idx > 0) {
            template_str += '\n';
        }
        template_str += line_str;
    }

    String::ReplaceAll(template_str, "{\\n}", "\n");
    return ret;
}

SamplingStrategy* InferenceEngine::GetSamplingStrategy(SamplingStrategyId id)
{
    return decoding_strategies_.Get(id);
}

SamplingStrategyId InferenceEngine::GetSamplingStrategyId(const string &str) const
{
    string strategy_str = str.empty() ? model_.spec.decoding_strategy : str;
    return decoding_strategies_.GetId(strategy_str);
}

bool InferenceEngine::BuildCpuLayers(const ModelSpec &model_spec,
    int encoder_layer_count, int decoder_layer_count)
{
    bool ret = true;
    auto &model = model_;

    ret = network_builder_.BuildGgmlNetwork(model, model_spec, context_.ggml_ctx,
        encoder_layer_count, decoder_layer_count);
    return ret;
}

#if defined(USE_CUDA)

bool InferenceEngine::BuildGpuLayers(const ModelSpec &model_spec,
    const ModelPartition &model_partition)
{
    bool ret = true;
    int builder_count = 4;
    auto &model = model_;

    if (model_spec.model_file_format == ModelFileFormat::LLAMA2_C)
    {
        ret = network_builder_.InitDeviceNetStructure(model.std_network, model_spec);
        Macro_RetxFalseIf(!ret, LogError("Failed to initialize the device net structure"));
    }

    ret = network_builder_.BuildDeviceNetwork(model, model_spec,
        model_partition, builder_count);
    Macro_RetFalseIf(!ret);

    NetworkStat stat;
    model.std_network.device_net.CalculateStat(stat);

    LogKeyInfo("VRAM usage statistics of model %s:", model.spec.sid.c_str());
    LogKeyInfo("    Overall: %.3f GB", stat.all_memory_usage);
    if (stat.encoder_layer_num > 0)
    {
        LogKeyInfo("    Encoder attention&ffn layers: %.3f GB (= %d * %.3f)",
            stat.encoder_layer_num * stat.encoder_layer_size[0],
            stat.encoder_layer_num, stat.encoder_layer_size[0]);
    }
    if (stat.decoder_layer_num > 0)
    {
        LogKeyInfo("    Decoder attention&ffn layers: %.3f GB (= %d * %.3f)",
            stat.decoder_layer_num * stat.decoder_layer_size[0],
            stat.decoder_layer_num, stat.decoder_layer_size[0]);
    }
    LogKeyInfo("    Other layers (embedding, pre, post): %.3f GB (= %.3f + %.3f + %.3f)",
        stat.embedding_size + stat.pre_layer_size + stat.post_layer_size,
        stat.embedding_size, stat.pre_layer_size, stat.post_layer_size);

    ElementTypeNameMap type_name_map;
    TensorCommon::InitElementTypeMap(type_name_map);

    LogKeyInfo("Tensor element type statistics (ignoring duplicate layers):");
    auto iter = stat.tensor_num_by_element_type.begin();
    for (; iter != stat.tensor_num_by_element_type.end(); iter++)
    {
        auto name_iter = type_name_map.find(iter->first);
        string type_name = name_iter == type_name_map.end() ? "" : name_iter->second;
        LogKeyInfo("    %s (id:%d): %d", type_name.c_str(), (int)iter->first, iter->second);
    }

    CudaUtil::SetDevice(default_device_id_);
    return ret;
}

bool InferenceEngine::CreateGpuWorkers(const ModelSpec &model_spec,
    const ModelPartition &model_partition)
{
    bool ret = true;
    gpu_worker_groups_.Clear(true);

    int device_group_num = (int)model_.spec.device_groups.size();
    int la_num1 = (int)model_partition.encoder_assignments.size();
    int la_num2 = (int)model_partition.decoder_assignments.size();
    int la_num = max(la_num1, la_num2);
    if (la_num != device_group_num)
    {
        LogError("Invalid layer assisngments");
        return false;
    }

    uint64_t aux_mem_usage = 0;
    bool is_by_layer = model_spec.multi_gpu_strategy == MultiGpuStrategy::BY_LAYER;
    LayerRange encoder_layer_range, decoder_layer_range;
    encoder_layer_range.layer_num = model_spec.hyper_params.encoder_layers;
    decoder_layer_range.layer_num = model_spec.hyper_params.decoder_layers;
    for (int group_idx = 0; group_idx < device_group_num; group_idx++)
    {
        GpuWorkerGroup *worker_group = new GpuWorkerGroup;
        gpu_worker_groups_.push_back(worker_group);

        const vector<int> &device_group = model_.spec.device_groups[group_idx];
        int group_size = (int)device_group.size();
        worker_group->global_data.Init(group_size);

        for (int worker_id = 0; worker_id < group_size; worker_id++)
        {
            int device_id = device_group[worker_id];
            CudaUtil::SetDevice(device_id); //!!!

            const LayerAssignment *encoder_la = group_idx >= la_num1 ? nullptr
                : &model_partition.encoder_assignments[group_idx];
            const LayerAssignment *decoder_la = group_idx >= la_num2 ? nullptr
                : &model_partition.decoder_assignments[group_idx];
            //int device_id = encoder_la != nullptr ? *encoder_la->devices->rbegin()
            //    : *decoder_la->devices->rbegin();

            GpuInferenceWorker *worker_ptr = new GpuInferenceWorker;
            worker_group->workers.push_back(worker_ptr);

            if (encoder_la != nullptr)
            {
                encoder_layer_range.start = encoder_la->start_layer;
                encoder_layer_range.end = encoder_la->end_layer;
            }
            if (decoder_la != nullptr)
            {
                decoder_layer_range.start = decoder_la->start_layer;
                decoder_layer_range.end = decoder_la->end_layer;
            }

            worker_ptr->Init(worker_id, group_size, group_idx, device_group_num,
                config_, config_ex_, model_, device_id,
                encoder_layer_range, decoder_layer_range,
                is_by_layer);
            aux_mem_usage += worker_ptr->AuxMemorySize();
        }
    } //for each group

    LogKeyInfo("Auxiliary memory usage of GPU inference workers: %.2f GB",
        aux_mem_usage / 1024.0f / 1024 /  1024);

    return ret;
}

bool InferenceEngine::SampleTokens(vector<IdWeight<float>> &next_tokens,
    const DeviceTensor &logits, int row, int query_id,
    const vector<int> &prefix_tokens, const vector<int> &cur_tokens,
    SamplingStrategyId strategy_id)
{
    next_tokens.clear();
    auto *strategy_ptr = GetSamplingStrategy(strategy_id);
    if (logits.ne[1] < 1 || strategy_ptr == nullptr) {
        return false;
    }

    TaskMonitor tm;
    SamplingInput input;
    input.query_id = query_id;
    input.prefix = &prefix_tokens;
    input.cur_tokens = &cur_tokens;
    //int last_row = logits.ne[1] - 1;

    if (logits.data_type == ElementType::F16) {
        logits.CopyRowToHost(input.candidates_fp16, row, 0);
    }
    else {
        logits.CopyRowToHost(input.candidates, row, 0);
    }
    //tm.ShowElapsedTime(L"copy-logits-to-host");

    //tm.Start();
    //vector<inferflow_fp16> fp16_candidates;
    //logits.CopyRowToHost(fp16_candidates, row, 0);
    //tm.ShowElapsedTime(L"copy-logits-to-host (fp16)");

    int thread_count = 1;
    SamplingOutput output;
    strategy_ptr->ChooseTokens(output, input, model_.vocabulary, strategy_id, thread_count);
    if (!output.selected.empty()) {
        next_tokens.push_back(output.selected[0]);
    }

    //const half *row_data = (const half*)logits.RowData(last_row);
    //if (row_data == nullptr) {
    //    return false;
    //}

    //IdWeight<float> max_token(UINT32_MAX, 0);
    //int column_count = logits.Columns();
    //LogKeyInfo("last row: %d, column_count: %d", last_row, column_count);

    //for (int col = 0; col < column_count; col++)
    //{
    //    float score = (float)row_data[col];
    //    if (max_token.id == UINT32_MAX || max_token.weight < score) {
    //        max_token.Set(col, score);
    //    }
    //}

    return true;
}

bool InferenceEngine::EncoderSampleTokens(vector<IdWeight<float>> &output_tokens,
    const DeviceTensor &logits, int row, int query_id, SamplingStrategyId strategy_id)
{
    output_tokens.clear();
    auto *strategy_ptr = GetSamplingStrategy(strategy_id);
    if (logits.ne[1] < 1 || strategy_ptr == nullptr) {
        return false;
    }

    SamplingInput input;
    input.query_id = query_id;
    input.prefix = nullptr;
    input.cur_tokens = nullptr;
    //int last_row = logits.ne[1] - 1;

    if (logits.data_type == ElementType::F16) {
        logits.CopyRowToHost(input.candidates_fp16, row, 0);
    }
    else {
        logits.CopyRowToHost(input.candidates, row, 0);
    }

    int thread_count = 1;
    SamplingOutput output;
    strategy_ptr->ChooseTokens(output, input, model_.vocabulary, strategy_id, thread_count);
    if (!output.token_pool.empty()) {
        output_tokens = output.token_pool;
    }
    else if (!output.selected.empty()) {
        output_tokens = output.selected;
    }

    return true;
}

#endif //USE_CUDA

bool InferenceEngine::SampleTokensCpu(vector<IdWeight<float>> &tokens, ggml_tensor *logits,
    int row, int query_id, const vector<int> &prefix_tokens, const vector<int> &cur_tokens,
    SamplingStrategyId strategy_id)
{
    tokens.clear();
    auto *strategy_ptr = GetSamplingStrategy(strategy_id);
    if (logits->ne[1] < 1 || strategy_ptr == nullptr) {
        return false;
    }

    SamplingInput input;
    input.query_id = query_id;
    input.prefix = &prefix_tokens;
    input.cur_tokens = &cur_tokens;

    input.candidates.resize(logits->ne[0]);
    memcpy(input.candidates.data(), (float *) ggml_get_data(logits) + (logits->ne[0] * row), logits->ne[0] * sizeof(float));

    SamplingOutput output;
    strategy_ptr->ChooseTokens(output, input, model_.vocabulary, strategy_id);
    if (!output.selected.empty()) {
        tokens.push_back(output.selected[0]);
    }

    return true;
}

void InferenceEngine::PrintPerfStat(ostream &strm, const InferencePerfStat &perf_stat) const
{
    auto iter = perf_stat.time_map.begin();
    for (; iter != perf_stat.time_map.end(); iter++)
    {
        if (iter->first == 0) {
            strm << "0 (E2E)\t" << iter->second << "\n";
        }
        else {
            strm << iter->first << "\t" << iter->second << "\n";
        }
    }
}

//static
int InferenceEngine::GetDefaultDevice(const ModelSpec &spec)
{
    if (!spec.device_groups.empty())
    {
        const auto &last_group = *spec.device_groups.rbegin();
        if (!last_group.empty()) {
            return *last_group.rbegin();
        }
    }
    return 0;
}

void InferenceEngine::InitKeyMap()
{
    key_map_["system_prompt"] = KeyId::SystemPrompt;
    key_map_["question"] = KeyId::QueryText;
    key_map_["query"] = KeyId::QueryText;
    key_map_["_question"] = KeyId::QueryTextWithPreSpace;
    key_map_["_query"] = KeyId::QueryTextWithPreSpace;
    key_map_["res_prefix"] = KeyId::ResPrefix;
    key_map_["answer"] = KeyId::ResPrefix;
    key_map_["bos"] = KeyId::Bos;
    key_map_["eos"] = KeyId::Eos;
    key_map_["user_token"] = KeyId::UserToken;
    key_map_["assistant_token"] = KeyId::AssistantToken;
}

void InferenceEngine::LockQueryStateTable() const
{
#   ifdef __GNUC__
#       pragma GCC diagnostic push
#       pragma GCC diagnostic ignored "-Wcast-qual"
#   endif

    ((mutex&)query_state_lock_).lock(); //lock

#   ifdef __GNUC__
#       pragma GCC diagnostic pop
#   endif
}

void InferenceEngine::UnLockQueryStateTable() const
{
#   ifdef __GNUC__
#       pragma GCC diagnostic push
#       pragma GCC diagnostic ignored "-Wcast-qual"
#   endif

    ((mutex&)query_state_lock_).unlock(); //unlock

#   ifdef __GNUC__
#       pragma GCC diagnostic pop
#   endif
}

TRANSFORMER_END
INFER_FLOW_END
