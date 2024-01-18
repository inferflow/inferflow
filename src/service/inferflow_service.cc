#include "inferflow_service.h"
#include <sstream>
#include <chrono>

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

////////////////////////////////////////////////////////////////////////////////
// class InferFlowServiceCore

string InferFlowServiceCore::Version() const
{
    return engine_.Version();
}

bool InferFlowServiceCore::LoadConfig(const string &config_path)
{
    auto &engine_config = config_.engine_config;
    string section = "transformer_engine";
    bool ret = transformer::InferenceEngine::LoadConfig(engine_config, config_path, section);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the inference config"));

    ConfigData cfg_data;
    ret = cfg_data.Load(config_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the config-data"));

    section = "main";
    ret = ret && cfg_data.GetItem(section, "http_port", config_.http_port, true);
    ret = ret && cfg_data.GetItem(section, "worker_count", config_.worker_count, true);
    cfg_data.GetItem(section, "is_study_mode", config_.is_study_mode, false);
    Macro_RetFalseIf(!ret);

    return ret;
}

bool InferFlowServiceCore::Init(const string &config_path)
{
    bool ret = LoadConfig(config_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the configuration"));

    //LogKeyInfo("Devices: %d", (int)config_.engine_config.devices.size());

    json_parser_.Init();

    TaskMonitor tm;
    ret = engine_.Init(config_.engine_config);
    Macro_RetxFalseIf(!ret, LogError("Failed to initialize the inference engine"));

    if (config_.is_study_mode) {
        tm.ShowElapsedTime(L"### Time of initializing the engine");
    }

    fn_map_[L"process_query"] = FunctionId::ProcessQuery;
    fn_map_[L"get_stat"] = FunctionId::GetStat;

    return ret;
}

//virtual
void InferFlowServiceCore::Run()
{
    bool ret = true;
    int max_output_len = 500;

    //int default_device_id = engine_.default_device_id();
    //cudaSetDevice(default_device_id);

    while (ret)
    {
        ret = Infer(max_output_len);
    }
}

bool InferFlowServiceCore::Infer(int max_output_len)
{
    const auto &vocab = engine_.vocabulary();
    //const TextTokenizer &tokenizer = *engine_.tokenizer();

    InferenceResult result;
    bool ret = engine_.Infer(result);
    if (result.items.empty())
    {
        Thread::SleepMilli(1);
        return ret;
    }

    map<int, QueryNextToken> query_map;
    for (int item_idx = 0; item_idx < (int)result.items.size(); item_idx++)
    {
        const QueryInferenceResult &query_res = *result.items[item_idx];
        int output_len = query_res.prefix_len + 1;

        int next_token_id = query_res.next_tokens[0].id;
        bool is_eos = next_token_id == vocab.eos();

        QueryNextToken next_token;
        next_token.id = next_token_id;
        next_token.is_end = is_eos || output_len >= max_output_len;
        query_map[query_res.query_id] = next_token;
        //LogKeyInfo("query_res.query_id: %d, next_token_id: %d",
        //    query_res.query_id, next_token_id);

        string token_str = engine_.OutputTokenIdToStr(next_token_id);
        //LogKeyInfo("token_str: %s", token_str.c_str());

        if (next_token.is_end)
        {
            LogKeyInfo("query_id: %d, output_len: %d, is_end: true",
                query_res.query_id, output_len);
        }

        result_lock_.lock(); //lock
        auto iter_find = query_to_result_.find(query_res.query_id);
        if (iter_find == query_to_result_.end())
        {
            QueryResult qr;
            qr.text = token_str;
            qr.is_end = next_token.is_end;
            query_to_result_[query_res.query_id] = qr;
        }
        else
        {
            iter_find->second.text += token_str;
            iter_find->second.is_end = next_token.is_end;
        }
        result_lock_.unlock(); //unlock
    }

    engine_.CommitInferenceResult(query_map);
    return ret;
}

Socket::RetCode InferFlowServiceCore::HandleRequest(
    BaseHttpServer::HttpResponseWriter &writer,
    const InferFlowRequest &request)
{
    auto ret_code = HandleRequest_Inner(&writer, nullptr, request);
    return ret_code;
}

Socket::RetCode InferFlowServiceCore::HandleRequest_Inner(
    BaseHttpServer::HttpResponseWriter *writer,
    InferFlowResponseChunk *chunk_ptr,
    const InferFlowRequest &request)
{
    Socket::RetCode ret_code = Socket::RetCode::Success;
    bool ret = true;
    bool is_streaming = writer != nullptr;

    InferFlowResponseChunk chunk; //current chunk
    chunk_ptr = is_streaming || chunk_ptr == nullptr ? &chunk : chunk_ptr;

    bool has_error = false;
    int query_id = 0;
    if (request.query.text.empty() && request.query.response_prefix.empty())
    {
        chunk_ptr->ret_code = L"error.empty_request";
        has_error = true;
    }
    else
    {
        const LlmQuery &query = request.query;

        EncoderInput encoder_input;
        DecoderPrefix decoder_prefix;
        engine_.BuildEncoderInput(encoder_input, query);
        engine_.BuildDecoderInput(decoder_prefix, query);

        SamplingStrategy::QueryOptions query_options;
        string decoding_alg = StringUtil::ToUtf8(request.decoding_alg);
        query_options.strategy_id = engine_.GetSamplingStrategyId(decoding_alg);
        query_options.random_seed = request.random_seed;
        query_options.temperature = request.temperature;

        LogKeyInfo("decoding_alg: %s, strategy_id: %d, temperature: %.2f",
            decoding_alg.c_str(), query_options.strategy_id, query_options.temperature);
        LogKeyInfo("Encoder input text: %s", encoder_input.core_text.c_str());

        wstringstream ss;
        for (const auto &section : decoder_prefix.sections)
        {
            if (section.is_text) {
                ss << StringUtil::Utf8ToWideStr(section.text);
            }
            else {
                ss << L"{" << StringUtil::Utf8ToWideStr(section.text) << L"}";
            }
        }
        ss << StringUtil::Utf8ToWideStr(decoder_prefix.res_prefix);
        LogKeyInfo(L"Decoder input text: %ls", ss.str().c_str());

        query_id = engine_.AddQuery(encoder_input, decoder_prefix, query_options, TokenizationAlg::FMM);
        if (query_id <= 0)
        {
            chunk_ptr->ret_code = L"error.busy";
            has_error = true;
        }
    }

    if (has_error)
    {
        if (is_streaming)
        {
            wstring response_str;
            chunk_ptr->ToJson(response_str);
            string utf8_str = StringUtil::ToUtf8(response_str);

            ret_code = writer->WriteChunk(utf8_str);
        }
        return ret_code;
    }

    string new_text;
    bool is_end = false;
    while (ret && !is_end)
    {
        Thread::SleepMilli(10);

        result_lock_.lock(); //lock
        auto iter = query_to_result_.find(query_id);
        if (iter != query_to_result_.end())
        {
            auto &res_item = iter->second;
            new_text += res_item.text;
            res_item.text.clear();
            is_end = is_end || res_item.is_end;

            if (is_end) {
                query_to_result_.erase(iter);
            }
        }
        result_lock_.unlock(); //unlock

        //if (!new_text.empty()) {
        //    LogKeyInfo("new_text: %s", new_text.c_str());
        //}

        int end_pos = GetUtf8EndPos(new_text);
        if (end_pos > 0)
        {
            chunk_ptr->text += StringUtil::Utf8ToWideStr(new_text.c_str(), end_pos);
            chunk_ptr->text_utf8_len += end_pos;
            new_text = new_text.substr(end_pos);
        }

        if (is_end || (chunk_ptr->text_utf8_len >= 16 && is_streaming))
        {
            chunk_ptr->ret_code = L"succ";
            if (is_streaming)
            {
                wstring response_str;
                chunk_ptr->is_end = is_end;
                chunk_ptr->ToJson(response_str);
                chunk_ptr->Clear(); //!!!

                string utf8_str = StringUtil::ToUtf8(response_str);

                ret_code = writer->WriteChunk(utf8_str);
                if (ret_code != Socket::RetCode::Success)
                {
                    engine_.RemoveQuery(query_id);
                    return ret_code;
                }
            }
        }
    }

    if (is_streaming)
    {
        string empty_str;
        ret_code = writer->WriteChunk(empty_str);
    }

    return ret_code;
}

bool InferFlowServiceCore::HandleRequest(string &response,
    const InferFlowRequest &request, FunctionId fn)
{
    bool ret = true;
    switch (fn)
    {
    case FunctionId::GetStat:
        ret = HandleRequest_GetStat(response, request);
        break;
    case FunctionId::ProcessQuery:
    default:
        ret = HandleRequest_ProcessQuery(response, request);
        break;
    }

    return ret;
}

InferFlowServiceCore::FunctionId InferFlowServiceCore::GetFunctionId(const string &url,
    const InferFlowRequest &request) const
{
    wstring fn_name = request.header.fn;
    if (!fn_name.empty())
    {
        auto pos = url.rfind('/');
        if (pos != string::npos) {
            fn_name = StringUtil::Utf8ToWideStr(url.substr(pos + 1));
        }
    }

    auto fn_iter = fn_map_.find(fn_name);
    FunctionId fn_id = fn_iter != fn_map_.end() ? fn_iter->second : FunctionId::ProcessQuery;
    return fn_id;
}

bool InferFlowServiceCore::HandleRequest_ProcessQuery(string &response,
    const InferFlowRequest &request)
{
    bool ret = true;
    response.clear();

    auto start_tm = chrono::steady_clock::now();
    InferFlowResponseChunk chunk;
    HandleRequest_Inner(nullptr, &chunk, request);
    auto tm = chrono::steady_clock::now();

    int time_cost = (int)chrono::duration_cast<chrono::milliseconds>(tm - start_tm).count();
    chunk.time_cost = time_cost / 1000.0f;

    if (chunk.ret_code.empty()) {
        chunk.ret_code = ret ? L"succ" : L"error.processing";
    }

    wstring json_str;
    chunk.ToJson(json_str);
    StringUtil::ToUtf8(response, json_str);

    /*
    if (nlu_response.header.ret_code.empty())
    {
        nlu_response.header.ret_code = ret ? L"succ" : L"error.processing";
    }

    auto cur_tm = chrono::steady_clock::now();
    int elapsed_time = (int)chrono::duration_cast<chrono::microseconds>(cur_tm - start_tm).count();
    nlu_response.header.time_cost = elapsed_time / 1000.0f;

    wstring jstr_response;
    nlu_response.ToJson(jstr_response);
    StringUtil::ToUtf8(response, jstr_response);

    //log
    LogStatusInfo(L"query_log.v2\tparse_text\t%ls\t%ls\t%.2f\t%.2f",
        encoded_request.c_str(),
        nlu_response.header.ret_code.c_str(),
        nlu_response.header.core_time_cost,
        nlu_response.header.time_cost);*/

    return ret;
}

bool InferFlowServiceCore::HandleRequest_GetStat(string &response,
    const InferFlowRequest &request)
{
    bool ret = true;
    response.clear();

    (void)request;
    return ret;
}

bool InferFlowServiceCore::ParseRequest(InferFlowRequest &request,
    const wstring &request_str)
{
    json_parsr_lock_.lock(); //lock the json parser
    bool ret = request.FromJson(request_str, json_parser_);
    json_parsr_lock_.unlock(); //unlock the json parser
    return ret;
}

//static
int InferFlowServiceCore::GetUtf8EndPos(const string &text)
{
    int len = (int)text.length();
    if (len <= 0) {
        return 0;
    }

    uint8_t last_ch = (uint8_t)text[len - 1];
    if ((last_ch & 0x80) == 0) {
        return len;
    }

    for (int idx = len - 1; idx >= 0; idx--)
    {
        uint8_t ch = (uint8_t)text[idx];
        if ((ch & 0x40) == 0) {
            continue;
        }

        int byte_num = (ch & 0x20) == 0 ? 2 : ((ch & 0x10) == 0 ? 3 : 4);
        return idx + byte_num == len ? len : idx;
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// class InferFlowService

bool InferFlowService::Init(const string &config_path)
{
    bool ret = core_.Init(config_path);
    if (!ret) {
        return false;
    }

    const auto &cfg = core_.config();
    if (cfg.http_port <= 0)
    {
        LogError("Invalid HTTP port");
        return false;
    }

    bool is_study_mode = false;
    ret = BaseHttpServer::Init(cfg.http_port, false, cfg.worker_count, is_study_mode);
    return ret;
}

bool InferFlowService::Start()
{
    const auto &core_cfg = core_.config();
    if (core_cfg.http_port > 0)
    {
        //start a thread
        core_.Create();

        //start the HTTP server
        return BaseHttpServer::Start();
    }

    return false;
}

string InferFlowService::Version() const
{
    return core_.Version();
}

bool InferFlowService::HandleRequest(HttpResponseWriter &writer,
    const HttpTask &task)
{
    bool ret = true;
    bool is_unknown_type = task.request.type == HttpRequest::Method::Unknown;

    HttpResponse response;
    response.status_code = 200;
    bool is_valid_query = true;
    const auto &http_request = task.request;
    if (is_unknown_type)
    {
        response.status_code = 400; //bad request
        is_valid_query = false;
    }
    else if (http_request.type != HttpRequest::Method::Get
        && http_request.type != HttpRequest::Method::Post
        && http_request.type != HttpRequest::Method::Put)
    {
        response.status_code = 501;
        is_valid_query = false;
    }

    uint32_t request_hash = 0;
    InferFlowRequest request;
    InferFlowResponseChunk res_chunk;
    if ((int)task.request.body.size() > MAX_REQUEST_LEN)
    {
        res_chunk.ret_code = L"error.too_long_request";
        is_valid_query = false;
    }
    else if (is_valid_query)
    {
        request_hash = Hash::HashCodeUInt32(task.request.body);
        wstring request_body = StringUtil::Utf8ToWideStr(task.request.body);

        //Parse the request (assuming JSON format)
        ret = core_.ParseRequest(request, request_body);
        if (!ret)
        {
            res_chunk.ret_code = L"error.invalid_request_format";
            is_valid_query = false;
        }

        wstring encoded_request = JsonBuilder::EncodeString(request_body);
        LogStatusInfo(L"query_start\t%u\t%ls", request_hash, encoded_request.c_str());
    }

    auto fn = core_.GetFunctionId(http_request.url, request);
    bool is_std_fn = fn == InferFlowServiceCore::FunctionId::ProcessQuery;

    bool be_keep_alive = !is_unknown_type && !task.hdr.is_close;
    string content_type = task.request.type == HttpRequest::Method::Get
        ? "text/html" : "application/json";

    Socket::RetCode ret_code = Socket::RetCode::Success;
    if (request.is_streaming_mode && is_std_fn)
    {
        int body_len = -1;
        ret_code = writer.WriteHeader(response.header_lines, content_type,
            response.status_code, be_keep_alive, body_len);
        if (is_valid_query && ret_code == Socket::RetCode::Success)
        {
            ret_code = core_.HandleRequest(writer, request);
        }
    }
    else
    {
        string response_body;
        if (is_valid_query) {
            ret = core_.HandleRequest(response_body, request, fn);
        }

        int body_len = (int)response_body.size();
        ret_code = writer.WriteHeader(response.header_lines, content_type,
            response.status_code, be_keep_alive, body_len);
        if (is_valid_query && ret_code == Socket::RetCode::Success
            && !response_body.empty())
        {
            ret_code = writer.WriteString(response_body);
        }
    }

    /*if (is_valid_query)
    {
        wstring encoded_fn = JsonBuilder::EncodeString(request.header.fn);
        wstring encoded_query_text = JsonBuilder::EncodeString(request.text);
        wstring encoded_prefix = JsonBuilder::EncodeString(request.prefix);
        wstring encoded_decoding_alg = JsonBuilder::EncodeString(request.decoding_alg);
        LogStatusInfo(L"query_log\t%u\t%ls\t%ls\t%ls\t%ls\t%d\t%d\t%.2f",
            request_hash, encoded_fn.c_str(), encoded_query_text.c_str(),
            encoded_prefix.c_str(), encoded_decoding_alg.c_str(),
            request.is_streaming_mode ? 1 : 0, response_len, time_cost);
    }*/

    return ret_code == Socket::RetCode::Success;
}

TRANSFORMER_END
INFER_FLOW_END
