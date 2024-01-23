#include "transformer/inference_engine.h"
#include <sstream>
#include "sslib/path.h"
#include "sslib/log.h"
#include "sslib/app_environment.h"

using namespace std;
using namespace sslib;
using namespace inferflow;
using namespace inferflow::transformer;

void PrintInputTokens(const vector<int> &tokens, const InferenceEngine &engine,
    ostream &writer, bool is_console)
{
    for (int idx = 0; idx < (int)tokens.size(); idx++)
    {
        int token_id = tokens[idx];
        const string utf8_token_str = engine.TokenIdToStr(token_id);
        string token_str = utf8_token_str;
        if (is_console)
        {
            wstring wide_token_str = StringUtil::Utf8ToWideStr(utf8_token_str);
            token_str = StringUtil::ToConsoleEncoding(wide_token_str);
        }
        writer << (idx > 0 ? ", " : "") << token_str << " (" << token_id << ")";
    }

    writer << "\n";
}

void PrintTokenStr_StudyMode(int token_id, const string &token_str, float token_score)
{
    (void)token_score;
    char hex_buf[16];
    string hex_token_str;
    if (token_str.size() <= 3)
    {
        for (char ch : token_str)
        {
            sprintf(hex_buf, " \\u%x", (uint8_t)ch);
            hex_token_str += hex_buf;
        }
    }
    wstring token_wstr;
    if (StringUtil::Utf8ToWideStr(token_wstr, token_str))
    {
        wstring hex_token_wstr = StringUtil::Utf8ToWideStr(hex_token_str);
        LogKeyInfo(L"next token: %ls (%d, %ls)", token_wstr.c_str(),
            token_id, hex_token_wstr.c_str());
    }
    else
    {
        LogKeyInfo("next token: %s (%d, %s)", token_str.c_str(),
            token_id, hex_token_str.c_str());
    }
}

void PrintPerfStat(ostream &perf_stat_writer, const InferenceResult &result,
    const InferenceEngine &engine)
{
    if (!result.perf_stat_pre.time_map.empty())
    {
        cout << "perf_stat_pre:\n";
        engine.PrintPerfStat(cout, result.perf_stat_pre);
        perf_stat_writer << "perf_stat_pre:\n";
        engine.PrintPerfStat(perf_stat_writer, result.perf_stat_pre);
    }

    cout << "perf_stat:\n";
    engine.PrintPerfStat(cout, result.perf_stat);
    perf_stat_writer << "perf_stat:\n";
    engine.PrintPerfStat(perf_stat_writer, result.perf_stat);
}

void PrintTokenStr(int token_id, const string &token_str, float token_score,
    bool is_bos, bool is_eos, int max_query_count, wstring &res_prefix_for_display)
{
    (void)token_id;
    if (is_eos)
    {
        cout << "<eos:" << token_score << ">";
        if (max_query_count > 1) {
            cout << "\nWaiting for other queries to complete...";
        }
        //is_continue = false;
    }
    else if (!is_bos)
    {
        if (!res_prefix_for_display.empty())
        {
            cout << StringUtil::ToConsoleEncoding(res_prefix_for_display);
            res_prefix_for_display.clear();
        }

        wstring wide_token_str = StringUtil::Utf8ToWideStr(token_str);
        cout << StringUtil::ToConsoleEncoding(wide_token_str);
        cout.flush();
    }
}

void PrintEncoderOutput(const QueryInferenceResult &query_res, const InferenceEngine &engine)
{
    cout << "Output:\n";
    int token_num = (int)query_res.next_tokens.size();
    for (int idx = 0; idx < token_num; idx++)
    {
        const auto &token = query_res.next_tokens[idx];
        string token_str = engine.OutputTokenIdToStr(token.id);
        wstring wide_token_str = StringUtil::Utf8ToWideStr(token_str);
        cout << (idx + 1) << "\t";
        cout << StringUtil::ToConsoleEncoding(wide_token_str);
        cout << "\t" << token.id << "\t" << token.weight << "\n";
    }
    cout.flush();
}

struct InferenceTestConfig
{
    string engine_config_file;
    string query_list_section;
    int query_random_seed = 0;
    float temperature = 1.0f;
    bool is_long_context_test = false;

    vector<LlmQuery> query_list;
};

bool LoadQueryList(vector<LlmQuery> &query_list, const ConfigData &config_data,
    const string &section)
{
    query_list.clear();

    JsonParser json_parser;
    json_parser.Init();

    int query_count = 0;
    vector<wstring> query_str_list;
    bool ret = config_data.GetItem(section, "query_count", query_count, true);
    ret = ret && config_data.GetItems(query_str_list, section, "query", 1, query_count, true);

    for (const auto &query_str : query_str_list)
    {
        LlmQuery query;
        bool is_valid_json = InferenceEngine::ParseQueryJson(query, query_str, json_parser);
        if (!is_valid_json)
        {
            query.Clear();
            query.text = query_str;
            WString::ReplaceAll(query.text, L"{\\r}", L"\r");
            WString::ReplaceAll(query.text, L"{\\n}", L"\n");
            WString::ReplaceAll(query.text, L"{\\t}", L"\t");
        }

        query_list.push_back(query);
    }

    return ret;
}

bool LoadInferenceTestConfig(InferenceTestConfig &config, const string &file_path)
{
    ConfigData config_data;
    bool ret = config_data.Load(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the configuration data"));

    ret = config_data.GetItem("main", "inference_engine_config",
        config.engine_config_file, ret);
    ret = ret && config_data.GetItem("main", "query_list",
        config.query_list_section, true);
    config_data.GetItem("main", "query_random_seed", config.query_random_seed, false);
    config_data.GetItem("main", "temperature", config.temperature, false);

    config_data.GetItem("main", "long_context_test", config.is_long_context_test, false);
    Macro_RetFalseIf(!ret);

    ret = LoadQueryList(config.query_list, config_data, config.query_list_section);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the query list"));

    return ret;
}

bool Run(const string &config_path)
{
    InferenceTestConfig test_config;
    bool ret = LoadInferenceTestConfig(test_config, config_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the configuration"));

    InferenceConfig engine_config;
    ret = InferenceEngine::LoadConfig(engine_config, test_config.engine_config_file,
        "transformer_engine");
    Macro_RetxFalseIf(!ret, LogError("Failed to load the inference configuration"));

    engine_config.data_dir = AppEnv::DataRootDir();

    InferenceEngine engine;
    ret = engine.Init(engine_config);
    if (!ret) {
        LogError("Failed to initialize the inference engine");
        return false;
    }

    const auto &vocab = engine.vocabulary();

    //TextTokenizer tokenizer;
    //tokenizer.Init(vocab);
    //const TextTokenizer *tokenizer = engine.tokenizer();
    //Macro_RetxFalseIf(tokenizer == nullptr, LogError("Null tokenizer"));

    string perf_stat_path = AppEnv::DataRootDir() + "perf_stat.txt";
    ofstream perf_stat_writer(perf_stat_path);
    if (!perf_stat_writer) {
        LogWarning("Failed to open the perf_stat file");
    }

    LogKeyInfo("========== Inference ==========");

    //const auto &context = engine.context();
    string prompt = " ";
    bool add_bos = false;
    TokenizationAlg tok_alg = TokenizationAlg::Auto;

    vector<int> input_tokens;
    engine.Tokenize(input_tokens, prompt, add_bos, tok_alg);

    bool is_study_mode = engine_config.debug.is_study_mode;
    if (is_study_mode)
    {
        cout << "tokens: ";
        PrintInputTokens(input_tokens, engine, cout, true);
    }

    int context_len = 0;

    InferenceResult result;
    //ret = engine.Evaluate(result, input_tokens, context_len, cpu_threads);

    context_len += (int)input_tokens.size();
    input_tokens.clear();

    const ModelSpec *model_spec = engine_config.models.empty() ? nullptr : &engine_config.models[0];
    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(model_spec->network_structure);
    bool is_bloom = model_spec != nullptr && model_spec->network_structure == NetworkType::BLOOM;
    if (is_bloom)
    {
        //tok_alg = TokenizationAlg::FMM;
        add_bos = false;
    }

    SamplingStrategy::QueryOptions query_options;
    query_options.strategy_id = engine.GetSamplingStrategyId();
    query_options.random_seed = test_config.query_random_seed;
    query_options.temperature = test_config.temperature;
    LogKeyInfo("Decoding strategy: %d", query_options.strategy_id);

    string prompt_template = engine.GetPromptTemplate();
    if (prompt_template.empty())
    {
        //prompt_template = "\n\n### Instruction:\n\n{question}\n\n### Response:\n\n{res_prefix}";
        prompt_template = "{question}\n{res_prefix}";
    }

    int query_id_for_display = 1;
    wstring res_prefix_for_display;

    const auto &query_list = test_config.query_list;
    int query_list_size = (int)query_list.size();
    //int max_query_count = engine_config.debug.is_study_mode ? 1 : query_list_size;
    int max_query_count = query_list_size;

    map<int, string> response_map;
    vector<int> encoder_input_tokens, decoder_prefix_tokens;
    EncoderInput encoder_input;
    DecoderPrefix decoder_prefix;
    int query_idx = 0;
    for (; query_idx < max_query_count; query_idx++)
    {
        int text_idx = query_idx % query_list_size;
        const LlmQuery &query = query_list[text_idx];
        engine.BuildEncoderInput(encoder_input, query);
        engine.BuildDecoderInput(decoder_prefix, query);

        int query_id = engine.AddQuery(encoder_input_tokens, decoder_prefix_tokens,
            encoder_input, decoder_prefix, query_options, tok_alg);
        if (query_id > 0)
        {
            response_map[query_id] = StringUtil::ToUtf8(query.response_prefix);
            if (query_id_for_display == query_id) {
                res_prefix_for_display = query.response_prefix;
            }
        }
        else if (query_id < 0)
        {
            LogError(query_id == -1 ? "Invalid query" : "Error occurred");
            return false;
        }
        else if(query_id == 0)
        {
            //LogWarning("Busy");
            LogKeyInfo("query_idx: %d", query_idx);
            break;
        };
    }

    if (max_query_count == 1)
    {
        wstring wide_str;
        //engine.Tokenize(input_tokens, prompt, add_bos, tok_alg);
        if (!encoder_input_tokens.empty())
        {
            wide_str = StringUtil::Utf8ToWideStr(encoder_input.core_text);
            cout << "Encoder input core text: " << StringUtil::ToConsoleEncoding(wide_str) << endl;
            cout << "tokens: ";
            PrintInputTokens(encoder_input_tokens, engine, cout, true);
        }

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
        cout << "Prompt: " << StringUtil::ToConsoleEncoding(ss.str()) << endl;
        cout << "Decoder prefix tokens: ";
        PrintInputTokens(decoder_prefix_tokens, engine, cout, true);
        //return true;
    }

    int output_len = 0, total_output_len = 0;
    //int max_output_len = is_study_mode ? 5 : 1000;
    int max_output_len = is_study_mode ? 3 : 300;
    if (!is_study_mode && test_config.is_long_context_test) {
        max_output_len = 50000;
    }

    cout << endl << "Inference results:" << endl;

    int end_query_count = 0;
    bool is_continue = true;

    TaskMonitor tm;
    while (ret && is_continue)
    {
        ret = engine.Infer(result);
        if (result.items.empty()) {
            break;
        }

        map<int, QueryNextToken> query_map;
        for (int item_idx = 0; item_idx < (int)result.items.size(); item_idx++)
        {
            const QueryInferenceResult &query_res = *result.items[item_idx];

            int next_token_id = query_res.next_tokens[0].id;
            float next_token_score = query_res.next_tokens[0].weight;
            bool is_eos = next_token_id == vocab.eos();
            bool is_bos = next_token_id == vocab.bos();

            if (!(is_eos || is_bos) && query_res.query_id == query_id_for_display) {
                output_len++;
            }
            if (!(is_eos || is_bos)) {
                total_output_len++;
            }

            QueryNextToken next_token;
            next_token.id = next_token_id;
            next_token.is_end = is_eos || is_bos || output_len >= max_output_len
                || is_encoder_only;
            //next_token.is_end = output_len >= max_output_len;
            query_map[query_res.query_id] = next_token;
            //LogKeyInfo("query_res.query_id: %d", query_res.query_id);

            if (next_token.is_end) {
                end_query_count++;
            }

            string token_str = engine.OutputTokenIdToStr(next_token_id);
            auto iter_find = response_map.find(query_res.query_id);
            if (iter_find == response_map.end()) {
                response_map[query_res.query_id] = token_str;
            }
            else {
                iter_find->second += token_str;
            }

            if (query_res.query_id == query_id_for_display)
            {
                if (is_study_mode)
                {
                    PrintTokenStr_StudyMode(next_token.id, token_str, next_token_score);
                    PrintPerfStat(perf_stat_writer, result, engine);
                    result.perf_stat_pre.Clear(); //!!!
                }
                else
                {
                    if (is_encoder_only)
                    {
                        PrintEncoderOutput(query_res, engine);
                    }
                    else
                    {
                        PrintTokenStr(next_token.id, token_str, next_token_score,
                            is_bos, is_eos, max_query_count, res_prefix_for_display);
                    }
                }
            }
        } //for each query

        engine.CommitInferenceResult(query_map);

        if (query_idx < max_query_count)
        {
            int text_idx = query_idx % query_list_size;
            const LlmQuery query = query_list[text_idx];
            engine.BuildEncoderInput(encoder_input, query);
            engine.BuildDecoderInput(decoder_prefix, query);
            int query_id = engine.AddQuery(encoder_input, decoder_prefix, query_options, tok_alg);
            if (query_id > 0) {
                query_idx++;
            }
        }

        int query_count = engine.QueryCount();
        if (is_study_mode) {
            LogKeyInfo("Query count: %d", query_count);
        }
        if (query_count <= 0 || end_query_count > 8) {
            break;
        }

        //context_len += (int)input_tokens.size();
        //input_tokens.clear();
        //input_tokens.push_back(result.next_tokens[0].id);
    }

    for (auto iter = response_map.begin(); iter != response_map.end(); iter++)
    {
        int query_id = iter->first;
        if (!is_study_mode && query_id == query_id_for_display) {
            continue;
        }

        cout << "\n" << "===== Results of query " << query_id << ":\n";
        wstring result_wstr = StringUtil::Utf8ToWideStr(iter->second);
        cout << StringUtil::ToConsoleEncoding(result_wstr) << "\n";
    }

    cout << endl;
    float time_cost = tm.GetElapsedTime() * 1.0f;
    LogKeyInfo("Time per token: %.2f ms (%.2fs / %d)", time_cost / output_len,
        time_cost / 1000, output_len);
    LogKeyInfo("Throughput: %.1f tokens/sec", total_output_len / time_cost * 1000);

    perf_stat_writer.close();
    return ret;
}

int main(int argc, const char *argv[])
{
    /// application environment
    string app_name = "llm_inference";
    string app_dir = Path::GetModuleDir();
    string config_path = app_dir + app_name + ".ini";
    if (argc > 1)
    {
        bool is_abs = Path::IsAbsolute(argv[1]);
        if (is_abs)
        {
            config_path = argv[1];
        }
        else
        {
            config_path = app_dir + argv[1];
            if (!Path::FileExists(config_path.c_str())) {
                config_path = app_dir + "../" + argv[1];
            }
        }
    }
    else if (!Path::FileExists(config_path.c_str()))
    {
        config_path = app_dir + "../" + app_name + ".ini";
    }

    string env_file = argc > 2 ? argv[2] : config_path;
    bool ret = InitAppEnv(env_file, app_name, "0.1.0");
    if (!ret) {
        LogError("Fail to initialize the application environment");
        return 9999;
    }

    ///
    Run(config_path);

    FinalizeAppEnv();
    return ret ? 0 : 1;
}
