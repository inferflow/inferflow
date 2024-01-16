#include "transformer/inference_engine.h"
#include "tensor/host_tensor.h"
#include <sstream>
#include <map>
#include <thread>
#include <mutex>
#include "sslib/path.h"
#include "sslib/log.h"
#include "sslib/app_environment.h"

using namespace std;
using namespace sslib;
using namespace inferflow;
using namespace inferflow::transformer;

struct PPLQuery
{
    vector<int> tokens;
    const HostTensor *logits = nullptr;
};

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

struct PerplexityTestConfig
{
    string engine_config_file;
    string test_data_file;

    int max_length = 512;
    int stride = 512;

    float temperature = 1.0f;

    vector<PPLQuery> query_list;
};

bool LoadQueryList(vector<PPLQuery> &query_list, PerplexityTestConfig &config, InferenceEngine &engine, bool add_bos=false)
{
    ifstream file(config.test_data_file);
    if (!file)
    {
        LogError("Failed to open the file: %s", config.test_data_file.c_str());
        return false;
    }

    string all_text;
    copy(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), back_inserter(all_text));
    if (!all_text.empty() && all_text.back() == '\n') {
        all_text.pop_back();
    }

    vector<int> tokens_ids;
    engine.Tokenize(tokens_ids, all_text, add_bos, TokenizationAlg::Auto);

    int tokens_count = (int)tokens_ids.size();

    const auto &vocab = engine.vocabulary();
    int bos_id = vocab.bos();

    int max_length = config.max_length;
    int stride = config.stride;

    int start_idx = 0;
    for (; start_idx < tokens_count; start_idx += stride)
    {
        int end_idx = start_idx + max_length;
        if (end_idx > tokens_count) {
            end_idx = tokens_count;
        }

        PPLQuery query;
        query.tokens.insert(query.tokens.end(), tokens_ids.begin() + start_idx, tokens_ids.begin() + end_idx);
        query_list.push_back(query);
    }

    return true;
}

bool LoadPerplexityConfig(PerplexityTestConfig &config, const string &file_path)
{
    ConfigData config_data;
    bool ret = config_data.Load(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the configuration data"));

    ret = config_data.GetItem("main", "inference_engine_config", config.engine_config_file, ret);
    ret = ret && config_data.GetItem("main", "test_data_file", config.test_data_file, ret);

    config_data.GetItem("main", "max_length", config.max_length, false);
    config_data.GetItem("main", "stride", config.stride, false);
    config_data.GetItem("main", "temperature", config.temperature, false);
    Macro_RetFalseIf(!ret);

    return true;
}

results_log_softmax log_softmax(const vector<float> &token_logits, int token_id)
{
    int vocab_size = (int)token_logits.size();

    float max_logit = token_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        max_logit = max(max_logit, token_logits[i]);
    }
    double sum_exp = 0;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf(token_logits[i] - max_logit);
    }

    results_log_softmax token_result;
    token_result.log_softmax = token_logits[token_id] - max_logit - log(sum_exp);
    token_result.logit = token_logits[token_id];
    token_result.prob = expf(token_logits[token_id] - max_logit) / (float)sum_exp;

    return token_result;
}

void ProcessLogits(vector<thread> &workers, const PPLQuery &query, double &nll, double &nll2, int n_token)
{
    std::mutex mtx;
    int counter = 0;
    auto compute = [&mtx, &counter, &query, &nll, &nll2, n_token] ()
    {
        double local_nll = 0;
        double local_nll2 = 0;

        while (true)
        {
            std::unique_lock<std::mutex> lock(mtx);
            int idx = counter++;
            if (idx >= n_token - 1) {
                nll += local_nll;
                nll2 += local_nll2;
                break;
            }
            lock.unlock();

            vector<float> token_logits;
            int token_id = query.tokens[idx + 1];
            query.logits->CopyRow(token_logits, idx);
            auto result = log_softmax(token_logits, token_id);
            const double v = -result.log_softmax;
            local_nll += v;
            local_nll2 += v * v;
        }
    };
    for (auto &w : workers) {
        w = thread(compute);
    }
    compute();
    for (auto &w : workers) {
        w.join();
    }
}

bool Run(const string &config_path)
{
    PerplexityTestConfig config;
    bool ret = LoadPerplexityConfig(config, config_path);
    Macro_RetFalseIf(!ret);

    InferenceConfig engine_config;
    ret = InferenceEngine::LoadConfig(engine_config, config.engine_config_file, "transformer_engine");

    engine_config.max_concurrent_queries = 1;
    
    Macro_RetxFalseIf(!ret, LogError("Failed to load the inference configuration"));
    engine_config.data_dir = AppEnv::DataRootDir();
    const ModelSpec *model_spec = engine_config.models.empty() ? nullptr : &engine_config.models[0];
    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(model_spec->network_structure);
    if (!is_decoder_only) {
        LogError("The model is not a decoder-only transformer");
        return false;
    }

    bool is_bloom = model_spec != nullptr && model_spec->network_structure == NetworkType::BLOOM;
    bool add_bos = true;
    if (is_bloom) {
        add_bos = false;
    }

    InferenceEngine engine;
    ret = engine.Init(engine_config);
    if (!ret) {
        LogError("Failed to initialize the inference engine");
        return false;
    }

    LoadQueryList(config.query_list, config, engine, add_bos);

    LogKeyInfo("========== Start Calculate PPL ==========");
    vector<thread> workers(std::thread::hardware_concurrency() - 1);
    map<int, PPLQuery> ppl_query_map;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    InferenceResult result;

    SamplingStrategy::QueryOptions query_options;
    query_options.strategy_id = engine.GetSamplingStrategyId();
    query_options.random_seed = 1;
    query_options.temperature = config.temperature;

    const auto &query_list = config.query_list;
    int query_list_size = (int)query_list.size();
    int max_query_count = query_list_size;

    int query_idx = 0;
    for (; query_idx < max_query_count; query_idx++)
    {
        const auto &query = query_list[query_idx];
        int query_id = engine.AddQuery(query.tokens, query_options);
        ppl_query_map[query_id] = query;

        if (query_id < 0)
        {
            LogError(query_id == -1 ? "Invalid query" : "Error occurred");
            return false;
        }
        else if(query_id == 0)
        {
            LogKeyInfo("query_idx: %d", query_idx);
            break;
        };
    }

    bool is_continue = true;

    int query_count = 0;
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

            const auto &output_tensor = query_res.output_tensor;

            ppl_query_map[query_res.query_id].logits = &output_tensor;
            QueryNextToken next_token;
            next_token.id = engine.vocabulary().eos();
            next_token.is_end = true;
            query_map[query_res.query_id] = next_token;
            ProcessLogits(workers, ppl_query_map[query_res.query_id], nll, nll2, output_tensor.Rows());
            count += output_tensor.Rows() - 1;
            printf("[%d]%.4lf\n", query_count++, exp(nll / count));
            fflush(stdout);
        }

        engine.CommitInferenceResult(query_map);

        if (query_idx < max_query_count)
        {
            int text_idx = query_idx % query_list_size;
            const PPLQuery query = query_list[text_idx];
            int query_id = engine.AddQuery(query.tokens, query_options);
            if (query_id > 0) {
                query_idx++;
                ppl_query_map[query_id] = query;
            }
        }
        else
        {
            is_continue = false;
        }
    }

    nll2 /= count;
    nll /= count;
    const double ppl = exp(nll);
    nll2 -= nll * nll;
    if (nll2 > 0) {
        nll2 = sqrt(nll2/(count-1));
        printf("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
    } else {
        printf("Unexpected negative standard deviation of log(prob)\n");
    }

    return true;
}

int main(int argc, const char *argv[])
{
    string app_name = "perplexity";
    string app_dir = Path::GetModuleDir();
    string config_path = app_dir + app_name + ".ini";
    if (argc > 1) {
        config_path = app_dir + argv[1];
    }
    else if (!Path::FileExists(config_path.c_str())) {
        config_path = app_dir + "../" + app_name + ".ini";
    }

    string env_file = argc > 2 ? argv[2] : config_path;
    bool ret = InitAppEnv(env_file, app_name, "0.1.0");
    if (!ret) {
        LogError("Fail to initialize the application environment");
        return 9999;
    }

    Run(config_path);

    FinalizeAppEnv();
    return ret ? 0 : 1;
}