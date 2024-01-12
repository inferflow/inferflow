#include "sampling_strategy.h"
#include <cmath>
#include <numeric>
#include "sslib/top_k_queue.h"
#include "sslib/log.h"
#include "sslib/thread.h"
#include <algorithm>
#include "transformer_types.h"
#include <random>

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

vector<IdWeight<float>> get_sorted_topk(const vector<IdWeight<float>> &pool, int max_queue_len)
{
    vector<IdWeight<float>> new_pool;
    TopKQueue top_k_queue(max_queue_len);
    for (const auto &item : pool) {
        top_k_queue.UpdateTopK(item);
    }
    top_k_queue.GetList(new_pool);
    return new_pool;

}

vector<IdWeight<float>> topp_topk_filter_on_sorted(const vector<IdWeight<float>> &pool, float top_p, int top_k)
{
    vector<IdWeight<float>> new_pool;
    float cumulative_p = 0; //cumulative probability
    for (const auto &item : pool)
    {
        cumulative_p += item.weight;
        new_pool.push_back(item);
        //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
        if (cumulative_p >= top_p || (int)new_pool.size() >= top_k) {
            break;
        }
    }
    return new_pool;
}

std::string print_vector(std::vector<int> x)
{
    std::string str = "[";
    bool begin = true;
    for (const auto& t : x){
         if (begin){
              begin = false;
         }
         else{
              str = str + ", ";
         }
         str = str + std::to_string(t);
    }
    str += "]";
    return str;
}

SamplingStrategy::~SamplingStrategy()
{
    for (auto iter = query_data_map_.begin(); iter != query_data_map_.end(); iter++)
    {
        if (iter->second != nullptr) {
            delete iter->second;
            iter->second = nullptr;
        }
    }
    query_data_map_.clear();
}

//bool SamplingStrategy::SetConfig(const JsonObject &jobj, const JsonDoc &jdoc)
//{
//    (void)jobj; (void)jdoc;
//    return false;
//}

void SamplingStrategy::BeginQuery(int query_id, const QueryOptions &opt,
    const string &config_str, const JsonParser *jparser)
{
    (void)config_str; (void)jparser;
    QueryData *query_data = AddOrFindQueryData(query_id);
    if (query_data != nullptr)
    {
        query_data->options = opt;
        if (opt.random_seed != 0) {
            query_data->rng.SetSeed(opt.random_seed);
        }
    }
}

void SamplingStrategy::EndQuery(int query_id)
{
    auto iter = query_data_map_.find(query_id);
    if (iter == query_data_map_.end()) {
        return;
    }

    if (iter->second != nullptr) {
        delete iter->second;
    }
    query_data_map_.erase(iter);
}

void SamplingStrategy::SoftMax(vector<IdWeight<float>> &items, float temperature)
{
    if (items.empty()) {
        return;
    }

    std::sort(items.begin(), items.end(), [](const IdWeight<float> & a, const IdWeight<float> & b) {
            return a.weight > b.weight;
        });

    const float  tiny_value = 0.001f;
    if (temperature < tiny_value) {
        temperature = tiny_value;
    }

    float max_value = items[0].weight;
    for (int idx = 0; idx < (int)items.size(); idx++)
    {
        const auto &item = items[idx];
        if (max_value < item.weight) {
            max_value = item.weight;
        }
    }

    float sum = 0;
    for (int idx = 0; idx < (int)items.size(); idx++)
    {
        auto &item = items[idx];
        item.weight = exp((item.weight - max_value) / temperature);
        sum += item.weight;
    }

    if (sum < 0.00001f) {
        sum = 0.00001f;
    }
    for (int idx = 0; idx < (int)items.size(); idx++)
    {
        auto &item = items[idx];
        item.weight /= sum;
    }
}

SamplingStrategy::QueryData* SamplingStrategy::AddOrFindQueryData(int query_id)
{
    auto iter = query_data_map_.find(query_id);
    if (iter != query_data_map_.end() && iter->second != nullptr) {
        return iter->second;
    }

    QueryData *query_data = NewQueryData();
    query_data_map_[query_id] = query_data;
    return query_data;
}

class TopKBuildingThread : public Thread
{
public:
    TopKBuildingThread() {};
    virtual ~TopKBuildingThread() {};

    bool Init(int id, int thread_count, const SamplingInput &input,
        const StdVocabulary &vocab, int max_queue_len)
    {
        id_ = id;
        thread_count_ = thread_count;
        input_ = &input;
        vocab_ = &vocab;
        max_queue_len_ = max_queue_len;
        time_cost_ = 0;
        return true;
    }

    const vector<IdWeight<float>>* Output() const
    {
        return &out_pool_;
    }

    float TimeCost() const
    {
        return time_cost_;
    }

    virtual void Run() override
    {
        TaskMonitor tm;
        uint32_t unk = (uint32_t)vocab_->unk();
        int vocab_size = vocab_->Size();
        bool is_fp16 = !input_->candidates_fp16.empty();
        int item_count = input_->CandidateCount();

        int items_per_thread = item_count / thread_count_;
        int start_idx = id_ * items_per_thread;
        int end_idx = id_ + 1 < thread_count_ ? (id_ + 1) * items_per_thread : item_count;
        //LogKeyInfo("idx range: [%d, %d)", start_idx, end_idx);

        TopKQueue top_k_queue(max_queue_len_);
        int proc_count = 0;
        for (int idx = start_idx; idx < end_idx; idx++)
        {
            if (idx < vocab_size && vocab_->token_array[idx].type != (int)TokenType::Invalid
                && vocab_->token_array[idx].id != (int)unk)
            {
                proc_count++;
                float score = is_fp16 ? (float)input_->candidates_fp16[idx] : input_->candidates[idx];
                top_k_queue.UpdateTopK(idx, score);
                //if (top_k_queue.GetTop().weight < score + 0.01f) {
                //    LogKeyInfo("idx: %d, score: %f", idx, score);
                //}
            }
        }

        top_k_queue.GetList(out_pool_);
        time_cost_ = tm.GetElapsedTime(false) / 1000.0f;
    }

    virtual void CancelThread() override {};

protected:
    int id_ = 0;
    int thread_count_ = 1;
    const SamplingInput *input_ = nullptr;
    const StdVocabulary *vocab_ = nullptr;
    int max_queue_len_ = 0;
    vector<IdWeight<float>> out_pool_;
    float time_cost_ = 0;
};

//static
bool SamplingStrategy::GetSortedTopK(vector<IdWeight<float>> &pool,
    const SamplingInput &input, const StdVocabulary &vocab,
    int max_queue_len, int thread_count)
{
    TopKQueue top_k_queue(max_queue_len);

    if (thread_count > 1)
    {
        TaskMonitor tm;
        auto mythread = std::thread([] {});
        mythread.join();
        float time_cost = tm.GetElapsedTime(false) / 1000.0f;
        LogKeyInfo("Empty thread time cost: %.3f", time_cost);
        tm.Start();

        ThreadList thread_list;
        for (int thread_id = 0; thread_id < thread_count; thread_id++)
        {
            auto *new_thread = new TopKBuildingThread;
            new_thread->Init(thread_id, thread_count, input, vocab, max_queue_len);
            new_thread->Create();
            thread_list.Add(new_thread);
        }

        thread_list.Join();
        time_cost = tm.GetElapsedTime(false) / 1000.0f;
        LogKeyInfo("overall time cost: %.3f", time_cost);
        
        for (int thread_id = 0; thread_id < thread_count; thread_id++)
        {
            const auto *thread_ptr = (const TopKBuildingThread*)thread_list.Get(thread_id);
            LogKeyInfo("Time cost of thread %d: %.3f", thread_id, thread_ptr->TimeCost());
        }

        for (int thread_id = 0; thread_id < thread_count; thread_id++)
        {
            const auto *thread_ptr = (const TopKBuildingThread*)thread_list.Get(thread_id);
            const auto *output = thread_ptr->Output();
            for (const auto &item : *output)
            {
                top_k_queue.UpdateTopK(item.id, item.weight);
            }
        }
    }
    else
    {
        uint32_t unk = (uint32_t)vocab.unk();
        int vocab_size = vocab.Size();
        bool is_fp16 = !input.candidates_fp16.empty();
        int item_count = input.CandidateCount();

        int proc_count = 0;
        for (int idx = 0; idx < item_count; idx++)
        {
            if (idx < vocab_size && vocab.token_array[idx].type != (int)TokenType::Invalid
                && vocab.token_array[idx].id != (int)unk)
            {
                proc_count++;
                float score = is_fp16 ? (float)input.candidates_fp16[idx] : input.candidates[idx];
                top_k_queue.UpdateTopK(idx, score);
                //if (top_k_queue.GetTop().weight < score + 0.01f) {
                //    LogKeyInfo("idx: %d, score: %f", idx, score);
                //}
            }
        }
    }

    top_k_queue.GetList(pool);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// class StdSamplingStrategy

StdSamplingStrategy::~StdSamplingStrategy()
{
    Clear();
}

void StdSamplingStrategy::Clear()
{
}

bool StdSamplingStrategy::Init(const Config &cfg)
{
    config_ = cfg;
    //rng_.SetSeed(1);
    return true;
}

void StdSamplingStrategy::BeginQuery(int query_id, const QueryOptions &opt,
    const string &config_str, const JsonParser *jparser)
{
    SamplingStrategy::BeginQuery(query_id, opt, config_str, jparser);

    QueryData *base_query_data = AddOrFindQueryData(query_id);
    StdQueryData *query_data = dynamic_cast<StdQueryData*>(base_query_data);
    if (query_data == nullptr) {
        return;
    }

    query_data->config = config_;
    if (jparser != nullptr && !config_str.empty())
    {
        JsonDoc jdoc;
        bool ret = jparser->ParseUtf8(jdoc, config_str);
        if (!ret)
        {
            LogWarning("The %s of query %d is not valid JSON format",
                "decoding strategy configuration", query_id);
            return;
        }

        JsonObject jobj = jdoc.GetJObject();
        auto &config = query_data->config;

        jobj.GetFieldValue(config.min_k, L"min_k", jdoc);
        jobj.GetFieldValue(config.max_k, L"max_k", jdoc);
        jobj.GetFieldValue(config.top_p, L"top_p", jdoc);
        jobj.GetFieldValue(config.pool_size, L"pool_size", jdoc);
        jobj.GetFieldValue(config.eos_bypassing_max, L"eos_bypassing_max", jdoc);
    }
}

bool StdSamplingStrategy::ChooseTokens(SamplingOutput &output,
    const SamplingInput &input, const StdVocabulary &vocab,
    SamplingStrategyId strategy_id, int thread_count)
{
    output.Clear();
    int item_count = input.CandidateCount();
    uint32_t eos = (uint32_t)vocab.eos();

    QueryData *base_query_data = AddOrFindQueryData(input.query_id);
    StdQueryData *query_data = dynamic_cast<StdQueryData*>(base_query_data);
    const Config &config = query_data != nullptr ? query_data->config : config_;
    auto &rng = query_data != nullptr ? query_data->rng : rng_;
    float temperature = query_data != nullptr ? query_data->options.temperature : 1.0f;

    int max_queue_len = 1;
    float top_p = 1.0f;
    if (strategy_id != SamplingStrategyId::Greedy) {
        max_queue_len = min(config.pool_size, item_count);
    }

    if (strategy_id == SamplingStrategyId::StdSampling
        || strategy_id == SamplingStrategyId::TopP)
    {
        top_p = config.top_p;
    }

    vector<IdWeight<float>> pool;
    GetSortedTopK(pool, input, vocab, max_queue_len, thread_count);

    SoftMax(pool, temperature);
    int top_k = min((int)pool.size(), config.max_k);
    output.token_pool = topp_topk_filter_on_sorted(pool, top_p, top_k);

    int token_pool_size = (int)output.token_pool.size();
    rng.RandomSampling(output.selected, output.token_pool, 1);

    if (!output.selected.empty() && output.token_pool[0].id == eos
        && output.selected[0].id != eos)
    {
        output.flag = 1;
        output.flag_score = output.selected[0].weight;
        //LogKeyInfo("[eos_not_selected: %d_%.3f]", output.token_pool[0].id,
        //    output.token_pool[0].weight);
    }

    if (config.eos_bypassing_max > 0 && !output.selected.empty())
    {
        const auto &selected = output.selected[0];
        //if (selected.id == eos) {
        //    LogKeyInfo("Selected: %d (%.3f)", selected.id, selected.weight);
        //}

        if (selected.id == eos && token_pool_size > 1 && query_data != nullptr
            && query_data->eos_bypassing_count < config.eos_bypassing_max)
        {
            output.flag = 2;
            output.flag_score = selected.weight;
            //LogKeyInfo("[eos_bypassing_%d: %d_%.3f]", query_data->eos_bypassing_count,
            //    selected.id, selected.weight);
            for (const auto &item : output.token_pool)
            {
                if (item.id != eos)
                {
                    output.selected[0] = item;
                    query_data->eos_bypassing_count++;
                    break;
                }
                //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
            }
        }
    }

    return true;
}

SamplingStrategy::QueryData* StdSamplingStrategy::NewQueryData()
{
    return new StdQueryData;
}

////////////////////////////////////////////////////////////////////////////////
// class FsdSamplingStrategy

FsdSamplingStrategy::~FsdSamplingStrategy()
{
    Clear();
}

void FsdSamplingStrategy::Clear()
{
}

bool FsdSamplingStrategy::Init(const Config &cfg)
{
    config_ = cfg;
    //rng_.SetSeed(1);
    return true;
}

bool FsdSamplingStrategy::ChooseTokens(SamplingOutput &output,
    const SamplingInput &input, const StdVocabulary &vocab,
    SamplingStrategyId strategy_id, int thread_count)
{
    (void)strategy_id;
    output.Clear();
    int item_count = input.CandidateCount();
    uint32_t eos = (uint32_t)vocab.eos();

    QueryData *base_query_data = AddOrFindQueryData(input.query_id);
    FsdQueryData *query_data = dynamic_cast<FsdQueryData*>(base_query_data);
    auto &rng = query_data != nullptr ? query_data->rng : rng_;
    const Config &config = query_data != nullptr ? query_data->config : config_;
    float temperature = query_data != nullptr ? query_data->options.temperature : 1.0f;

    (void)rng;
    vector<IdWeight<float>> pool;
    int max_queue_len = min(config.pool_size, item_count);
    GetSortedTopK(pool, input, vocab, max_queue_len, thread_count);

    SoftMax(pool, temperature);
    pool = get_sorted_topk(pool, config.k);

    vector<int> pool_ids;
    for (const auto &item : pool) {
        pool_ids.push_back(item.id);
    }

    if (! query_data->started ){
        vector<int> prompt;
        for (const auto token : *input.prefix){
            prompt.push_back(token);
        }
        for (const auto token : *input.cur_tokens){
            prompt.push_back(token);
        }
        query_data->ngram.initialize(prompt);
        query_data->started = true;
       
    }
    std::map<int, float> penalty = query_data->ngram.penalize(pool_ids);
    for (auto &item : pool){
        if (! (penalty.find(item.id) == penalty.end())){
            item.weight = (1 - config_.alpha) * item.weight - config_.alpha * penalty[item.id];
        }
    }
    output.token_pool = get_sorted_topk(pool, (int) pool.size());
    int token_pool_size = (int)output.token_pool.size();
    output.selected.push_back(output.token_pool[0]);

    if (!output.selected.empty() && output.token_pool[0].id == eos
        && output.selected[0].id != eos)
    {
        LogKeyInfo("[eos_not_selected: %d_%.3f]", output.token_pool[0].id,
            output.token_pool[0].weight);
    }

    if (config_.eos_bypassing_max > 0 && !output.selected.empty())
    {
        const auto &selected = output.selected[0];
        //if (selected.id == eos) {
        //    LogKeyInfo("Selected: %d (%.3f)", selected.id, selected.weight);
        //}

        if (selected.id == eos && token_pool_size > 1 && query_data != nullptr
            && query_data->eos_bypassing_count < config_.eos_bypassing_max)
        {
            LogKeyInfo("[eos_bypassing_%d: %d_%.3f]", query_data->eos_bypassing_count,
                selected.id, selected.weight);
            for (const auto &item : output.token_pool)
            {
                if (item.id != eos)
                {
                    output.selected[0] = item;
                    query_data->eos_bypassing_count++;
                    break;
                }
                //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
            }
        }
    }

    query_data->ngram.update(output.selected[0].id);

    return true;
}

SamplingStrategy::QueryData* FsdSamplingStrategy::NewQueryData()
{
    auto *query_data = new FsdQueryData;
    query_data->ngram = NGram(config_.n, config_.beta);
    return query_data;
}

////////////////////////////////////////////////////////////////////////////////
// class FSDRandomSamplingStrategy

RandomizedFSDSamplingStrategy::~RandomizedFSDSamplingStrategy()
{
    Clear();
}

void RandomizedFSDSamplingStrategy::Clear()
{
}

bool RandomizedFSDSamplingStrategy::Init(const Config &cfg)
{
    config_ = cfg;
    //rng_.SetSeed(1);
    return true;
}

bool RandomizedFSDSamplingStrategy::ChooseTokens(SamplingOutput &output,
    const SamplingInput &input, const StdVocabulary &vocab,
    SamplingStrategyId strategy_id, int thread_count)
{
    (void)strategy_id; (void)thread_count;
    output.Clear();
    int item_count = input.CandidateCount();
    uint32_t eos = (uint32_t)vocab.eos();

    QueryData *base_query_data = AddOrFindQueryData(input.query_id);
    RandomizedFSDQueryData *query_data = dynamic_cast<RandomizedFSDQueryData*>(base_query_data);
    auto &rng = query_data != nullptr ? query_data->rng : rng_;
    const Config &config = query_data != nullptr ? query_data->config : config_;
    float temperature = query_data != nullptr ? query_data->options.temperature : 1.0f;

    vector<IdWeight<float>> pool;
    int max_queue_len = min(config.pool_size, item_count);
    GetSortedTopK(pool, input, vocab, max_queue_len, thread_count);

    if  (query_data->new_tokens >= 10 || rng.NextFloat(0.0f, 1.0f) >= 0.5f)
    {
        SoftMax(pool, temperature);
        pool = get_sorted_topk(pool, config_.k);
        vector<int> pool_ids; 
        for (const auto &item : pool){
            pool_ids.push_back(item.id);
        }

        if (! query_data->started ){
            vector<int> prompt;
            for (const auto token : *input.prefix){
                prompt.push_back(token);
            }
            for (const auto token : *input.cur_tokens){
                prompt.push_back(token);
            }
            query_data->ngram.initialize(prompt);
            query_data->started = true;
        }
        std::map<int, float> penalty = query_data->ngram.penalize(pool_ids);
        for (auto &item : pool){
            if (! (penalty.find(item.id) == penalty.end())){
                item.weight = (1 - config_.alpha) * item.weight - config_.alpha * penalty[item.id];
            }
        }
        output.token_pool = get_sorted_topk(pool, (int) pool.size());
        output.selected.push_back(output.token_pool[0]);
    }
    else
    {
        float top_p = 1.0f;
        max_queue_len = min(config_.pool_size, item_count);
        top_p = config_.top_p;

        pool = get_sorted_topk(pool, max_queue_len);
        SoftMax(pool, temperature);
        int top_k = min((int)pool.size(), config_.max_k);
        pool = topp_topk_filter_on_sorted(pool, top_p, top_k);
        output.token_pool = pool;
        rng.RandomSampling(output.selected, output.token_pool, 1);
    }

    int token_pool_size = (int)output.token_pool.size();
    if (!output.selected.empty() && output.token_pool[0].id == eos
        && output.selected[0].id != eos)
    {
        LogKeyInfo("[eos_not_selected: %d_%.3f]", output.token_pool[0].id,
            output.token_pool[0].weight);
    }

    if (config_.eos_bypassing_max > 0 && !output.selected.empty())
    {
        const auto &selected = output.selected[0];
        //if (selected.id == eos) {
        //    LogKeyInfo("Selected: %d (%.3f)", selected.id, selected.weight);
        //}

        if (selected.id == eos && token_pool_size > 1 && query_data != nullptr
            && query_data->eos_bypassing_count < config_.eos_bypassing_max)
        {
            LogKeyInfo("[eos_bypassing_%d: %d_%.3f]", query_data->eos_bypassing_count,
                selected.id, selected.weight);
            for (const auto &item : output.token_pool)
            {
                if (item.id != eos)
                {
                    output.selected[0] = item;
                    query_data->eos_bypassing_count++;
                    break;
                }
                //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
            }
        }
    }

    query_data->ngram.update(output.selected[0].id);
    query_data->new_tokens = query_data->new_tokens + 1;

    return true;
}

SamplingStrategy::QueryData* RandomizedFSDSamplingStrategy::NewQueryData()
{
    auto *query_data = new RandomizedFSDQueryData;
    query_data->ngram = NGram(config_.n, config_.beta);
    return query_data;
}


////////////////////////////////////////////////////////////////////////////////
// class MinPSamplingStrategy

MinPSamplingStrategy::~MinPSamplingStrategy()
{
    Clear();
}

void MinPSamplingStrategy::Clear()
{
}

bool MinPSamplingStrategy::Init(const Config &cfg)
{
    config_ = cfg;
    //rng_.SetSeed(1);
    return true;
}

bool MinPSamplingStrategy::ChooseTokens(SamplingOutput &output,
    const SamplingInput &input, const StdVocabulary &vocab,
    SamplingStrategyId strategy_id, int thread_count)
{
    (void)strategy_id; (void)thread_count;
    output.Clear();
    int item_count = input.CandidateCount();
    uint32_t eos = (uint32_t)vocab.eos();

    QueryData *base_query_data = AddOrFindQueryData(input.query_id);
    MinPQueryData *query_data = dynamic_cast<MinPQueryData*>(base_query_data);
    auto &rng = query_data != nullptr ? query_data->rng : rng_;
    const Config &config = query_data != nullptr ? query_data->config : config_;
    float temperature = query_data != nullptr ? query_data->options.temperature : 1.0f;

    vector<IdWeight<float>> pool;
    int max_queue_len = min(config.pool_size, item_count);
    GetSortedTopK(pool, input, vocab, max_queue_len, thread_count);

    SoftMax(pool, temperature);

    float scale = pool[0].weight; // scale by max prob
    vector<IdWeight<float>> new_pool = { pool[0] };
    for (int i = 1; i < (int) pool.size(); ++i) {
        if (pool[i].weight < config_.min_p * scale) {
            break; // prob too small
        }
        new_pool.push_back(pool[i]);
    }
    output.token_pool = new_pool;
    rng.RandomSampling(output.selected, output.token_pool, 1);

    int token_pool_size = (int)output.token_pool.size();
    if (!output.selected.empty() && output.token_pool[0].id == eos
        && output.selected[0].id != eos)
    {
        LogKeyInfo("[eos_not_selected: %d_%.3f]", output.token_pool[0].id,
            output.token_pool[0].weight);
    }

    if (config_.eos_bypassing_max > 0 && !output.selected.empty())
    {
        const auto &selected = output.selected[0];
        //if (selected.id == eos) {
        //    LogKeyInfo("Selected: %d (%.3f)", selected.id, selected.weight);
        //}

        if (selected.id == eos && token_pool_size > 1 && query_data != nullptr
            && query_data->eos_bypassing_count < config_.eos_bypassing_max)
        {
            LogKeyInfo("[eos_bypassing_%d: %d_%.3f]", query_data->eos_bypassing_count,
                selected.id, selected.weight);
            for (const auto &item : output.token_pool)
            {
                if (item.id != eos)
                {
                    output.selected[0] = item;
                    query_data->eos_bypassing_count++;
                    break;
                }
                //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
            }
        }
    }
    return true;
}

SamplingStrategy::QueryData* MinPSamplingStrategy::NewQueryData()
{
    auto *query_data = new MinPQueryData;
    return query_data;
}

////////////////////////////////////////////////////////////////////////////////
// class TFSSamplingStrategy

TFSSamplingStrategy::~TFSSamplingStrategy()
{
    Clear();
}

void TFSSamplingStrategy::Clear()
{
}

bool TFSSamplingStrategy::Init(const Config &cfg)
{
    config_ = cfg;
    //rng_.SetSeed(1);
    return true;
}

bool TFSSamplingStrategy::ChooseTokens(SamplingOutput &output,
    const SamplingInput &input, const StdVocabulary &vocab,
    SamplingStrategyId strategy_id, int thread_count)
{
    (void)strategy_id; (void)thread_count;
    output.Clear();
    int item_count = input.CandidateCount();
    uint32_t eos = (uint32_t)vocab.eos();

    QueryData *base_query_data = AddOrFindQueryData(input.query_id);
    TFSQueryData *query_data = dynamic_cast<TFSQueryData*>(base_query_data);
    auto &rng = query_data != nullptr ? query_data->rng : rng_;
    const Config &config = query_data != nullptr ? query_data->config : config_;
    float temperature = query_data != nullptr ? query_data->options.temperature : 1.0f;

    vector<IdWeight<float>> pool;
    int max_queue_len = min(config.pool_size, item_count);
    GetSortedTopK(pool, input, vocab, max_queue_len, thread_count);

    SoftMax(pool, temperature);

    std::vector<float> first_derivatives((int) pool.size() - 1);
    std::vector<float> second_derivatives((int) pool.size() - 2);
    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = pool[i].weight - pool[i+1].weight;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = std::abs(second_derivatives[i]);
    }

    const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);
    if (second_derivatives_sum > 1e-6f) {
        for (float & value : second_derivatives) {
            value /= second_derivatives_sum;
        }
    } else {
        for (float & value : second_derivatives) {
            value = 1.0f / second_derivatives.size();
        }
    }
    float cum_sum = 0.0f;
    vector<IdWeight<float>> new_pool = { pool[0] };
    for (size_t i = 1; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];
        if (cum_sum > config_.z) {
            break;
        }
        new_pool.push_back(pool[i]);
    }
    output.token_pool = new_pool;
    rng.RandomSampling(output.selected, output.token_pool, 1);

    int token_pool_size = (int)output.token_pool.size();
    if (!output.selected.empty() && output.token_pool[0].id == eos
        && output.selected[0].id != eos)
    {
        LogKeyInfo("[eos_not_selected: %d_%.3f]", output.token_pool[0].id,
            output.token_pool[0].weight);
    }

    if (config_.eos_bypassing_max > 0 && !output.selected.empty())
    {
        const auto &selected = output.selected[0];
        //if (selected.id == eos) {
        //    LogKeyInfo("Selected: %d (%.3f)", selected.id, selected.weight);
        //}

        if (selected.id == eos && token_pool_size > 1 && query_data != nullptr
            && query_data->eos_bypassing_count < config_.eos_bypassing_max)
        {
            LogKeyInfo("[eos_bypassing_%d: %d_%.3f]", query_data->eos_bypassing_count,
                selected.id, selected.weight);
            for (const auto &item : output.token_pool)
            {
                if (item.id != eos)
                {
                    output.selected[0] = item;
                    query_data->eos_bypassing_count++;
                    break;
                }
                //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
            }
        }
    }
    return true;
}

SamplingStrategy::QueryData* TFSSamplingStrategy::NewQueryData()
{
    auto *query_data = new TFSQueryData;
    return query_data;
}

////////////////////////////////////////////////////////////////////////////////
// class TypicalSamplingStrategy

TypicalSamplingStrategy::~TypicalSamplingStrategy()
{
    Clear();
}

void TypicalSamplingStrategy::Clear()
{
}

bool TypicalSamplingStrategy::Init(const Config &cfg)
{
    config_ = cfg;
    //rng_.SetSeed(1);
    return true;
}

bool TypicalSamplingStrategy::ChooseTokens(SamplingOutput &output,
    const SamplingInput &input, const StdVocabulary &vocab,
    SamplingStrategyId strategy_id, int thread_count)
{
    (void)strategy_id; (void)thread_count;
    output.Clear();
    int item_count = input.CandidateCount();
    uint32_t eos = (uint32_t)vocab.eos();

    QueryData *base_query_data = AddOrFindQueryData(input.query_id);
    TypicalQueryData *query_data = dynamic_cast<TypicalQueryData*>(base_query_data);
    auto &rng = query_data != nullptr ? query_data->rng : rng_;
    const Config &config = query_data != nullptr ? query_data->config : config_;
    float temperature = query_data != nullptr ? query_data->options.temperature : 1.0f;

    vector<IdWeight<float>> pool;
    int max_queue_len = min(config.pool_size, item_count);
    GetSortedTopK(pool, input, vocab, max_queue_len, thread_count);

    SoftMax(pool, temperature);

    float entropy = 0.0f;
    for (size_t i = 0; i < pool.size(); ++i) {
        entropy += -pool[i].weight * logf(pool[i].weight);
    }

    std::vector<float> shifted_scores;
    for (size_t i = 0; i < pool.size(); ++i) {
        float shifted_score = fabsf(-logf(pool[i].weight) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    std::vector<size_t> indices(pool.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    float cum_sum = 0.0f;
    vector<IdWeight<float>> new_pool = { pool[indices[0]]};

    for (size_t i = 1; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += pool[idx].weight;
        if (cum_sum > config_.p) {
            break;
        }
        new_pool.push_back(pool[idx]);
    }

    output.token_pool = new_pool;
    rng.RandomSampling(output.selected, output.token_pool, 1);

    int token_pool_size = (int)output.token_pool.size();
    if (!output.selected.empty() && output.token_pool[0].id == eos
        && output.selected[0].id != eos)
    {
        LogKeyInfo("[eos_not_selected: %d_%.3f]", output.token_pool[0].id,
            output.token_pool[0].weight);
    }

    if (config_.eos_bypassing_max > 0 && !output.selected.empty())
    {
        const auto &selected = output.selected[0];
        //if (selected.id == eos) {
        //    LogKeyInfo("Selected: %d (%.3f)", selected.id, selected.weight);
        //}

        if (selected.id == eos && token_pool_size > 1 && query_data != nullptr
            && query_data->eos_bypassing_count < config_.eos_bypassing_max)
        {
            LogKeyInfo("[eos_bypassing_%d: %d_%.3f]", query_data->eos_bypassing_count,
                selected.id, selected.weight);
            for (const auto &item : output.token_pool)
            {
                if (item.id != eos)
                {
                    output.selected[0] = item;
                    query_data->eos_bypassing_count++;
                    break;
                }
                //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
            }
        }
    }

    return true;
}

SamplingStrategy::QueryData* TypicalSamplingStrategy::NewQueryData()
{
    auto *query_data = new TypicalQueryData;
    return query_data;
}

////////////////////////////////////////////////////////////////////////////////
// class MirostatSamplingStrategy

MirostatSamplingStrategy::~MirostatSamplingStrategy()
{
    Clear();
}

void MirostatSamplingStrategy::Clear()
{
}

bool MirostatSamplingStrategy::Init(const Config &cfg)
{
    config_ = cfg;
    //rng_.SetSeed(1);
    return true;
}

bool MirostatSamplingStrategy::ChooseTokens(SamplingOutput &output,
    const SamplingInput &input, const StdVocabulary &vocab,
    SamplingStrategyId strategy_id, int thread_count)
{
    (void)strategy_id; (void)thread_count;
    //std::cout << " Params:"<<config_.k << " " << config_.alpha << " " << config_.n  << " " << config_.beta  << " " << config_.eos_bypassing_max << endl;
    //std::cout << "-> prefix length: " << (*input.prefix).size() << std::endl;
    //std::cout << print_vector(*input.prefix) << std::endl;
    output.Clear();
    int item_count = input.CandidateCount();
    uint32_t eos = (uint32_t)vocab.eos();

    QueryData *base_query_data = AddOrFindQueryData(input.query_id);
    MirostatQueryData *query_data = dynamic_cast<MirostatQueryData*>(base_query_data);
    auto &rng = query_data != nullptr ? query_data->rng : rng_;
    const Config &config = query_data != nullptr ? query_data->config : config_;
    float temperature = query_data != nullptr ? query_data->options.temperature : 1.0f;

    vector<IdWeight<float>> pool;
    int max_queue_len = min(config.pool_size, item_count);
    GetSortedTopK(pool, input, vocab, max_queue_len, thread_count);

    map<int, float> logits;
    for (const auto & item : pool){
        logits[item.id] = item.weight;
    } 
    SoftMax(pool, temperature);

    int candidates_size = (int)std::distance(pool.begin(), std::find_if(pool.begin(), pool.end(), [&](const IdWeight<float> & item) {
        return -log2f(item.weight) > query_data->mu;
    }));

    if (candidates_size == 0) {
        candidates_size = 1;
    }

    vector<IdWeight<float>> new_pool;
    for (int i = 0; i < candidates_size; ++i) {
        new_pool.push_back(IdWeight<float>(pool[i].id, logits[pool[i].id]));
    }
    SoftMax(new_pool, temperature);
    // Normalize the probabilities of the remaining words
    output.token_pool = new_pool;
    rng.RandomSampling(output.selected, output.token_pool, 1);
    int token_pool_size = (int)output.token_pool.size();

    if (!output.selected.empty() && output.token_pool[0].id == eos
        && output.selected[0].id != eos)
    {
        LogKeyInfo("[eos_not_selected: %d_%.3f]", output.token_pool[0].id,
            output.token_pool[0].weight);
    }

    if (config_.eos_bypassing_max > 0 && !output.selected.empty())
    {
        const auto &selected = output.selected[0];
        //if (selected.id == eos) {
        //    LogKeyInfo("Selected: %d (%.3f)", selected.id, selected.weight);
        //}

        if (selected.id == eos && token_pool_size > 1 && query_data != nullptr
            && query_data->eos_bypassing_count < config_.eos_bypassing_max)
        {
            LogKeyInfo("[eos_bypassing_%d: %d_%.3f]", query_data->eos_bypassing_count,
                selected.id, selected.weight);
            for (const auto &item : output.token_pool)
            {
                if (item.id != eos)
                {
                    output.selected[0] = item;
                    query_data->eos_bypassing_count++;
                    break;
                }
                //LogKeyInfo("%d\t%s\t%.3f", item.id, vocab.IdToStr(item.id).c_str(), item.weight);
            }
        }
    }

    size_t X_idx = std::distance(new_pool.begin(), std::find_if(new_pool.begin(), new_pool.end(), [&](const IdWeight<float> & item) {
        return item.id == output.selected[0].id;
    }));
    float observed_surprise = -log2f(new_pool[X_idx].weight);
    query_data->mu = query_data->mu - config_.eta * (observed_surprise - config_.tau);
    return true;
}

SamplingStrategy::QueryData* MirostatSamplingStrategy::NewQueryData()
{
    auto *query_data = new MirostatQueryData;
    query_data->mu = 2 * config_.tau;
    return query_data;
}

TRANSFORMER_END
INFER_FLOW_END
