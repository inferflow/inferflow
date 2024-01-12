#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include "sslib/prime_types.h"
#include "sslib/random.h"
#include "sslib/json.h"
#include "common/data_types.h"
#include "common/std_vocabulary.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::vector;
using sslib::IdWeight;
using sslib::Random;
using sslib::JsonParser;
//using sslib::JsonDoc;
//using sslib::JsonObject;

std::string print_vector(std::vector<int> x);

struct SamplingInput
{
    int query_id = 0;
    const vector<int> *prefix = nullptr;
    const vector<int> *cur_tokens = nullptr;
    vector<float> candidates;
    vector<inferflow_fp16> candidates_fp16;

    int CandidateCount() const {
        return candidates_fp16.empty() ? (int)candidates.size() : (int)candidates_fp16.size();
    }
};

struct SamplingOutput
{
    vector<IdWeight<float>> token_pool;
    vector<IdWeight<float>> selected;
    int flag = 0;
    float flag_score = 0;

    void Clear()
    {
        token_pool.clear();
        selected.clear();
        flag = 0;
        flag_score = 0;
    }
};

enum class DecodingStrategyId
{
    Auto = 0,
    StdSampling,
    Greedy,
    TopK,
    TopP,
    FSD,
    RandomizedFSD,
    MinP,
    TFS,
    Typical,
    Mirostat
};
typedef DecodingStrategyId SamplingStrategyId;

//A sampling strategy or decoding strategy
class SamplingStrategy
{
public:
    struct QueryOptions
    {
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto;
        int random_seed = 0;
        float temperature = 1.0f;
    };

    class QueryData
    {
    public:
        QueryOptions options;
        Random rng; //random number generator;

    public:
        virtual ~QueryData() {};
    };

public:
    virtual ~SamplingStrategy();

    //virtual bool SetConfig(const JsonObject &jobj, const JsonDoc &jdoc);

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) = 0;

    //the config_str should be in JSON format
    virtual void BeginQuery(int query_id, const QueryOptions &opt, const string &config_str,
        const JsonParser *jparser = nullptr);
    virtual void EndQuery(int query_id);

    static void SoftMax(vector<IdWeight<float>> &items, float temperature = 1.0f);

protected:
    Random rng_; //random number generator
    map<int, QueryData*> query_data_map_;

protected:
    virtual QueryData* NewQueryData() = 0;

    QueryData* AddOrFindQueryData(int query_id);

    static bool GetSortedTopK(vector<IdWeight<float>> &pool, const SamplingInput &input,
        const StdVocabulary &vocab, int max_queue_len, int thread_count = 1);
    static bool BuildCandidates(vector<IdWeight<float>> &pool, const SamplingInput &input,
        const StdVocabulary &vocab);
};

class NGram
{
public:
    NGram(){}
    NGram(int n, float beta = 0.9, float sw_coeff = 0., std::vector<int> stop_words_ids = {})
        : n(n), beta(beta), sw_coeff(sw_coeff), stop_words_ids(stop_words_ids)
    {
        //assert(sw_coeff >= 0.);
    }

    std::map<int, float> penalize(const std::vector<int> &candidates)
    {
        std::map<int, float> penalty;
        //std::cout << n <<" "<<print_vector(tokens) << std::endl;

        if ((int)tokens.size() < n - 1) {
            return penalty;
        }

        for (const auto &cand : candidates) {
            float remaining = 1;
            float score = 0;

            for (int i = n - 1; i >= 0; i--) {
                std::vector<int> key;
                //if (tokens.size() < i) continue;
                if (i != 0) {
                    key = std::vector<int>(tokens.end() - i, tokens.end());
                    //std::cout << print_vector(key) << std::endl;
                }
                auto ngram_cands = generated_ngrams[i][key];
                std::map<int, int> ngram_count;

                for (const auto &v : ngram_cands) {
                    ngram_count[v]++;
                }
                if (ngram_count.find(cand) == ngram_count.end()) {
                    continue;
                }
               
                int total = 0;
                for (const auto &kv : ngram_count) {
                    total += kv.second;
                }

                float cur_score;
                if (i == 0) {
                    cur_score = static_cast<float>(ngram_count[cand]) / total;
                    score += remaining * cur_score;
                } else {
                    cur_score = static_cast<float>(ngram_count[cand]) / (total + 1);
                    score += remaining * beta * cur_score;
                }

                remaining = remaining - remaining * beta;

            }

            if (std::find(stop_words_ids.begin(), stop_words_ids.end(), cand) != stop_words_ids.end()) {
                penalty[cand] = sw_coeff * score;
            } else {
                penalty[cand] = score;
            }
        }
        //for (const auto& item : penalty){ std::cout << item.first << " " << item.second<<std::endl;}
        return penalty;
    }

    void initialize(std::vector<int> input_ids)
    {
        tokens = input_ids;
        int input_len = (int) input_ids.size();
        generated_ngrams.resize(n);
        if (input_ids.empty()) return;
        for (int idx = 1; idx <= n; idx++) {
            auto &generated_ngram = generated_ngrams[idx - 1];
            for (int i = 0; i < input_len - idx + 1; i++) {
                std::vector<int> ngram(input_ids.begin() + i, input_ids.begin() + i + idx);
                std::vector<int> prev_ngram_tuple(ngram.begin(), ngram.end() - 1);
                generated_ngram[prev_ngram_tuple].push_back(ngram.back());

            }
        }
    }

    void update(int new_token)
    {
        for (int i = 0; i < n; i++)
        {
            std::vector<int> key;
            if ((int)tokens.size() < i) continue;
            if (i != 0) {
                key = std::vector<int>(tokens.end() - i, tokens.end());
            }
            generated_ngrams[i][key].push_back(new_token);
        }
        tokens.push_back(new_token);        
    }

    int get_prefix_length()
    {
        return (int) tokens.size();
    }

private:
    int n;
    std::vector<int> tokens;
    float beta;
    float sw_coeff;
    std::vector<int> stop_words_ids;
    std::vector<std::map<std::vector<int>, std::vector<int>>> generated_ngrams;
};

//greedy, top-k, top-p
class StdSamplingStrategy : public SamplingStrategy
{
public:
    struct Config
    {
        int min_k = 1;
        int max_k = 8;
        float top_p = 0.9f;
        int pool_size = 50;
        int eos_bypassing_max = 2;
    };

public:
    virtual ~StdSamplingStrategy();
    void Clear();

    bool Init(const Config &cfg);

    //virtual bool SetConfig(const JsonObject &jobj, const JsonDoc &jdoc) override;

    //the config should be in JSON format
    virtual void BeginQuery(int query_id, const QueryOptions &opt, const string &config_str,
        const JsonParser *jparser = nullptr) override;

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) override;

protected:
    struct StdQueryData : public QueryData
    {
        Config config;
        int eos_bypassing_count = 0;
    };

protected:
    Config config_;

protected:
    virtual QueryData* NewQueryData() override;
};

//FSD: Frustratingly Simple Decoding (https://arxiv.org/abs/2305.12675)
class FsdSamplingStrategy : public SamplingStrategy
{
public:
    struct Config
    {
        int pool_size = 50;
        int k = 6;
        float alpha = 0.5f;
        int n = 3;
        float beta = 0.9f;
        int eos_bypassing_max = 2;
    };

public:
    virtual ~FsdSamplingStrategy();
    void Clear();

    bool Init(const Config &cfg);

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) override;


protected:
    Config config_;

protected:
    struct FsdQueryData : public QueryData
    {
        Config config;
        NGram ngram;
        int eos_bypassing_count = 0;
        bool started = false;
    };

protected:
    virtual QueryData* NewQueryData() override;
};


//RandomizedFSD: a stochastic version of Frustratingly Simple Decoding (https://arxiv.org/abs/2305.12675)
class RandomizedFSDSamplingStrategy : public SamplingStrategy
{
public:
    struct Config
    {
        int k = 6;
        float alpha = 0.5f;
        int n = 3;
        float beta = 0.9f;
        int eos_bypassing_max = 2;

        int min_k = 1;
        int max_k = 8;
        float top_p = 0.93f;
        int pool_size = 50;
    };

public:
    virtual ~RandomizedFSDSamplingStrategy();
    void Clear();

    bool Init(const Config &cfg);

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) override;


protected:
    Config config_;

protected:
    struct RandomizedFSDQueryData : public QueryData
    {
        Config config;
        NGram ngram;
        int eos_bypassing_count = 0;
        bool started = false;
        int new_tokens = 0;
    };

protected:
    virtual QueryData* NewQueryData() override;
};


// MinP Sampling
// Sets a minimum base probability threshold for token selection (default: 0.05).
// The parameter *min_p* represents the minimum probability for a token to be considered, relative to the probability of the most likely token.
class MinPSamplingStrategy : public SamplingStrategy
{
public:
    struct Config
    {
        int pool_size = 50;
        float min_p = 0.05f;
        int eos_bypassing_max = 2;
    };

public:
    virtual ~MinPSamplingStrategy();
    void Clear();

    bool Init(const Config &cfg);

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) override;


protected:
    Config config_;

protected:
    struct MinPQueryData : public QueryData
    {
        Config config;
        int eos_bypassing_count = 0;
    };

protected:
    virtual QueryData* NewQueryData() override;
};

// Tail Free Sampling (TFS)
// Enable tail free sampling with parameter z
// Tail free sampling (TFS) is a text generation technique that aims to reduce the impact of less likely tokens, which may be less relevant, less coherent, or nonsensical, on the output. 
//TFS filters out logits based on the second derivative of their probabilities. Adding tokens is stopped after the sum of the second derivatives reaches the parameter z.
class TFSSamplingStrategy : public SamplingStrategy
{
public:
    struct Config
    {
        int pool_size = 50;
        float z = 0.95f;
        int eos_bypassing_max = 2;
    };

public:
    virtual ~TFSSamplingStrategy();
    void Clear();

    bool Init(const Config &cfg);

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) override;


protected:
    Config config_;

protected:
    struct TFSQueryData : public QueryData
    {
        Config config;
        int eos_bypassing_count = 0;
    };

protected:
    virtual QueryData* NewQueryData() override;
};


//Locally Typical Sampling
//`p`: Enable locally typical sampling with parameter p.
//Locally typical sampling promotes the generation of contextually coherent and diverse text by sampling tokens that are typical or expected based on the surrounding context. 
class TypicalSamplingStrategy : public SamplingStrategy
{
public:
    struct Config
    {
        int pool_size = 50;
        float p = 0.95f;
        int eos_bypassing_max = 2;
    };

public:
    virtual ~TypicalSamplingStrategy();
    void Clear();

    bool Init(const Config &cfg);

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) override;


protected:
    Config config_;

protected:
    struct TypicalQueryData : public QueryData
    {
        Config config;
        int eos_bypassing_count = 0;
    };

protected:
    virtual QueryData* NewQueryData() override;
};

//Mirostat Sampling v2.0
//`eta`: Set the Mirostat learning rate, parameter eta (default: 0.1).
//`tau`: Set the Mirostat target entropy, parameter tau (default: 5.0).
//Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. It aims to strike a balance between coherence and diversity, avoiding low-quality output caused by excessive repetition (boredom traps) or incoherence (confusion traps).
class MirostatSamplingStrategy : public SamplingStrategy
{
public:
    struct Config
    {
        int pool_size = 50;
        float eta = 0.1f;
        float tau = 5.0f;
        int eos_bypassing_max = 2;
    };

public:
    virtual ~MirostatSamplingStrategy();
    void Clear();

    bool Init(const Config &cfg);

    virtual bool ChooseTokens(SamplingOutput &output,
        const SamplingInput &input, const StdVocabulary &vocab,
        SamplingStrategyId strategy_id = SamplingStrategyId::Auto,
        int thread_count = 1) override;


protected:
    Config config_;

protected:
    struct MirostatQueryData : public QueryData
    {
        Config config;
        float mu = 0.f;
        int eos_bypassing_count = 0;
    };

protected:
    virtual QueryData* NewQueryData() override;
};

TRANSFORMER_END
INFER_FLOW_END
