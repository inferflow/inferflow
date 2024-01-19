#include "decoding_strategies.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

DecodingStrategies::DecodingStrategies()
{
}

DecodingStrategies::~DecodingStrategies()
{
    Clear();
}

void DecodingStrategies::Clear()
{
    strategy_list_.Clear(true);
    strategy_map_.clear();
}

bool DecodingStrategies::Init()
{
    DecodingStrategies::InitStrategyMap(name_to_id_);

    auto *std_strategy = new StdSamplingStrategy();
    StdSamplingStrategy::Config std_config;
    std_config.eos_bypassing_max = 0;
    std_strategy->Init(std_config);

    SamplingStrategy *strategy = std_strategy;
    strategy_list_.push_back(strategy);
    strategy_map_[SamplingStrategyId::StdSampling] = strategy;
    strategy_map_[SamplingStrategyId::Greedy] = strategy;
    strategy_map_[SamplingStrategyId::TopK] = strategy;
    strategy_map_[SamplingStrategyId::TopP] = strategy;

    strategy = new FsdSamplingStrategy();
    strategy_list_.push_back(strategy);
    strategy_map_[SamplingStrategyId::FSD] = strategy;

    strategy = new RandomizedFSDSamplingStrategy();
    strategy_list_.push_back(strategy);
    strategy_map_[SamplingStrategyId::RandomizedFSD] = strategy;

    strategy = new MinPSamplingStrategy();
    strategy_list_.push_back(strategy);
    strategy_map_[SamplingStrategyId::MinP] = strategy;

    strategy = new TFSSamplingStrategy();
    strategy_list_.push_back(strategy);
    strategy_map_[SamplingStrategyId::TFS] = strategy;

    strategy = new TypicalSamplingStrategy();
    strategy_list_.push_back(strategy);
    strategy_map_[SamplingStrategyId::Typical] = strategy;

    strategy = new MirostatSamplingStrategy();
    strategy_list_.push_back(strategy);
    strategy_map_[SamplingStrategyId::Mirostat] = strategy;

    return true;
}

SamplingStrategy* DecodingStrategies::Get(SamplingStrategyId id)
{
    if (id == DecodingStrategyId::Auto) {
        id = DecodingStrategyId::StdSampling;
    }
    auto iter = strategy_map_.find(id);
    return iter == strategy_map_.end() ? nullptr : iter->second;
}

const SamplingStrategy* DecodingStrategies::Get(SamplingStrategyId id) const
{
    if (id == DecodingStrategyId::Auto) {
        id = DecodingStrategyId::StdSampling;
    }
    auto iter = strategy_map_.find(id);
    return iter == strategy_map_.end() ? nullptr : iter->second;
}

SamplingStrategyId DecodingStrategies::GetId(const string &strategy_name) const
{
    auto iter = name_to_id_.find(strategy_name);
    return iter != name_to_id_.end() ? iter->second : SamplingStrategyId::Auto;
}

//static
void DecodingStrategies::InitStrategyMap(DecodingStrategyMap &the_map)
{
    the_map.clear();

    the_map["sample.std"] = SamplingStrategyId::StdSampling;
    the_map["greedy"] = SamplingStrategyId::Greedy;
    the_map["sample.greedy"] = SamplingStrategyId::Greedy;
    the_map["top_k"] = SamplingStrategyId::TopK;
    the_map["sample.top_k"] = SamplingStrategyId::TopK;
    the_map["top_p"] = SamplingStrategyId::TopP;
    the_map["sample.top_p"] = SamplingStrategyId::TopP;
    the_map["fsd"] = SamplingStrategyId::FSD;
    the_map["sample.fsd"] = SamplingStrategyId::FSD;
    the_map["random_fsd"] = SamplingStrategyId::RandomizedFSD;
    the_map["sample.random_fsd"] = SamplingStrategyId::RandomizedFSD;

    the_map["min_p"] = SamplingStrategyId::MinP;
    the_map["tfs"] = SamplingStrategyId::TFS;
    the_map["typical"] = SamplingStrategyId::Typical;
    the_map["mirostat"] = SamplingStrategyId::Mirostat;
}

TRANSFORMER_END
INFER_FLOW_END
