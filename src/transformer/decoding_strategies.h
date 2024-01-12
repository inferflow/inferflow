#pragma once

#include "sslib/string.h"
#include "sslib/vector_ex.h"
#include "sampling_strategy.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using sslib::StrLessNoCase;
using sslib::PtrVector;

typedef map<string, DecodingStrategyId, StrLessNoCase> DecodingStrategyMap;

class DecodingStrategies
{
public:
    DecodingStrategies();
    virtual ~DecodingStrategies();
    void Clear();

    bool Init();

    const SamplingStrategy* Get(DecodingStrategyId id) const;
    SamplingStrategy* Get(DecodingStrategyId id);
    DecodingStrategyId GetId(const string &strategy_name) const;

    static void InitStrategyMap(DecodingStrategyMap &the_map);

protected:
    PtrVector<SamplingStrategy> strategy_list_;
    map<SamplingStrategyId, SamplingStrategy*> strategy_map_;
    DecodingStrategyMap name_to_id_;
};

TRANSFORMER_END
INFER_FLOW_END
