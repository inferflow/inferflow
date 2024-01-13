#pragma once

#include <vector>
#include <map>
#include "../namespace.inc"
#if defined(USE_CUDA)
#   include "tensor/device_tensor.h"
#   include "tensor/device_memory_heap.h"
#endif
#include "network_structure.h"
#include "model.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::vector;
using std::map;

struct QueryState
{
    static const int MAX_PROC_ID = 15;

    int query_id = 0;
    int proc_id = 0;
    vector<int> encoder_input_tokens;
    vector<int> decoder_tokens;
    vector<int> prefix_tokens;
    int prefix_len = 0;
    int sampling_strategy = 0;

    bool is_encoder_only = false;
    int next_net = 0; //0: encoder, 1: decoder
    int next_layer = 0;

#if defined(USE_CUDA)
    DeviceTensor encoder_output, decoder_layer_output;
    DeviceMemoryHeap heap;
#endif

    QueryState(int qid)
    {
        query_id = qid;
    }

    virtual ~QueryState();
};

class QueryStateTable
{
public:
    QueryStateTable();
    virtual ~QueryStateTable();
    void Clear();

    int Size() const;
    void Get(map<int, const QueryState*> &query_map, NetworkType net_type,
        int max_context_len, int max_query_num = -1, int max_token_num = -1) const;

    int Add(const vector<int> &encoder_input_tokens,
        const vector<int> &decoder_prefix_tokens,
        const ModelSpec &model_spec, int sampling_strategy = 0);

#if defined(USE_CUDA)
    bool UpdateEncoderEnd(int query_id, const DeviceTensor &output_tensor,
        int start_row, int row_num);
#endif

    bool Update(int query_id, int next_token_id, bool is_end);
    bool Remove(int query_id);

    const QueryState* GetQueryState(int query_id) const;

protected:
    map<int, QueryState*> query_map_;
    int next_query_id_ = 1;
    //static const int MAX_QUERY_ID = 1000 * 10000;
    static const int MAX_QUERY_ID = 10000;

protected:
    int FindFreeProcId() const;
};

TRANSFORMER_END
INFER_FLOW_END
