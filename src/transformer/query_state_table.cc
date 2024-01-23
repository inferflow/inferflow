#include "query_state_table.h"
#include "sslib/log.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace sslib;

QueryState::~QueryState()
{
}

QueryStateTable::QueryStateTable()
{
}

QueryStateTable::~QueryStateTable()
{
    Clear();
}

void QueryStateTable::Clear()
{
    for (auto iter = query_map_.begin(); iter != query_map_.end(); iter++)
    {
        if (iter->second != nullptr) {
            delete iter->second;
        }
        iter->second = nullptr;
    }

    query_map_.clear();
    next_query_id_ = 1;
}

int QueryStateTable::Size() const
{
    return (int)query_map_.size();
}

void QueryStateTable::Get(map<int, const QueryState*> &query_map, NetworkType net_type,
    int max_context_len, int max_query_num, int max_token_num) const
{
    query_map.clear();

    bool is_encoder_only = NetworkStructure::IsEncoderOnlyTransformer(net_type);
    bool is_decoder_only = NetworkStructure::IsDecoderOnlyTransformer(net_type);

    int next_net = -1;
    int token_num = 0;
    for (auto iter = query_map_.begin(); iter != query_map_.end(); iter++)
    {
        const QueryState *query_state = iter->second;

        bool is_compatible = next_net < 0 || is_encoder_only || is_decoder_only
            || next_net == query_state->next_net;
        if (!is_compatible) {
            continue;
        }

        bool is_encoder = !is_decoder_only && next_net == 0;
        (void)max_context_len;
        /*if (!is_encoder && query_state->prefix_len >= max_context_len)
        {
            QueryState *new_state = iter->second;
            //int moveing_start = 2 * max_context_len / 3;
            int moveing_start = max_context_len / 3;
            new_state->decoder_tokens.insert(new_state->decoder_tokens.begin(),
                new_state->prefix_tokens.begin() + moveing_start,
                new_state->prefix_tokens.end());
            new_state->prefix_len = 0; //max_context_len / 3;
            new_state->prefix_tokens.resize(new_state->prefix_len);
        }*/

        query_map[iter->first] = iter->second;
        if (next_net < 0) {
            next_net = query_state->next_net;
        }

        if (is_encoder) {
            token_num += (int)query_state->encoder_input_tokens.size();
        }
        else {
            token_num += (int)query_state->decoder_tokens.size();
        }

        if (max_token_num > 0 && token_num >= max_token_num) {
            break;
        }
        if (max_query_num > 0 && (int)query_map.size() >= max_query_num) {
            break;
        }
    }
}

int QueryStateTable::Add(const vector<int> &encoder_input_tokens,
    const vector<int> &decoder_prefix_tokens,
    const ModelSpec &model_spec, int sampling_strategy,
    int max_output_tokens)
{
    const auto &hparams = model_spec.hyper_params;

    int proc_id = FindFreeProcId();
    if (proc_id < 0 || proc_id > QueryState::MAX_PROC_ID) {
        return -1;
    }

    int query_id = next_query_id_;
    QueryState *state = new QueryState(query_id);
    state->proc_id = proc_id;
    state->encoder_input_tokens = encoder_input_tokens;
    state->decoder_tokens = decoder_prefix_tokens;
    state->prefix_len = 0;
    state->initial_prefix_len = (int)decoder_prefix_tokens.size();
    state->sampling_strategy = sampling_strategy;
    state->max_output_tokens = max_output_tokens;
    state->is_encoder_only = hparams.decoder_layers <= 0;

#if defined(USE_CUDA)
    if (hparams.encoder_layers > 0)
    {
        state->encoder_output.New(model_spec.device_weight_data_type,
            hparams.embd_dims, model_spec.max_context_len);
    }
    if (hparams.decoder_layers > 0)
    {
        state->decoder_layer_output.New(model_spec.device_weight_data_type,
            hparams.embd_dims, model_spec.max_context_len);
    }
#endif //USE_CUDA

    query_map_[query_id] = state;

    next_query_id_ = next_query_id_ < MAX_QUERY_ID ? (next_query_id_ + 1) : 1;
    return query_id;
}

#if defined(USE_CUDA)
bool QueryStateTable::UpdateEncoderEnd(int query_id, const DeviceTensor &output_tensor,
    int start_row, int row_num)
{
    bool ret = true;
    auto iter = query_map_.find(query_id);
    if (iter == query_map_.end()) {
        return false;
    }

    QueryState *state = iter->second;
    if (state == nullptr) {
        return false;
    }

    state->next_net = 1;
    int bytes = (int)TensorCommon::ByteCount(output_tensor.data_type, output_tensor.ne[0] * row_num);
    state->encoder_output.data_type = output_tensor.data_type;
    state->encoder_output.SetStructure(output_tensor.ne[0], row_num);
    int start_bytes = (int)TensorCommon::ByteCount(output_tensor.data_type, output_tensor.ne[0] * start_row);
    const void *source_data = (const uint8_t*)output_tensor.data + start_bytes;
    CudaUtil::DeviceToDeviceMemcpy(state->encoder_output.data, source_data, bytes);

    state->prefix_tokens.clear();
    state->prefix_len = 0;

    if (state->is_encoder_only) {
        ret = Remove(query_id);
    }
    return ret;
}
#endif //USE_CUDA

bool QueryStateTable::Update(int query_id, int next_token_id, bool is_end)
{
    bool ret = true;
    auto iter = query_map_.find(query_id);
    if (iter == query_map_.end()) {
        return false;
    }

    QueryState *state = iter->second;
    if (state == nullptr) {
        return false;
    }

    state->prefix_tokens.insert(state->prefix_tokens.end(),
        state->decoder_tokens.begin(), state->decoder_tokens.end());
    state->prefix_len = (int)state->prefix_tokens.size();
    state->decoder_tokens.clear();
    state->decoder_tokens.push_back(next_token_id);

    if (is_end) {
        ret = Remove(query_id);
    }
    return ret;
}

bool QueryStateTable::Remove(int query_id)
{
    auto iter = query_map_.find(query_id);
    if (iter == query_map_.end()) {
        return false;
    }

    if (iter->second != nullptr)
    {
        delete iter->second;
        iter->second = nullptr;
    }

    query_map_.erase(iter);
    return true;
}

const QueryState* QueryStateTable::GetQueryState(int query_id) const
{
    auto iter = query_map_.find(query_id);
    return iter != query_map_.end() ? iter->second : nullptr;
}

int QueryStateTable::FindFreeProcId() const
{
    vector<int> freq_list(1 + QueryState::MAX_PROC_ID, 0);
    for (auto iter = query_map_.begin(); iter != query_map_.end(); iter++)
    {
        freq_list[iter->second->proc_id]++;
    }

    for (int idx = 0; idx < (int)freq_list.size(); idx++)
    {
        if (freq_list[idx] == 0) {
            return idx;
        }
    }

    return -1;
}

TRANSFORMER_END
INFER_FLOW_END

