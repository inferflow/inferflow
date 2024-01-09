#pragma once

#include "tensor/tensor_common.h"
#include "transformer_types.h"
#include "model.h"
#include "query_state_table.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::string;
using std::wstring;
using std::vector;
using std::map;
using sslib::IdWeight;
using sslib::PtrVector;

class InferenceConfig
{
public:
    struct DebugOptions
    {
        bool is_study_mode = false;
        bool show_tensors = false;
        bool enable_perf_stat = true;
    };

public:
    vector<ModelSpec> models;
    string data_dir;
    //vector<int> devices;
    int max_concurrent_queries = 10;

    vector<vector<int>> device_groups;
    int encoder_cpu_layer_count = 0;
    int decoder_cpu_layer_count = 0;
    int cpu_threads = 8;

    map<string, string> prompt_templates;

    bool return_output_tensors = false;
    DebugOptions debug;
};

struct InferenceConfigEx
{
    bool is_cpu_tensor_row_major = false;
    bool is_gpu_tensor_row_major = false;
    MatrixMulAlg matrix_mul_alg = MatrixMulAlg::Cublas;
    VectorMatrixMulAlg gemv_alg = VectorMatrixMulAlg::Cublas;
    bool enable_full_quant_gemv = true;
};

struct LlmQuery
{
    wstring system_prompt;
    wstring text;
    wstring response_prefix; //prefix of the response
    wstring encoder_input_template;
    wstring decoder_input_template;

    void Clear()
    {
        system_prompt.clear();
        text.clear();
        //q_prefix.clear();
        //q_suffix.clear();
        response_prefix.clear();
        encoder_input_template.clear();
        decoder_input_template.clear();
    }
};

struct EncoderInput
{
    vector<int> prefix_tokens;
    string core_text;
    vector<int> suffix_tokens;

    void Clear()
    {
        prefix_tokens.clear();
        core_text.clear();
        suffix_tokens.clear();
    }
};

struct DecoderPrefixSection
{
    bool is_text = true;
    int token_id = -1;
    string text;
};

struct DecoderPrefix
{
    vector<DecoderPrefixSection> sections;
    //vector<int> prefix_tokens;
    //string prompt_text;
    string res_prefix;

    void Clear()
    {
        sections.clear();
        //prefix_tokens.clear();
        //prompt_text.clear();
        res_prefix.clear();
    }
};

struct InferencePerfStat
{
    std::map<uint32_t, float> time_map;

    void Clear()
    {
        time_map.clear();
    }
};

struct QueryInferenceInput
{
    int query_id = 0;
    vector<int> tokens;
    int prefix_len = 0;
    int sampling_strategy = 0;
};

struct InferenceInput
{
    PtrVector<QueryInferenceInput> items; 
};

struct QueryInferenceResult
{
    int query_id = 0;
    int prefix_len = 0;
    vector<IdWeight<float>> next_tokens;
    HostTensor output_tensor;
};

struct QueryNextToken
{
    int id = 0; //token id
    bool is_end = false;
};

struct InferenceResult
{
    PtrVector<QueryInferenceResult> items;
    InferencePerfStat perf_stat;
    InferencePerfStat perf_stat_pre;
};

struct QueryProcInput
{
    int query_id = 0;
    int proc_id = 0;
    int prefix_len = 0;
    int token_num = 0;
    int start_row = 0;
    int sampling_strategy = 0;
    const QueryState *state = nullptr;
};

struct LayerRange
{
    int layer_num = 0;
    int start = 0;
    int end = 0;

    LayerRange(int p_num = 0, int p_start = 0, int p_end = 0)
    {
        layer_num = p_num;
        start = p_start;
        end = p_end;
    }
};

TRANSFORMER_END
INFER_FLOW_END
