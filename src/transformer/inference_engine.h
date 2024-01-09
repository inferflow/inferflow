#pragma once

#include <mutex>
#include "sslib/config_data.h"
#include "sslib/json.h"
#include "common/text_tokenizer.h"
#include "tensor/host_float_buffer.h"
#include "query_state_table.h"
#include "model_reader.h"
#include "network_builder.h"
#include "decoding_strategies.h"
#include "cpu_inference_worker.h"
#if defined(USE_CUDA)
#   include "inference_worker.h"
#endif //USE_CUDA

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::string;
using std::vector;
using std::map;
using std::ostream;
using std::mutex;
using std::ofstream;
using sslib::IdWeight;
using sslib::PtrVector;
using sslib::StrLessNoCase;
using sslib::ConfigData;
using sslib::JsonParser;

class InferenceEngine
{
public:
    InferenceEngine();
    virtual ~InferenceEngine();
    void Clear();

    bool Init(const InferenceConfig &cfg);

    //return:
    //  > 0: Query-id
    //  = 0: Busy
    //  < 0: Error occurred
    int AddQuery(const EncoderInput &encoder_input,
        const DecoderPrefix &decoder_prefix,
        const SamplingStrategy::QueryOptions &query_options,
        TokenizationAlg alg = TokenizationAlg::Auto);
    int AddQuery(vector<int> &encoder_input_tokens,
        vector<int> &decoder_prefix_tokens,
        const EncoderInput &encoder_input,
        const DecoderPrefix &decoder_prefix,
        const SamplingStrategy::QueryOptions &query_options,
        TokenizationAlg alg = TokenizationAlg::Auto);
    int AddQuery(const vector<int> &tokens,
        const SamplingStrategy::QueryOptions &query_options);
    int AddQuery(const vector<int> &encoder_input_tokens,
        const vector<int> &decoder_prefix_tokens,
        const SamplingStrategy::QueryOptions &query_options);

    int QueryCount() const;

    bool Infer(InferenceResult &res);

    bool RemoveQuery(int query_id);
    bool CommitInferenceResult(const map<int, QueryNextToken> &query_map);

    const TransformerContext& context() const {
        return context_;
    }

    bool Tokenize(vector<int> &tokens, const string &text, bool add_bos,
        TokenizationAlg alg = TokenizationAlg::Auto);

    std::string TokenIdToStr(int id, bool enable_decoding = true) const;
    int TokenStrToId(const std::string &str) const;
    std::string OutputTokenIdToStr(int id, bool enable_decoding = true) const;
    int OutputTokenStrToId(const std::string &str) const;

    TransformerContext& context() {
        return context_;
    }

    const StdVocabulary& vocabulary() const {
        return model_.vocabulary;
    }

    const TextTokenizer* tokenizer() const {
        return &tokenizer_;
    }

    int default_device_id() const {
        return default_device_id_;
    }

    string GetEncoderInputTemplate(const string &model = "") const;
    string GetPromptTemplate(const string &model = "") const;

    void BuildEncoderInput(EncoderInput &obj, const LlmQuery &query);
    void BuildDecoderInput(DecoderPrefix &obj, const LlmQuery &query);
    string BuildPrompt(const string &prompt_template, const string &question,
        const string &answer_prefix, const string &system_prompt = "");
    string BuildPrompt(const string &prompt_template, const LlmQuery &query);

    static bool ParseQueryJson(LlmQuery &query, const wstring &query_str,
        const JsonParser &json_parser);
    static bool ParseQueryJson(LlmQuery &query, const JsonObject &jobj,
        const JsonDoc &jdoc);

    SamplingStrategyId GetSamplingStrategyId(const string &str = "") const;

    void PrintPerfStat(ostream &strm, const InferencePerfStat &perf_stat) const;

    string Version() const {
        return version_;
    }

    static bool LoadConfig(InferenceConfig &config, const string &config_path,
        const string &section, const string &data_root_dir = "");
    static bool LoadModelSpec(ModelSpec &spec, const ConfigData &cfg_data,
        const string &section, const ElementTypeMap &element_type_map,
        const MultiGpuStrategyMap &multi_gpu_strategy_map,
        const DecodingStrategyMap &decoding_strategy_map,
        const JsonParser &jparser);
    static bool LoadDeviceGroups(vector<vector<int>> &device_groups,
        const ConfigData &cfg_data, const string &section, const string &key);
    static bool LoadPromptTemplates(map<string, string> &prompt_templates,
        const ConfigData &cfg_data, const string &section);

protected:
    string version_;
    InferenceConfig config_;
    InferenceConfigEx config_ex_;
    TransformerModel model_;
    TransformerContext context_;
    ofstream tensor_writer_;
    bool is_cpu_only_ = false;
    bool is_cpu_involved_ = false;

    JsonParser json_parser_;
    TextTokenizer tokenizer_;

    ModelFileFormatMap model_file_format_map_;
    NetworkStructureMap network_structure_map_;

#if defined(USE_CUDA)
    //for the current call of evaluate
    DeviceMemoryHeap local_device_heap_;
    //BlockedAllocator<DeviceTensor> local_device_tensor_heap_;
#endif //USE_CUDA

    ModelReader model_reader_;
    NetworkBuilder network_builder_;

    //CublasEngine cublas_engine_;

    int default_device_id_ = 0;

    QueryStateTable query_state_table_;
    std::mutex query_state_lock_;

    ElementTypeMap element_type_map_;

    DecodingStrategies decoding_strategies_;

protected:
    struct LocalInput
    {
        vector<QueryProcInput> query_list;
        vector<int> tokens;
#       if defined(USE_CUDA)
        const DeviceTensor *input_tensor = nullptr;
#       endif //USE_CUDA
        bool is_encoder = false;

        void Clear()
        {
            query_list.clear();
            tokens.clear();
#           if defined(USE_CUDA)
            input_tensor = nullptr;
#           endif //USE_CUDA
            is_encoder = false;
        }
    };

    struct LocalOutput
    {
#if defined(USE_CUDA)
        const DeviceTensor *layer_output = nullptr;
#endif //USE_CUDA
        int device_id = 0;
    };

    enum class KeyId
    {
        UnknownKey = 0,
        SystemPrompt,
        QueryText,
        QueryTextWithPreSpace,
        ResPrefix,
        Bos,
        Eos,
        UserToken,
        AssistantToken
    };

#if defined(USE_CUDA)
    struct GpuWorkerGroup
    {
        GpuInfGlobalData global_data; //GPU inference global data
        PtrVector<GpuInferenceWorker> workers;
    };
#endif //USE_CUDA

protected:
    CpuInferenceWorker *cpu_worker_;

    ggml_context *cpu_ctx_ = nullptr;
    ggml_cgraph gf_ = {};

#if defined(USE_CUDA)
    PtrVector<GpuWorkerGroup> gpu_worker_groups_;
#endif //USE_CUDA

protected:
    map<string, KeyId> key_map_;

protected:
    bool Infer_CpuOnly(InferenceResult &res, int cpu_threads);

    bool GetLocalInput(LocalInput &input, vector<int> &saturated_queries);

    static void HandleSaturatedQueries(InferenceResult &res,
        const vector<int> &saturated_queries,
        const TransformerModel &model);

#if defined(USE_CUDA)
    bool Infer_Gpu(InferenceResult &res);

    bool GetEmbdTensor(DeviceTensor &embd_tensor, const LocalInput &input, bool is_encoder);

    bool BuildLocalOutput(LocalOutput &output, InferencePerfStat &perf_stat,
        const LocalInput &input);

    bool Infer_Std(LocalOutput &output, InferencePerfStat &perf_stat,
        const LocalInput &input);
    bool Infer_TensorParallelism(LocalOutput &output, InferencePerfStat &perf_stat,
        const LocalInput &input);
#endif //USE_CUDA

    static bool LoadPromptTemplate(string &template_str, const ConfigData &cfg_data,
        const string &section);

    SamplingStrategy* GetSamplingStrategy(SamplingStrategyId id);

    bool BuildCpuLayers(const ModelSpec &model_spec, int encoder_layer_count,
        int decoder_layer_count);

    bool SampleTokensCpu(vector<IdWeight<float>> &tokens, ggml_tensor *logits,
        int row, int query_id, const vector<int> &prefix_tokens,
        const vector<int> &cur_tokens, SamplingStrategyId strategy_id);

#if defined(USE_CUDA)
    bool BuildGpuLayers(const ModelSpec &model_spec, const ModelPartition &model_partition);
    bool CreateGpuWorkers(const ModelSpec &model_spec, const ModelPartition &model_partition);

    bool SampleTokens(vector<IdWeight<float>> &next_tokens, const DeviceTensor &logits,
        int row, int query_id, const vector<int> &prefix_tokens,
        const vector<int> &cur_tokens, SamplingStrategyId strategy_id);
    bool EncoderSampleTokens(vector<IdWeight<float>> &output_tokens, const DeviceTensor &logits,
        int row, int query_id, SamplingStrategyId strategy_id);
#endif //USE_CUDA

    static int GetDefaultDevice(const ModelSpec &spec);

    void InitKeyMap();

    void LockQueryStateTable() const;
    void UnLockQueryStateTable() const;

    void ClearResult(InferenceResult &res)
    {
        res.items.Clear(true);
        res.perf_stat.Clear();
    }
};

TRANSFORMER_END
INFER_FLOW_END
