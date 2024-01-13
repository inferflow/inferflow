#pragma once

#include <string>
#include "sslib/string.h"
#include "sslib/vector_ex.h"
#include "ggml/ggml.h"
#include "common/std_vocabulary.h"
#include "common/text_tokenizer.h"
#include "tensor/host_float_buffer.h"
#include "network_structure.h"
#include "tensor/host_tensor_opr.h"
#if defined(USE_CUDA)
#   include "tensor/tensor_opr.h"
#endif

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::string;
using std::vector;
using std::map;
using sslib::PtrVector;
using sslib::StrLessNoCase;

//serialization format
enum class ModelFileFormat
{
    Unknown,
    Std,
    Pickle,
    Safetensors,
    GGML, //GGML format (https://github.com/ggerganov/llama.cpp)
    GGUF, //GGUF format (https://github.com/ggerganov/llama.cpp)
    LLAMA2_C //llama2.c format (https://github.com/karpathy/llama2.c)
};

typedef map<string, ModelFileFormat, StrLessNoCase> ModelFileFormatMap;
typedef map<string, NetworkType, StrLessNoCase> NetworkStructureMap;
typedef map<string, MultiGpuStrategy, StrLessNoCase> MultiGpuStrategyMap;

struct ModelHyperParams
{
    int vocab_size = 0;         //vocabulary size
    int output_vocab_size = 0;  //output vocabulary size
    int embd_dims = 4096;       //token embedding dimensions

    //encoder
    int encoder_layers = 0;
    int encoder_heads = 0;
    int encoder_kv_heads = 0;

    //decoder
    int decoder_layers = 0;
    int decoder_heads = 0;
    int hidden_dim = 0;
    int decoder_kv_heads = 0;

    int training_context_len = -1;
};

struct ModelSpec
{
    string sid; //model id or standard name
    ModelHyperParams hyper_params;

    string dir; //model directory
    string spec_file;               //name of the model spec file
    vector<string> model_files;     //names of the model files
    string config_file;             //name of the model configuration file
    vector<string> tokenizer_files; //tokenizer file names
    string token_remap_file;        //token id remap file
    TokenizationAlg tokenization_algorithm = TokenizationAlg::BPE;
    string generation_config_file;
    int token_bytes_mapping = 0;
    ModelFileFormat model_file_format = ModelFileFormat::Unknown;
    NetworkType network_structure = NetworkType::General;

    TensorNormAlg norm_alg = TensorNormAlg::STD;
    ActivationFn activation_fn = ActivationFn::SILU;
    PositionEmbeddingAlg pos_embedding_alg = PositionEmbeddingAlg::ROPE;
    float rope_theta = 10000.0f;
    map<string, string> tensor_name_map;
    map<string, string> tensor_name_pre_map;

    int qk_column_order = 0;
    int qkv_format = 0; //0: first split by head, then Q+K+V; 1: Q+K+V
    bool transform_qk = false;
    bool normalize_lm_head = false;
    bool is_parallel_attn = false;
    bool mlp_attn_share_input = false;
    string tensor_name_prefix;

    bool use_self_attn_pre_norm = true;

    //special tokens
    string unk_token, pad_token, bos_token, eos_token, mask_token;

    string decoding_strategy; //the default decoding strategy
    string decoding_strategy_config;

    //the default input templates for the encoder and decoder
    //(users can use new templates to replace the default one) 
    string encoder_input_template;
    string decoder_input_template; //also called prompt_template

    bool be_host_embeddings = true;
    ElementType device_weight_data_type = ElementType::F16;
    ElementType device_kv_cache_data_type = ElementType::F16;
    ElementType host_weight_data_type = ElementType::F16;
    float delta_tensor_ratio = 0;
    int tensor_quant_threshold = 2000 * 2000;
    int host_kv_cache_percent = 0; //[0, 100]
    bool has_cross_attn_kv_cache = true;

    static const int DEFAULT_MAX_CONTEXT_LEN = 1024;
    int max_context_len = -1;
    int max_input_len = 1024;

    MultiGpuStrategy multi_gpu_strategy = MultiGpuStrategy::BY_LAYER;
    vector<vector<int>> device_groups;
    int encoder_cpu_layer_count = -1;
    int decoder_cpu_layer_count = -1;

    bool is_eager_device_building = false;
};

struct NetworkStat
{
    float all_memory_usage = 0; //in GB
    int encoder_layer_num = 0;
    float encoder_layer_size[3]; //in GB, for all, self-attn and ffn
    int decoder_layer_num = 0;
    float decoder_layer_size[4]; //in GB, for all, self-attn, cross-attn, and ffn
    float embedding_size = 0; //in GB
    float pre_layer_size = 0; //in GB
    float post_layer_size = 0; //in GB

    map<ElementType, int> tensor_num_by_element_type;
};

#if defined(USE_CUDA)
class StdDeviceNetwork
{
public:
    struct AtomicLayer
    {
        DeviceTensor *weight = nullptr;
        DeviceTensor *bias = nullptr;
    };

    struct SimpleLayer
    {
        AtomicLayer dense;
        AtomicLayer pre_norm, post_norm;
    };

    struct SubLayer
    {
        map<LayerTensorId, DeviceTensorEx*> tensor_map;
    };

    struct AttentionLayer : public SubLayer
    {
        // normalization
        DeviceTensorEx pre_norm, pre_norm_b;
        DeviceTensorEx post_norm, post_norm_b;

        // attention
        DeviceTensorEx qkv, qkv_b; // (wq, wk, wv)
        DeviceTensorEx wq, wq_b;
        DeviceTensorEx wk, wk_b;
        DeviceTensorEx wv, wv_b;
        DeviceTensorEx wo, wo_b;
    };

    struct FeedForwardLayer : public SubLayer
    {
        // normalization
        DeviceTensorEx pre_norm, pre_norm_b;
        DeviceTensorEx post_norm, post_norm_b;

        // ff
        DeviceTensorEx w1, w1_b;
        DeviceTensorEx w2, w2_b;
        DeviceTensorEx w3, w3_b;
        DeviceTensorEx w1n3, w1n3_b; // (w1, w3)
    };

    struct EncoderLayer
    {
        AttentionLayer self_attn;
        FeedForwardLayer ffn;
    };

    struct DecoderLayer : public EncoderLayer
    {
        AttentionLayer cross_attn;
    };

public:
    DeviceTensor *encoder_embeddings = nullptr;
    DeviceTensor *decoder_embeddings = nullptr;

    DeviceTensor *encoder_pos_embeddings = nullptr;
    DeviceTensor *encoder_token_type_embeddings = nullptr;

    DeviceTensor *encoder_input_norm = nullptr;
    DeviceTensor *encoder_input_norm_b = nullptr;
    DeviceTensor *decoder_input_norm = nullptr;
    DeviceTensor *decoder_input_norm_b = nullptr;

    DeviceTensor *encoder_output_norm = nullptr;
    DeviceTensor *encoder_output_norm_b = nullptr;
    DeviceTensor *encoder_output_post_norm = nullptr;
    DeviceTensor *encoder_output_post_norm_b = nullptr;
    DeviceTensor *encoder_output = nullptr;
    DeviceTensor *encoder_output_b = nullptr;

    DeviceTensor *decoder_output_norm = nullptr;
    DeviceTensor *decoder_output_norm_b = nullptr;
    DeviceTensor *output = nullptr; //lm_head (decoder output)
    DeviceTensor *output_quant = nullptr; //quantized version of lm_head

    SimpleLayer output_transform;

    PtrVector<EncoderLayer> encoder_layers;
    PtrVector<DecoderLayer> decoder_layers;

    map<string, LayerTypeAndTensorId> tensor_map;

public:
    void CalculateStat(NetworkStat &stat) const;

    int MaxTensorSize() const;
    static int MaxTensorSize(const EncoderLayer &net);
    static int MaxTensorSize(const DecoderLayer &net);
    static int MaxTensorSize(const AttentionLayer &layer);
    static int MaxTensorSize(const FeedForwardLayer &layer);
};
#endif //USE_CUDA

class StdGgmlNetwork
{
public:
    struct AtomicLayer
    {
        ggml_tensor *weight = nullptr;
        ggml_tensor *bias = nullptr;
    };

    struct SimpleLayer
    {
        AtomicLayer dense;
        AtomicLayer pre_norm, post_norm;
    };

    struct SubLayer
    {
        map<LayerTensorId, ggml_tensor*> tensor_map;
    };

    struct AttentionLayer : public SubLayer
    {
        // normalization
        ggml_tensor *pre_norm = nullptr;
        ggml_tensor *pre_norm_b = nullptr;
        ggml_tensor *post_norm = nullptr;
        ggml_tensor *post_norm_b = nullptr;

        // attention
        ggml_tensor *qkv = nullptr; //qkv: query-key-value
        ggml_tensor *qkv_b = nullptr;
        ggml_tensor *wq = nullptr;
        ggml_tensor *wq_b = nullptr;
        ggml_tensor *wk = nullptr;
        ggml_tensor *wk_b = nullptr;
        ggml_tensor *wv = nullptr;
        ggml_tensor *wv_b = nullptr;
        ggml_tensor *wo = nullptr;
        ggml_tensor *wo_b = nullptr;
    };

    struct FeedForwardLayer : public SubLayer
    {
        // normalization
        ggml_tensor *pre_norm = nullptr;
        ggml_tensor *pre_norm_b = nullptr;
        ggml_tensor *post_norm = nullptr;
        ggml_tensor *post_norm_b = nullptr;

        // ff
        ggml_tensor *w1 = nullptr;
        ggml_tensor *w1_b = nullptr;
        ggml_tensor *w2 = nullptr;
        ggml_tensor *w2_b = nullptr;
        ggml_tensor *w3 = nullptr;
        ggml_tensor *w3_b = nullptr;
        ggml_tensor *w1n3 = nullptr;
        ggml_tensor *w1n3_b = nullptr;
    };

    struct EncoderLayer
    {
        AttentionLayer self_attn;
        FeedForwardLayer ffn;
    };

    struct DecoderLayer : public EncoderLayer
    {
        AttentionLayer cross_attn;
    };

public:
    ggml_tensor *encoder_embeddings;
    ggml_tensor *decoder_embeddings;

    ggml_tensor *encoder_pos_embeddings = nullptr;
    ggml_tensor *encoder_token_type_embeddings = nullptr;

    ggml_tensor *encoder_input_norm = nullptr;     //encoder embeddings layernorm: weight
    ggml_tensor *encoder_input_norm_b = nullptr;   //encoder embeddings layernorm: bias
    ggml_tensor *decoder_input_norm = nullptr;     //decoder embeddings layernorm: weight
    ggml_tensor *decoder_input_norm_b = nullptr;   //decoder embeddings layernorm: bias

    ggml_tensor *encoder_output_norm = nullptr;
    ggml_tensor *encoder_output_norm_b = nullptr;
    ggml_tensor *encoder_output_post_norm = nullptr;
    ggml_tensor *encoder_output_post_norm_b = nullptr;
    ggml_tensor *encoder_output = nullptr;
    ggml_tensor *encoder_output_b = nullptr;

    ggml_tensor *decoder_output_norm = nullptr;
    ggml_tensor *decoder_output_norm_b = nullptr;
    ggml_tensor *output = nullptr;

    SimpleLayer output_transform;

    PtrVector<EncoderLayer> encoder_layers;
    PtrVector<DecoderLayer> decoder_layers;
};

class StdHostNetwork
{
public:
    struct AtomicLayer
    {
        const HostTensor *weight = nullptr;
        const HostTensor *bias = nullptr;
    };

    struct SimpleLayer
    {
        AtomicLayer dense;
        AtomicLayer pre_norm, post_norm;
    };

    struct SubLayer
    {
        map<LayerTensorId, const HostTensor*> tensor_map;
    };

    struct AttentionLayer : public SubLayer
    {
        // normalization
        const HostTensor *pre_norm = nullptr;
        const HostTensor *pre_norm_b = nullptr;
        const HostTensor *post_norm = nullptr;
        const HostTensor *post_norm_b = nullptr;

        // attention
        const HostTensor *qkv = nullptr; //qkv: query-key-value
        const HostTensor *qkv_b = nullptr;
        const HostTensor *wq = nullptr;
        const HostTensor *wq_b = nullptr;
        const HostTensor *wk = nullptr;
        const HostTensor *wk_b = nullptr;
        const HostTensor *wv = nullptr;
        const HostTensor *wv_b = nullptr;
        const HostTensor *wo = nullptr;
        const HostTensor *wo_b = nullptr;
    };

    struct FeedForwardLayer : public SubLayer
    {
        // normalization
        const HostTensor *pre_norm = nullptr;
        const HostTensor *pre_norm_b = nullptr;
        const HostTensor *post_norm = nullptr;
        const HostTensor *post_norm_b = nullptr;

        // ff
        const HostTensor *w1 = nullptr;
        const HostTensor *w1_b = nullptr;
        const HostTensor *w2 = nullptr;
        const HostTensor *w2_b = nullptr;
        const HostTensor *w3 = nullptr;
        const HostTensor *w3_b = nullptr;
        const HostTensor *w1n3 = nullptr;
        const HostTensor *w1n3_b = nullptr;
    };

    struct EncoderLayer
    {
        AttentionLayer self_attn;
        FeedForwardLayer ffn;
    };

    struct DecoderLayer : public EncoderLayer
    {
        AttentionLayer cross_attn;
    };

public:
    HostHalfBuffer encoder_embeddings; //encoder word/token embeddings
    HostHalfBuffer decoder_embeddings; //decoder word/token embeddings

    const HostTensor *encoder_pos_embeddings = nullptr;
    const HostTensor *encoder_token_type_embeddings = nullptr;

    const HostTensor *encoder_input_norm = nullptr;     //encoder embeddings layernorm: weight
    const HostTensor *encoder_input_norm_b = nullptr;   //encoder embeddings layernorm: bias
    const HostTensor *decoder_input_norm = nullptr;     //decoder embeddings layernorm: weight
    const HostTensor *decoder_input_norm_b = nullptr;   //decoder embeddings layernorm: bias

    const HostTensor *encoder_output_norm = nullptr;
    const HostTensor *encoder_output_norm_b = nullptr;
    const HostTensor *encoder_output_post_norm = nullptr;
    const HostTensor *encoder_output_post_norm_b = nullptr;
    const HostTensor *encoder_output = nullptr;
    const HostTensor *encoder_output_b = nullptr;

    const HostTensor *decoder_output_norm = nullptr;
    const HostTensor *decoder_output_norm_b = nullptr;
    const HostTensor *output = nullptr;

    SimpleLayer output_transform;

    PtrVector<EncoderLayer> encoder_layers;
    PtrVector<DecoderLayer> decoder_layers;

public:
    int MaxTensorSize() const;
    static int MaxTensorSize(const EncoderLayer &net);
    static int MaxTensorSize(const DecoderLayer &net);
    static int MaxTensorSize(const AttentionLayer &layer);
    static int MaxTensorSize(const FeedForwardLayer &layer);
};

typedef map<int, HostTensor*> HostTensorMap;

class StdNetwork
{
public:
    StdHostNetwork host_net;
    StdGgmlNetwork ggml_net;

#if defined(USE_CUDA)
    StdDeviceNetwork device_net;
    //When the network is divided by tensor, one sub-net is assigned to one GPU card
    PtrVector<StdDeviceNetwork> device_sub_nets;
#endif

    HostTensorMap tensor_map;
};

struct GenerationConfig
{
    int user_token_id = -1;
    int assistant_token_id = -1;

    int max_new_tokens = -1;

    float temperature = 1.0f;
    int top_k = 5;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
};

//A Transformer model
class TransformerModel
{
public:
    ModelSpec spec;
    StdVocabulary vocabulary;
    StdVocabulary output_vocabulary;
    TensorSpecTable tensor_spec_table;
    GenerationConfig generation_config;

    int pad_token_id = -1;
    int sep_token_id = -1;
    int mask_token_id = -1;
    //int decoder_start_token_id = -1;

    bool is_cpu_tensor_row_major = false;

    const HostTensor *encoder_embeddings = nullptr;
    const HostTensor *decoder_embeddings = nullptr;
    //DeviceTensor *embeddings_gpu = nullptr;
    //HostHalfBuffer embeddings_cpu;

    NetworkStructure network;
    StdNetwork std_network;

public:
    const HostTensor* FindHostTensor(const string &tensor_name) const;
    HostTensor* FindHostTensor(const string &tensor_name);

    static bool CheckModelSpec(const ModelSpec &spec);

    static void InitModelFileFormatMap(ModelFileFormatMap &the_map);
    static void InitNetworkStructureMap(NetworkStructureMap &the_map);
    static void InitMultiGpuStrategyMap(MultiGpuStrategyMap &the_map);
};

TRANSFORMER_END
INFER_FLOW_END
