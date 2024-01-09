#pragma once

#include <cstring>
#include <vector>
#include <map>
#include "tensor/tensor_common.h"
#include "namespace.inc"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::string;
using std::vector;
using std::map;
using std::pair;

struct TensorSpec
{
public:
    static const int MaxDimCount = 3;

public:
    int id = 0;
    string name;
    int dims = 0;
    int ne[MaxDimCount];
    int data_type = 0; //data type on disk (or in the file)
    uint32_t size = 0;
    int file_idx = 0;
    uint64_t offset_in_file = 0;
    int flag = 0;
    ElementType data_type_host = ElementType::Auto;
    ElementType data_type_device = ElementType::Auto;
    int data_source = -1;

    TensorSpec()
    {
        memset(ne, 0, MaxDimCount * sizeof(int));
    }

    void Clear()
    {
        id = 0;
        name.clear();
        dims = 0;
        memset(ne, 0, MaxDimCount * sizeof(int));
        data_type = 0;
        size = 0;
        file_idx = 0;
        offset_in_file = 0;
        flag = 0;
        data_type_host = ElementType::Auto;
        data_type_device = ElementType::Auto;
        data_source = -1;
    }
};

class TensorSpecTable
{
public:
    vector<TensorSpec> tensor_array;
    map<string, int> name2idx;

public:
    int Size() const {
        return (int)tensor_array.size();
    }

    int Add(const TensorSpec &spec)
    {
        int ret_idx = -1;
        auto iter = this->name2idx.find(spec.name);
        if (iter == this->name2idx.end())
        {
            ret_idx = (int)tensor_array.size();
            tensor_array.push_back(spec);
            name2idx[spec.name] = ret_idx;
        }
        else
        {
            ret_idx = iter->second;
        }

        return ret_idx;
    }

    int GetIndex(const string &tensor_name) const
    {
        auto iter = name2idx.find(tensor_name);
        return iter == name2idx.end() ? -1 : iter->second;
    }
};

enum class NetworkType
{
    General,
    Transformer,
    EncoderDecoder_Transformer,
    //add more encoder-decoder model types here
    EncoderOnly_Transformer,
    BERT,
    //add more encoder-only model types here
    DecoderOnly_Transformer,
    BLOOM,
    LLAMA,
    MISTRAL
    //add more decoder-only model types here
};

enum class MultiGpuStrategy
{
    BY_LAYER,
    BY_TENSOR,
    HYBRID
};

enum class LayerType
{
    SELF_ATTN,
    CROSS_ATTN,
    FFN //feed forward
};

enum class LayerTensorId
{
    ATTN_PRE_NORM,
    ATTN_PRE_NORM_B,
    ATTN_POST_NORM,
    ATTN_POST_NORM_B,

    QKV,
    QKV_B,
    WQ,
    WQ_B,
    WK,
    WK_B,
    WV,
    WV_B,
    WO,
    WO_B,

    FFN_PRE_NORM,
    FFN_PRE_NORM_B,
    FFN_POST_NORM,
    FFN_POST_NORM_B,

    W1,
    W1_B,
    W2,
    W2_B,
    W3,
    W3_B,
    W1N3,
    W1N3_B
};

typedef pair<LayerType, LayerTensorId> LayerTypeAndTensorId;

class NetworkStructure
{
public:
    bool Init(NetworkType network_type, int encoder_layer_count,
        int decoder_layer_count, const string &tensor_name_prefix,
        const map<string, string> *tensor_map_ptr = nullptr);
    bool UpdateTensorSpecTable(TensorSpecTable &spec_table) const;

    bool TransTensorName(string &target_name, const string &src_name) const;

    static bool IsEncoderDecoderTransformer(NetworkType t);
    static bool IsEncoderOnlyTransformer(NetworkType t);
    static bool IsDecoderOnlyTransformer(NetworkType t);

    static void BuildTensorNameToIdMap(map<string, LayerTypeAndTensorId> &tensor_map);

protected:
    //the mapping from external names to standard names
    map<string, string> tensor_name_map_;

    map<string, string> tensor_map_ex_;

protected:
    void ExpandTensorNameMap(map<string, string> &tensor_map_ex,
        const map<string, string> &tensor_map, int layers) const;

    void BuildTensorNameMap(map<string, string> &tensor_map, NetworkType net_type,
        const string &tensor_name_prefix);
    void BuildTensorNameMap_TransformerEncoder(map<string, string> &tensor_map,
        const string &tensor_name_prefix);
    void BuildTensorNameMap_TransformerDecoder(map<string, string> &tensor_map,
        const string &tensor_name_prefix);
    void BuildTensorNameMap_TransformerDecoderOnly(map<string, string> &tensor_map,
        const string &tensor_name_prefix);
    void BuildTensorNameMap_Llama(map<string, string> &tensor_map,
        const string &tensor_name_prefix);
    void BuildTensorNameMap_Bloom(map<string, string> &tensor_map,
        const string &tensor_name_prefix);
};

TRANSFORMER_END
INFER_FLOW_END
