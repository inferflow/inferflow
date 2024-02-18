#pragma once

#include "sslib/binary_stream.h"
#include "sslib/json.h"
#include "common/pickle_reader.h"
#include "tensor/host_tensor.h"
#include "model.h"
#include "transformer_types.h"
#include "network_builder.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::string;
using std::wstring;
using std::map;
using sslib::RawString;
using sslib::IBinaryStream;
using sslib::JsonParser;
using sslib::JsonDoc;
using sslib::JsonObject;
using sslib::JsonArray;
using sslib::TaskMonitor;

class ModelReader
{
public:
    void Init(NetworkBuilder *network_builder);

    bool Load(TransformerModel &model, TransformerContext &ctx,
        const ModelSpec &spec, bool is_study_mode = false);

    static bool LoadModelSpecJson(ModelSpec &model_spec, const string &file_path,
        const JsonParser &jparser);

protected:
    struct TokenizerConfig
    {
        string unk_str, bos_str, eos_str;
        int bos_id = -1, eos_id = -1;
        int unk_id = -1, pad_id = -1, sep_id = -1;
        int decoder_start_id = -1;

        void Clear()
        {
            unk_str.clear();
            bos_str.clear();
            eos_str.clear();
            bos_id = -1;
            eos_id = -1;
            unk_id = -1;
            pad_id = -1;
            sep_id = -1;
            decoder_start_id = -1;
        }
    };

    struct SpecialTokens
    {
        string bos, eos, unk, sep, pad, mask;
        vector<string> additional;

        void Clear()
        {
            bos.clear();
            eos.clear();
            unk.clear();
            sep.clear();
            pad.clear();
            mask.clear();
            additional.clear();
        }
    };

protected:
    map<int, uint8_t> token_char_to_byte_map_;
    NetworkBuilder *network_builder_ = nullptr;

protected:
    static bool LoadConfigJson(TransformerModel &model, TokenizerConfig &tok_config,
        const ModelSpec &spec, const string &file_path, const JsonParser &jparser,
        bool has_embedded_hparams);

    static bool LoadGeneratinConfig(TransformerModel &model,
        const string &file_path, const JsonParser &jparser);

    bool LoadTokenizer(TransformerModel &model, const ModelSpec &spec,
        const JsonParser &jparser);
    bool LoadSpecialTokens(SpecialTokens &special_tokens, const string &file_path,
        const JsonParser &jparser);
    bool LoadVocabJson(StdVocabulary &vocab, const ModelSpec &spec,
        const JsonObject &vocab_obj, const map<int, int> &token_map,
        int token_bytes_mapping);
    bool LoadTokenMerges(StdVocabulary &vocab, const ModelSpec &spec,
        const JsonArray &merge_array, int token_bytes_mapping);
    bool LoadTokenizer_Json(StdVocabulary &vocab, const ModelSpec &spec,
        const string &file_path, const JsonParser &jparser,
        const SpecialTokens &special_tokens, int token_bytes_mapping,
        const string &token_remap_file);
    bool LoadTokenizer_Txt(StdVocabulary &vocab, const ModelSpec &spec,
        const string &file_path, bool is_output_vocab);
    bool LoadTokenizer_Bin(StdVocabulary &vocab, const ModelSpec &spec,
        const string &file_path, const SpecialTokens &special_tokens,
        int token_bytes_mapping) const;

    bool ReadVocabulary_Std(TransformerModel &model, IBinaryStream &strm, bool has_score) const;
    bool ReadVocabulary_Format2(TransformerModel &model, IBinaryStream &strm) const;

    static void GetSpecialTokenIds(TransformerModel &model, const JsonObject &jobj,
        const JsonDoc &jdoc);

    static bool LoadTokenRemapData(map<int, int> &token_map, const string &file_path,
        bool is_reverse = false);

    static bool LoadWeightMap(map<string, string> &weight_map,
        const string &file_path, JsonParser &jparser);

    static void GetFileSetFromWeightMap(vector<string> &file_list,
        const map<string, string> &weight_map);

    static HostTensor* ReadTensor(ElementType data_type, int cx, int cy,
        IBinaryStream &strm, TransformerContext &ctx);

    static bool ReadAndTransTensor(HostTensor &tensor, HostTensor *mem_tensor,
        TensorSpec &spec, TransformerContext &ctx, IBinaryStream &strm);

    void InitTokenByteMap();
    bool DecodeTokenStr(string &target, const wstring &src, int alg) const;

public:
    const static int MAX_TENSORS_PER_DATA_SECTION = 5;
    struct StrAndCount
    {
        string str;
        int count = 0;
        int tensor_id_arr[MAX_TENSORS_PER_DATA_SECTION];

        StrAndCount(const string &s = "", int c = 0)
        {
            str = s;
            count = c;
        }
    };

protected:

    ////////////////////////////////////////////////////////////////////////////
    // Std format
    ////////////////////////////////////////////////////////////////////////////

    static bool LoadModel_Std(TransformerModel &model, TransformerContext &ctx,
        const ModelSpec &spec, bool is_study_mode = false);

    ////////////////////////////////////////////////////////////////////////////
    // Pickle format
    ////////////////////////////////////////////////////////////////////////////

    //Pytorch model save format (typically with a .pt, .pth or .bin file extension)
    bool LoadModel_Pickle(TransformerModel &model, TransformerContext &ctx,
        const ModelSpec &spec, const vector<string> &model_file_list,
        bool is_study_mode = false);

    static bool Pickle_ReadHeader(TransformerModel &model,
        map<string, StrAndCount> &section_to_tensor_name_map,
        IBinaryStream &strm, PickleReader &reader, int file_idx);

    int Pickle_ReadTensor(TransformerModel &model, TransformerContext &ctx,
        IBinaryStream &strm, PickleReader &reader, int file_idx,
        const map<string, StrAndCount> &section_to_tensor_name_map,
        const ModelPartition &model_partition,
        const PtrVector<NetworkBuilder> &builder_list,
        bool is_study_mode);

    static void Pickle_HandleSectionName(map<string, StrAndCount> &section_to_tensor_name_map,
        TensorSpec &tensor, const TransformerModel &model, const string &section_name);

    ////////////////////////////////////////////////////////////////////////////
    // Safetensors format
    ////////////////////////////////////////////////////////////////////////////

    //https://huggingface.co/docs/safetensors/index
    bool LoadModel_Safetensors(TransformerModel &model, TransformerContext &ctx,
        const ModelSpec &spec, const vector<string> &model_file_list,
        JsonParser &jparser, bool is_study_mode = false);

    static bool Safetensors_ReadHeader(TransformerModel &model,
        IBinaryStream &strm, JsonParser &jparser);
    bool Safetensors_ReadTensors(TransformerModel &model, TransformerContext &ctx,
        IBinaryStream &strm, int base_idx, TaskMonitor &tm);

    ////////////////////////////////////////////////////////////////////////////
    // llama2.c format
    ////////////////////////////////////////////////////////////////////////////

    bool LoadModel_Llama2DotC(TransformerModel &model, TransformerContext &ctx,
        const ModelSpec &spec, bool is_study_mode = false);

    static bool Llama2DotC_ReadTensors(TransformerModel &model, TransformerContext &ctx,
        IBinaryStream &strm, int file_version, uint64_t file_length,
        bool is_shared_classifier);

    ////////////////////////////////////////////////////////////////////////////
    // GGML format (legacy format of llama.cpp)
    ////////////////////////////////////////////////////////////////////////////

    bool LoadModel_GGML(TransformerModel &model, TransformerContext &ctx,
        const ModelSpec &spec, bool is_study_mode = false);

    static bool LlamaCpp_ReadMagic(int &file_version, IBinaryStream &strm);
    static bool LlamaCpp_ReadHyperParams(TransformerModel &model, IBinaryStream &strm);
    static bool LlamaCpp_ReadTensors(TransformerModel &model, TransformerContext &ctx,
        IBinaryStream &strm, int file_version, uint64_t file_length);
    static uint32_t LlamaCpp_CalcTensorSize(const TensorSpec &spec, int file_version);

    ////////////////////////////////////////////////////////////////////////////
    // llama.cpp GGUF format
    ////////////////////////////////////////////////////////////////////////////

    struct GgufHeader
    {
        uint32_t magic = 0;
        uint32_t version = 2;
        uint64_t tensor_num = 0;
        uint64_t attr_num = 0; //or :kv_num
    };

    enum class GgufValueType
    {
        T_UINT8 = 0,
        T_INT8 = 1,
        T_UINT16 = 2,
        T_INT16 = 3,
        T_UINT32 = 4,
        T_INT32 = 5,
        T_FLOAT32 = 6,
        T_BOOL = 7,
        T_STRING = 8,
        T_ARRAY = 9,
        T_UINT64 = 10,
        T_INT64 = 11,
        T_FLOAT64 = 12,
        T_COUNT //type count
    };

    struct GgufStr
    {
        uint64_t size = 0;
        char *data = nullptr;
    };

    struct GgufArr
    {
        GgufValueType value_type = GgufValueType::T_UINT32;
        uint64_t size = 0;
        void *data = nullptr;
    };

    union GgufAttrValue
    {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        bool     bval;

        GgufStr str;
        GgufArr arr;

        GgufAttrValue() {};
    };

    //GGUF attribute (called a KV in llama.cpp)
    struct GgufAttr
    {
        string name;
        GgufAttrValue value;
        GgufValueType value_type;
    };

    bool LoadModel_GGUF(TransformerModel &model, TransformerContext &ctx,
        const ModelSpec &spec, bool is_study_mode = false);

    bool GGUF_ReadHeader(GgufHeader &header, IBinaryStream &strm);
    bool GGUF_ReadAttributes(map<string, GgufAttr> &attr_map, IBinaryStream &strm,
        const GgufHeader &header);
    bool GGUF_ReadTensorSpecTable(TensorSpecTable &tensor_spec_table,
        IBinaryStream &strm, const GgufHeader &header);
    bool GGUF_ReadHyperParams(TransformerModel &model, const map<string, GgufAttr> &attr_map);
    bool GGUF_ReadVocabulary(TransformerModel &model, const map<string, GgufAttr> &attr_map,
        int token_bytes_mapping);

    bool GGUF_ReadAttrValue(GgufAttrValue &value, IBinaryStream &strm,
        GgufValueType value_type, int version);
    static bool GGUF_ReadString(GgufStr &str, IBinaryStream &strm, int version);
    static bool GGUF_ReadString(string &str, IBinaryStream &strm, int version,
        char *buf, uint64_t buf_len);
    static bool GGUF_ReadSize(uint64_t &size, IBinaryStream &strm, int version);

    static uint32_t GGUF_GetAttrValue_U32(const map<string, GgufAttr> &attr_map,
        const string &attr_name);
    static int GGUF_GetAttrValue_I32(const map<string, GgufAttr> &attr_map,
        const string &attr_name);
    static string GGUF_GetAttrValue_Str(const map<string, GgufAttr> &attr_map,
        const string &attr_name);
    static const GgufArr* GGUF_GetAttrValue_Arr(const map<string, GgufAttr> &attr_map,
        const string &attr_name);

    static string GGUF_BuildAttrName(const string &name_template, const string &model_name);
};

TRANSFORMER_END
INFER_FLOW_END
