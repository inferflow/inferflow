#pragma once

#include "sslib/service_data_types.h"
#include "inference_types.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::string;
using std::wstring;
using sslib::JsonDoc;
using sslib::JsonObject;
using sslib::JsonParser;
using sslib::JsonArray;

class InferFlowRequest
{
public:
    sslib::ServiceRequestHeader header;
    LlmQuery query;
    wstring decoding_alg;
    uint32_t max_output_len = 1024; //maximal number of output tokens
    int random_seed = 0;
    float temperature = 1.0f;
    bool is_streaming_mode = false;

public:
    void Clear();

    void ToJson(std::wostream &writer) const;
    bool FromJson(const wstring &jstr, JsonParser &jparser);
    bool FromJson(const JsonObject &jobj, const JsonDoc &jdoc);

    bool FromJson_Std(const JsonObject &jobj, const JsonDoc &jdoc);
    bool FromJson_OpenAI(const JsonArray &jobj, const JsonDoc &jdoc);
};

class InferFlowResponseChunk
{
public:
    wstring ret_code;
    wstring error_text;
    float time_cost = 0;
    wstring text;
    int text_utf8_len = 0;
    bool is_end = false;

public:
    void Clear();

    void ToJson(wstring &jstr) const;
    void ToJson_OpenAI(wstring &jstr) const;
    void ToJson_OpenAI_Chunk(wstring &jstr) const;

    bool FromJson(const wstring &jstr, JsonParser &parser);
    bool FromJson(const JsonObject &jobj, const JsonDoc &jdoc);
};

TRANSFORMER_END
INFER_FLOW_END
