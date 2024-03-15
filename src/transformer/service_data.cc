#include "service_data.h"
#include <sstream>

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

void InferFlowRequest::Clear()
{
    header.Clear();
    query.Clear();
    decoding_alg.clear();
    random_seed = 0;
    temperature = 1.0f;
    is_streaming_mode = true;
}

void InferFlowRequest::ToJson(std::wostream &writer) const
{
    writer << L"{";
    writer << L"\n\t\"header\":";
    header.ToJson(writer);

    writer << L",\n\t\"text\":"; 
    JsonBuilder::AppendFieldValue(writer, query.text);

    if (!query.system_prompt.empty())
    {
        writer << L",\n\t\"system_prompt\":";
        JsonBuilder::AppendFieldValue(writer, query.system_prompt);
    }

    if (!query.response_prefix.empty())
    {
        writer << L",\n\t\"res_prefix\":";
        JsonBuilder::AppendFieldValue(writer, query.response_prefix);
    }

    if (!query.encoder_input_template.empty())
    {
        writer << L",\n\t\"encoder_input_template\":";
        JsonBuilder::AppendFieldValue(writer, query.encoder_input_template);
    }

    if (!query.decoder_input_template.empty())
    {
        writer << L",\n\t\"decoder_input_template\":";
        JsonBuilder::AppendFieldValue(writer, query.decoder_input_template);
    }

    if (!decoding_alg.empty())
    {
        writer << L",\n\t\"decoding_alg\":";
        JsonBuilder::AppendFieldValue(writer, decoding_alg);
    }

    writer << L",\n\t\"max_output_len\":" << max_output_len;
    writer << L",\n\t\"random_seed\":" << random_seed;
    writer << L",\n\t\"temperature\":" << temperature;
    writer << L",\n\t\"is_streaming_mode\":" << (is_streaming_mode ? 1 : 0);

    writer << L"}";
}

bool InferFlowRequest::FromJson(const wstring &jstr, JsonParser &jparser)
{
    Clear();

    JsonDoc jdoc;
    bool ret = jparser.Parse(jdoc, jstr);
    Macro_RetFalseIf(!ret);

    JsonObject jobj;
    jdoc.GetJObject(jobj);
    ret = FromJson(jobj, jdoc);
    return ret;
}

bool InferFlowRequest::FromJson(const JsonObject &jobj,
    const JsonDoc &jdoc)
{
    Clear();

    //header
    JsonObject header_obj;
    jobj.GetFieldValue(header_obj, L"header", jdoc);
    bool ret = header.FromJson(header_obj, jdoc);

    JsonArray messages_array;
    bool messages_ret = jobj.GetFieldValue(messages_array, L"messages", jdoc);
    if (messages_ret)
    {
        ret = FromJson_OpenAI(messages_array, jdoc);

        jobj.GetFieldValue(max_output_len, L"max_tokens", jdoc);
        jobj.GetFieldValue(is_streaming_mode, L"stream", jdoc);
        jobj.GetFieldValue(temperature, L"temperature", jdoc);
        jobj.GetFieldValue(random_seed, L"seed", jdoc);

        return ret;
    }

    //
    ret = FromJson_Std(jobj, jdoc);

    return ret;
}

bool InferFlowRequest::FromJson_Std(const JsonObject &jobj,
    const JsonDoc &jdoc)
{
    bool ret = true;

    ret = jobj.GetFieldValue(query.text, L"text", jdoc);
    jobj.GetFieldValue(query.system_prompt, L"system_prompt", jdoc);
    jobj.GetFieldValue(query.response_prefix, L"res_prefix", jdoc);
    jobj.GetFieldValue(query.encoder_input_template, L"encoder_input_template", jdoc);
    jobj.GetFieldValue(query.decoder_input_template, L"decoder_input_template", jdoc);

    jobj.GetFieldValue(max_output_len, L"max_output_len", jdoc);
    jobj.GetFieldValue(max_output_len, L"max_new_tokens", jdoc);

    jobj.GetFieldValue(decoding_alg, L"decoding_alg", jdoc);
    jobj.GetFieldValue(random_seed, L"random_seed", jdoc);
    jobj.GetFieldValue(temperature, L"temperature", jdoc);

    int num = 0;
    if (jobj.GetFieldValue(num, L"is_streaming_mode", jdoc))
    {
        is_streaming_mode = num != 0;
    }
    jobj.GetFieldValue(is_streaming_mode, L"is_streaming_mode", jdoc);

    return ret;
}

bool InferFlowRequest::FromJson_OpenAI(const JsonArray &jarr,
    const JsonDoc &jdoc)
{
    bool ret = true;

    for (size_t i = 0; i < jarr.size; i++) 
    {
        JsonObject message_obj = jarr.items[i].GetJObject();
        wstring role;
        ret = message_obj.GetFieldValue(role, L"role", jdoc);
        Macro_RetFalseIf(!ret);

        if (role == L"user") {
            message_obj.GetFieldValue(query.text, L"content", jdoc);
        }
        else if (role == L"system") {
            message_obj.GetFieldValue(query.system_prompt, L"content", jdoc);
        }
        else if (role == L"assistant") {
            message_obj.GetFieldValue(query.response_prefix, L"content", jdoc);
        }
    }

    if (query.text.empty()) {
        ret = false;
    }

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// InferFlowResponseChunk

void InferFlowResponseChunk::Clear()
{
    ret_code.clear();
    error_text.clear();
    time_cost = 0;
    text.clear();
    text_utf8_len = 0;
    is_end = false;
}

void InferFlowResponseChunk::ToJson(wstring &jstr) const
{
    wstringstream ss;
    ss << L"{";
    ss << L"\"text\":";
    JsonBuilder::AppendFieldValue(ss, text);

    if (!ret_code.empty())
    {
        ss << L",\"ret_code\":";
        JsonBuilder::AppendFieldValue(ss, ret_code);
    }

    if (!error_text.empty())
    {
        ss << L",\"error_text\":";
        JsonBuilder::AppendFieldValue(ss, error_text);
    }

    ss << L",\"time_cost\":" << time_cost;
    if (is_end) {
        ss << L",\"is_end\":" << (is_end ? 1 : 0);
    }

    ss << L"}";
    jstr = ss.str();
}

void InferFlowResponseChunk::ToJson_OpenAI(wstring &jstr) const
{
    wstringstream ss;
    ss << L"{";
    ss << L"\"object\":\"chat.completion\"";
    ss << L",\"choices\":[";
    ss << L"{";
    ss << L"\"index\":0";
    ss << L",\"logprobs\":null";
    ss << L",\"finish_reason\":";
    if (error_text.empty()) {
        ss << L"\"stop\"";
    }
    else {
       JsonBuilder::AppendFieldValue(ss, error_text);
    }
    ss << L",\"message\":{";
    ss << L"\"content\":";
    JsonBuilder::AppendFieldValue(ss, text);
    ss << L"}";
    ss << L"}";
    ss << L"]";
    ss << L"}";
    jstr = ss.str();
}

void InferFlowResponseChunk::ToJson_OpenAI_Chunk(wstring &jstr) const
{
    wstringstream ss;
    ss << L"data: ";
    ss << L"{";
    ss << L"\"object\":\"chat.completion.chunk\"";
    ss << L",\"choices\":[";
    ss << L"{";
    ss << L"\"index\":0";
    ss << L",\"logprobs\":null";
    ss << L",\"finish_reason\":";
    if (!is_end) {
        ss << L"null";
    }
    else {
        ss << L"\"stop\"";
    }
    ss << L",\"delta\":{";
    ss << L"\"content\":";
    JsonBuilder::AppendFieldValue(ss, text);
    ss << L"}";
    ss << L"}";
    ss << L"]";
    ss << L"}";
    jstr = ss.str();
}

bool InferFlowResponseChunk::FromJson(const wstring &jstr, JsonParser &parser)
{
    Clear();

    JsonDoc jdoc;
    bool ret = parser.Parse(jdoc, jstr);
    Macro_RetFalseIf(!ret);

    JsonObject jobj;
    jdoc.GetJObject(jobj);
    ret = FromJson(jobj, jdoc);
    return ret;
}

bool InferFlowResponseChunk::FromJson(const JsonObject &jobj,
    const JsonDoc &jdoc)
{
    Clear();
    jobj.GetFieldValue(ret_code, L"ret_code", jdoc);
    jobj.GetFieldValue(error_text, L"error_text", jdoc);
    jobj.GetFieldValue(time_cost, L"time_cost", jdoc);

    int num = 0;
    jobj.GetFieldValue(num, L"is_end", jdoc);
    is_end = num != 0;

    bool ret = jobj.GetFieldValue(text, L"text", jdoc);
    return ret;
}

TRANSFORMER_END
INFER_FLOW_END
