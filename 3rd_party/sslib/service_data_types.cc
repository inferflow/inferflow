#include "service_data_types.h"

namespace sslib
{

////////////////////////////////////////////////////////////////////////////////
// ServiceRequestHeader

void ServiceRequestHeader::Clear()
{
    fn.clear();
    user_id.clear();
    passcode.clear();
}

void ServiceRequestHeader::ToJson(wostream &strm) const
{
    wstring encoded_str;
    strm << L"{";

    JsonBuilder::EncodeString(encoded_str, fn);
    strm << L"\"fn\":\"" << encoded_str << L"\"";
    JsonBuilder::EncodeString(encoded_str, user_id);
    strm << L",\"user\":\"" << encoded_str << L"\"";
    JsonBuilder::EncodeString(encoded_str, passcode);
    strm << L",\"passcode\":\"" << encoded_str << L"\"";

    strm << L"}";
}

bool ServiceRequestHeader::FromJson(const JsonObject &jobj, const JsonDoc &jdoc)
{
    jobj.GetFieldValue(fn, L"fn", jdoc);
    jobj.GetFieldValue(user_id, L"user", jdoc);
    jobj.GetFieldValue(passcode, L"passcode", jdoc);
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// ServiceResponseHeader

void ServiceResponseHeader::Clear()
{
    type.clear();
    time_cost = 0;
    core_time_cost = 0;
    ret_code.clear();
    error_text.clear();
}

void ServiceResponseHeader::ToJson(wostream &strm) const
{
    wstring encoded_str;
    strm << L"{";

    strm << L"\"time_cost_ms\":" << time_cost;
    strm << L",\"time_cost\":" << time_cost / 1000.0f;
    strm << L",\"core_time_cost_ms\":" << core_time_cost;
    if (!type.empty())
    {
        JsonBuilder::EncodeString(encoded_str, type);
        strm << L",\"type\":\"" << encoded_str << L"\"";
    }

    JsonBuilder::EncodeString(encoded_str, ret_code);
    strm << L",\"ret_code\":\"" << encoded_str << L"\"";

    if (!error_text.empty())
    {
        JsonBuilder::EncodeString(encoded_str, error_text);
        strm << L",\"error_text\":\"" << encoded_str << L"\"";
    }

    strm << L"}";
}

bool ServiceResponseHeader::FromJson(const JsonObject &jobj, const JsonDoc &jdoc)
{
    jobj.GetFieldValue(time_cost, L"time_cost_ms", jdoc);
    jobj.GetFieldValue(core_time_cost, L"core_time_cost_ms", jdoc);
    jobj.GetFieldValue(type, L"type", jdoc);
    jobj.GetFieldValue(ret_code, L"ret_code", jdoc);
    jobj.GetFieldValue(error_text, L"error_text", jdoc);
    return true;
}

} //end of namespace

