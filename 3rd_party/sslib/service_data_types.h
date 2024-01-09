#pragma once

#include <string>
#include <iostream>
#include "json.h"

namespace sslib
{

using std::wstring;
using std::ostream;

struct ServiceRequestHeader
{
public:
    wstring fn; //function
    wstring user_id, passcode;

public:
    void Clear();

    void ToJson(wostream &strm) const;
    bool FromJson(const JsonObject &jobj, const JsonDoc &jdoc);
};

struct ServiceResponseHeader
{
public:
    wstring type; //response type
    float time_cost = 0, core_time_cost = 0;
    wstring ret_code, error_text;

public:
    void Clear();

    void ToJson(wostream &strm) const;
    bool FromJson(const JsonObject &jobj, const JsonDoc &jdoc);
};

} //end of namespace
