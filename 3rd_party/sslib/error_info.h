#pragma once

#include <string>
#include "prime_types.h"
#include "int_range.h"

namespace sslib
{

using std::wstring;

class ErrorInfo
{
public:
    uint32_t id = 0; //error id
    wstring sid; //id in string format
    UIntRange pos; //position
    PairUInt32 ext; //extended information
    wstring text; //error text

public:
    void Clear();

    bool HasError() const;
    void SetId(uint32_t p_id, const wstring &p_sid);
    void Set(uint32_t p_id, uint32_t offset);
    void Set(uint32_t p_id, const UIntRange &p_pos);
    void Set(uint32_t p_id, const wstring &p_sid, const UIntRange &p_pos);

    void FormatText(const wchar_t *format_str, ...);
    void FormatText(const char *format_str, ...);

protected:
    const static int MAX_MSG_SIZE = 8192;
};

} //end of namespace
