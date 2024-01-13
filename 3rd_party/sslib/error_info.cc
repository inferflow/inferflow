#include "error_info.h"
#include <stdio.h>
#include <stdarg.h>
#include "string_util.h"

namespace sslib
{

void ErrorInfo::Clear()
{
    id = 0;
    sid.clear();
    pos.Set(0, 0);
    ext.Set(0, 0);
    text.clear();
}

bool ErrorInfo::HasError() const
{
    return id != 0;
}

void ErrorInfo::SetId(uint32_t p_id, const wstring &p_sid)
{
    id = p_id;
    sid = p_sid;
}

void ErrorInfo::Set(uint32_t p_id, uint32_t p_offset)
{
    id = p_id;
    pos.Set(p_offset, p_offset);
}

void ErrorInfo::Set(uint32_t p_id, const UIntRange &p_pos)
{
    id = p_id;
    pos = p_pos;
}

void ErrorInfo::Set(uint32_t p_id, const wstring &p_sid, const UIntRange &p_pos)
{
    id = p_id;
    sid = p_sid;
    pos = p_pos;
}

void ErrorInfo::FormatText(const wchar_t *format_str, ...)
{
    text.clear();
    if (format_str != nullptr)
    {
        va_list args;
        va_start(args, format_str);
        wchar_t buf[MAX_MSG_SIZE + 1];
        int count = vswprintf(buf, MAX_MSG_SIZE, format_str, args);
        if (count > 0 && count <= MAX_MSG_SIZE) {
            text.assign(buf, count);
        }
        va_end(args);
    }
}

void ErrorInfo::FormatText(const char *format_str, ...)
{
    text.clear();
    if (format_str != nullptr)
    {
        va_list args;
        va_start(args, format_str);
        char buf[MAX_MSG_SIZE + 1];
        int count = vsnprintf(buf, MAX_MSG_SIZE, format_str, args);
        if (count > 0 && count <= MAX_MSG_SIZE) {
            StringUtil::Utf8ToWideStr(text, buf, count);
        }
        va_end(args);
    }
}

} //end of namespace
