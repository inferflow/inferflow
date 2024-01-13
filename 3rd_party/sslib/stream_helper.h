#pragma once

#include "binary_stream.h"
#include "prime_types.h"
#include <string>
#include "string_blocked_heap.h"
#include "number.h"

namespace sslib
{

using std::string;
using std::wstring;

class BinStreamHelper
{
public:
    struct ReadStringParams
    {
        char *buffer;
        uint32_t buffer_len;
        uint32_t max_len_allowed;

        ReadStringParams(char *p_buffer = nullptr, uint32_t p_buffer_len = 0,
            uint32_t p_max_len_allowed = 0)
        {
            buffer = p_buffer;
            buffer_len = p_buffer_len;
            max_len_allowed = p_max_len_allowed;
        }
    };

public:
    static bool ReadString32(IBinStream &strm, string &str, void *params = nullptr)
    {
        return ReadString32(strm, str, (ReadStringParams*)params);
    }
    static bool ReadWString32(IBinStream &strm, wstring &str, void *params = nullptr)
    {
        return ReadWString32(strm, str, (ReadStringParams*)params);
    }

    static bool ReadString16(IBinStream &strm, string &str, void *params = nullptr)
    {
        return ReadString16(strm, str, (ReadStringParams*)params);
    }
    static bool ReadWString16(IBinStream &strm, wstring &str, void *params = nullptr)
    {
        return ReadWString16(strm, str, (ReadStringParams*)params);
    }

    static bool ReadString32(IBinStream &strm, string &str, ReadStringParams *params);
    static bool ReadWString32(IBinStream &strm, wstring &str, ReadStringParams *params);
    static bool ReadString16(IBinStream &strm, string &str, ReadStringParams *params);
    static bool ReadWString16(IBinStream &strm, wstring &str, ReadStringParams *params);

    static bool ReadString32(IBinStream &strm, const char *&str, StringBlockedHeap &heap);
    static bool ReadWString32(IBinStream &strm, const wchar_t *&str, StringBlockedHeap &heap);
    static bool ReadString16(IBinStream &strm, const char *&str, StringBlockedHeap &heap);
    static bool ReadWString16(IBinStream &strm, const wchar_t *&str,
        StringBlockedHeap &heap, bool is_debug_mode = false);

    static bool WriteString32(IBinStream &strm, const string &str, void *params = nullptr)
    {
        (void)params;
        return WriteString32(strm, str.c_str(), (uint32_t)str.size());
    }
    static bool WriteWString32(IBinStream &strm, const wstring &str, void *params = nullptr)
    {
        (void)params;
        return WriteWString32(strm, str.c_str(), (uint32_t)str.size());
    }

    static bool WriteString16(IBinStream &strm, const string &str, void *params = nullptr)
    {
        (void)params;
        return WriteString16(strm, str.c_str(), (uint16_t)str.size());
    }
    static bool WriteWString16(IBinStream &strm, const wstring &str, void *params = nullptr)
    {
        (void)params;
        return WriteWString16(strm, str.c_str(), (uint16_t)str.size());
    }

    static bool WriteString32(IBinStream &strm, const char *data, uint32_t data_len = UINT32_MAX);
    static bool WriteWString32(IBinStream &strm, const wchar_t *data, uint32_t data_len = UINT32_MAX);
    static bool WriteString16(IBinStream &strm, const char *data, uint16_t data_len = UINT16_MAX);
    static bool WriteWString16(IBinStream &strm, const wchar_t *data, uint16_t data_len = UINT16_MAX);
};

} //end of namespace
