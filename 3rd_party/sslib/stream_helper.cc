#include "stream_helper.h"

namespace sslib
{

//static
bool BinStreamHelper::ReadString32(IBinStream &strm, string &str, ReadStringParams *params)
{
    str.clear();
    uint32_t len = 0;
    bool ret = strm.Read((char*)&len, sizeof(len));
    if (!ret) {
        return false;
    }

    if (len > 0 && (params == nullptr || len <= params->max_len_allowed || params->max_len_allowed <= 0))
    {
        bool is_new_buffer = params == nullptr || params->buffer == nullptr || params->buffer_len < len;
        char *buf = is_new_buffer ? new char[len] : params->buffer;
        ret = strm.Read(buf, len);
        if (ret) {
            str.assign(buf, len);
        }

        if (is_new_buffer) {
            delete[] buf;
        }
    }
    else
    {
        str.clear();
        if (len != 0) {
            ret = false;
        }
    }
    return ret;
}

//static
bool BinStreamHelper::ReadWString32(IBinStream &strm, wstring &str, ReadStringParams *params)
{
    str.clear();
    uint32_t len = 0;
    bool ret = strm.Read(len);
    if (!ret) {
        return false;
    }

    if (len > 0 && (params == nullptr || len <= params->max_len_allowed || params->max_len_allowed <= 0))
    {
        bool is_new_buffer = params == nullptr || params->buffer == nullptr || params->buffer_len < len;
        char *buf = is_new_buffer ? new char[len + 1] : params->buffer;
        ret = strm.Read(buf, len);
        if (ret)
        {
            //str.assign((const wchar_t*)buf, len/2);
            str.reserve(len / 2);
            for (uint32_t char_idx = 0; char_idx < len; char_idx += 2) {
                str += (wchar_t)*(uint16_t*)(buf + char_idx);
            }
        }

        if (is_new_buffer) {
            delete[] buf;
        }
    }
    else
    {
        str.clear();
        if (len != 0) {
            ret = false;
        }
    }
    return ret;
}

//static
bool BinStreamHelper::ReadString16(IBinStream &strm, string &str, ReadStringParams *params)
{
    str.clear();
    uint16_t len = 0;
    bool ret = strm.Read((char*)&len, sizeof(len));
    if (ret)
    {
        if (len > 0 && (params == nullptr || len <= params->max_len_allowed || params->max_len_allowed <= 0))
        {
            bool is_new_buffer = params == nullptr || params->buffer == nullptr || params->buffer_len < len;
            char *buf = is_new_buffer ? new char[len] : params->buffer;
            ret = strm.Read(buf, len);
            if (ret) {
                str.assign(buf, len);
            }

            if (is_new_buffer) {
                delete[] buf;
            }
        }
        else
        {
            str.clear();
            if (len != 0) {
                ret = false;
            }
        }
    }
    return ret;
}

//static
bool BinStreamHelper::ReadWString16(IBinStream &strm, wstring &str, ReadStringParams *params)
{
    str.clear();
    uint16_t len = 0;
    bool ret = strm.Read((char*)&len, sizeof(len));
    if (ret)
    {
        if (len > 0 && (params == nullptr || len <= params->max_len_allowed || params->max_len_allowed <= 0))
        {
            bool is_new_buffer = params == nullptr || params->buffer == nullptr || params->buffer_len < len;
            char *buf = is_new_buffer ? new char[len + 1] : params->buffer;
            ret = strm.Read(buf, len);
            if (ret)
            {
                //str.assign((const wchar_t*)buf, len / 2);
                str.reserve(len / 2);
                for (int char_idx = 0; char_idx < len; char_idx += 2) {
                    str += (wchar_t)*(uint16_t*)(buf + char_idx);
                }
            }

            if (is_new_buffer) {
                delete[] buf;
            }
        }
        else
        {
            str.clear();
            if (len != 0) {
                ret = false;
            }
        }
    }
    return ret;
}

//static
bool BinStreamHelper::ReadString32(IBinStream &strm, const char *&str, StringBlockedHeap &heap)
{
    str = nullptr;
    uint32_t len = 0;
    bool ret = strm.Read((char*)&len, sizeof(len));
    if (!ret) {
        return false;
    }

    if (len > 0)
    {
        char *buf = heap.New(len + 1);
        buf[len] = '\0';
        ret = strm.Read(buf, len);
        str = buf;
    }

    return ret;
}

//static
bool BinStreamHelper::ReadWString32(IBinStream &strm, const wchar_t *&str, StringBlockedHeap &heap)
{
    str = nullptr;
    uint32_t len = 0;
    bool ret = strm.Read((char*)&len, sizeof(len));
    if (!ret) {
        return false;
    }

    uint16_t iData = 0;
    if (len > 0)
    {
        len /= 2;
        wchar_t *wbuf = heap.NewWchar(len + 1);
        //ret = strm.Read((char*)wbuf, len*2);
        for (uint32_t idx = 0; ret && idx < len; idx++)
        {
            ret = strm.Read(iData);
            wbuf[idx] = iData;
        }
        wbuf[len] = L'\0';
        str = wbuf;
    }

    return ret;
}

//static
bool BinStreamHelper::ReadString16(IBinStream &strm, const char *&str, StringBlockedHeap &heap)
{
    str = nullptr;
    uint16_t len = 0;
    bool ret = strm.Read((char*)&len, sizeof(len));
    if (!ret) {
        return false;
    }

    if (len > 0)
    {
        char *buf = heap.New((uint32_t)len + 1);
        buf[len] = '\0';
        ret = strm.Read(buf, len);
        str = buf;
    }

    return ret;
}

//static
bool BinStreamHelper::ReadWString16(IBinStream &strm, const wchar_t *&str,
    StringBlockedHeap &heap, bool debugMode)
{
    str = nullptr;
    uint16_t len = 0;
    bool ret = strm.Read((char*)&len, sizeof(len));
    if (!ret) {
        return false;
    }

    uint16_t iData = 0;
    if (len > 0)
    {
        len /= 2;
        wchar_t *wbuf = heap.NewWchar((uint32_t)len + 1);
        wbuf[len] = L'\0';
        //ret = strm.Read((char*)wbuf, len*2);
        for (uint32_t idx = 0; ret && idx < len; idx++)
        {
            ret = strm.Read(iData);
            wbuf[idx] = iData;
        }

        if (debugMode) {
            LogKeyInfo(L"len: %u, str: %ls", wcslen(wbuf), wbuf);
        }
        str = wbuf;
    }

    return ret;
}

//static
bool BinStreamHelper::WriteString32(IBinStream &strm, const char *data, uint32_t data_len)
{
    if (data_len == UINT32_MAX) {
        data_len = data != nullptr ? (uint32_t)strlen(data) : 0;
    }

    bool ret = strm.Write((const char*)&data_len, sizeof(data_len));
    if (ret && data_len > 0) {
        ret = strm.Write(data, (size_t)data_len);
    }
    return ret;
}

//static
bool BinStreamHelper::WriteWString32(IBinStream &strm, const wchar_t *data, uint32_t data_len)
{
    if (data_len == UINT32_MAX) {
        data_len = data != nullptr ? (uint32_t)wcslen(data) : 0;
    }

    data_len *= 2;
    bool ret = strm.Write(data_len);
    if (ret && data_len > 0)
    {
        //ret = strm.Write((const char*)data, (size_t)data_len);
        for (uint32_t char_idx = 0; char_idx < data_len / 2; char_idx++) {
            strm.Write((uint16_t)data[char_idx]);
        }
    }
    return ret;
}

//static
bool BinStreamHelper::WriteString16(IBinStream &strm, const char *data, uint16_t data_len)
{
    if (data_len == UINT16_MAX) {
        data_len = data != nullptr ? (uint16_t)strlen(data) : 0;
    }

    bool ret = strm.Write((const char*)&data_len, sizeof(data_len));
    if (ret && data_len > 0) {
        ret = strm.Write(data, (size_t)data_len);
    }
    return ret;
}

//Please pay attention that wchar_t has 4 bytes on Linux while 2 bytes on Windows
//static
bool BinStreamHelper::WriteWString16(IBinStream &strm, const wchar_t *data, uint16_t data_len)
{
    if (data_len == UINT16_MAX) {
        data_len = data != nullptr ? (uint16_t)wcslen(data) : 0;
    }

    data_len *= 2;
    bool ret = strm.Write(data_len);
    if (ret && data_len > 0)
    {
        //ret = strm.Write((const char*)data, (size_t)data_len);
        for (uint16_t char_idx = 0; char_idx < data_len / 2; char_idx++) {
            strm.Write((uint16_t)data[char_idx]);
        }
    }
    return ret;
}

} //end of namespace
