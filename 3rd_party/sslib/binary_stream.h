#pragma once

#include <string>
#include <cstring>
#include <vector>
#include "prime_types.h"

namespace sslib
{

class IBinaryStream
{
public:
    enum SeekWayEnum
    {
        SEEK_WAY_ABS    = 0, //absolute
        SEEK_WAY_BEGIN  = 1,
        SEEK_WAY_END    = 2
    };

public:
    virtual bool Read(char *buf, size_t size, void *params = 0) = 0;
    virtual bool Write(const char *buf, size_t size, void *params = 0) = 0;

    virtual bool IsGood() const {
        return true;
    }

    virtual bool SeekRd(uint64_t offset, uint32_t seek_way = 0) {
        (void)offset; (void)seek_way; return false;
    }
    virtual bool SeekWr(uint64_t offset, uint32_t seek_way = 0) {
        (void)offset; (void)seek_way; return false;
    }
    virtual uint64_t TellRd() const { return 0; };
    virtual uint64_t TellWr() const { return 0; };

    template <class T>
    inline bool Read(T &data) {
        return Read((char*)&data, sizeof(data));
    }

    template <class T>
    inline bool Write(const T &data) {
        return Write((const char*)&data, sizeof(data));
    }

    template <class T>
    bool ReadVector(std::vector<T> &vec)
    {
        uint32_t num = 0;
        T value;
        bool ret = Read(num);
        for (uint32_t idx = 0; ret && idx < num; idx++)
        {
            ret = ret && Read(value);
            vec.push_back(value);
        }

        return ret && IsGood();
    }

    template <class T>
    bool WriteVector(const std::vector<T> &vec)
    {
        uint32_t num = (uint32_t)vec.size();
        bool ret = Write(num);
        for (size_t idx = 0; ret && idx < vec.size(); idx++)
        {
            ret = ret && Write(vec[idx]);
        }

        return ret && IsGood();
    }

    virtual bool GetLine(std::string &line_str, char delim = '\n')
    {
        (void)line_str; (void)delim;
        return false;
    };
    virtual bool GetLine(std::wstring &line_str, wchar_t delim = L'\n')
    {
        (void)line_str; (void)delim;
        return false;
    };

    virtual ~IBinaryStream() {};
};

typedef IBinaryStream IBinStream;

} //end of namespace
