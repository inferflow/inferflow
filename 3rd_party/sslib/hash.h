#pragma once

#include <string>
#include "raw_array.h"
#include "macro.h"

namespace sslib
{

class Hash
{
public:
    Hash() {};
    virtual ~Hash() {};

    static inline int HashCode(const char *str, int len = -1);
    static inline int HashCode(const std::string &str);
    static inline uint32_t HashCodeUInt32(const char *str, int len = -1);
    static inline uint32_t HashCodeUInt32(const std::string &str);
    static inline uint32_t HashCodeUInt32(const wchar_t *wstr, int len = -1);
    static inline uint32_t HashCodeUInt32(const std::wstring &wstr);
    static inline uint64_t HashCodeUInt64(const char *str, int len = -1);
    static inline uint64_t HashCodeUInt64(const std::string &str);
    static inline uint64_t HashCodeUInt64(const wchar_t *wstr, int len = -1);
    static inline uint64_t HashCodeUInt64(const std::wstring &wstr);
}; //class Hash

int Hash::HashCode(const char *str, int len)
{
    int result = 0;
    if (len < 0) len = (int)strlen(str);

    for (int i = 0; i < len; i++)
    {
        result = 31 * result + str[i];
    }

    return result;
}

int Hash::HashCode(const std::string &str)
{
    int result = 0;
    std::string::size_type len = str.size();

    for (std::string::size_type i = 0; i < len; i++)
    {
        result = 31 * result + str[i];
    }

    return result;
}

uint32_t Hash::HashCodeUInt32(const char *str, int len)
{
    uint32_t result = 0;
    if (len < 0) len = (int)strlen(str);

    for (int i = 0; i < len; i++)
    {
        result = 31 * result + str[i];
    }

    return result;
}

uint32_t Hash::HashCodeUInt32(const std::string &str)
{
    uint32_t result = 0;
    std::string::size_type len = str.size();

    for (std::string::size_type i = 0; i < len; i++)
    {
        result = 31 * result + str[i];
    }

    return result;
}

uint32_t Hash::HashCodeUInt32(const wchar_t *wstr, int len)
{
    uint32_t result = 0;
    if (len < 0) {
        len = (int)wcslen(wstr);
    }

    for (int i = 0; i < len; i++) {
        result = 31 * result + (uint16_t)wstr[i];
    }

    return result;
}

uint32_t Hash::HashCodeUInt32(const std::wstring &wstr)
{
    return HashCodeUInt32(wstr.c_str(), (int)wstr.size());
}

uint64_t Hash::HashCodeUInt64(const char *str, int len)
{
    uint64_t result = 0;
    if (len < 0) len = (int)strlen(str);

    for (int i = 0; i < len; i++)
    {
        result = 31 * result + str[i];
    }

    return result;
}

uint64_t Hash::HashCodeUInt64(const std::string &str)
{
    return HashCodeUInt64(str.c_str(), (int)str.size());
}

uint64_t Hash::HashCodeUInt64(const wchar_t *wstr, int len)
{
    uint64_t result = 0;
    if (len < 0) {
        len = (int)wcslen(wstr);
    }

    for (int i = 0; i < len; i++) {
        result = 31 * result + (uint16_t)wstr[i];
    }

    return result;
}

uint64_t Hash::HashCodeUInt64(const std::wstring &wstr)
{
    return HashCodeUInt64(wstr.c_str(), (int)wstr.size());
}

///////////////////////////////////////////////////////////////////////////////
// class CryptHash

#if defined ENABLE_CRYPT_HASH

class DefaultAttribute CryptHash
{
public:
    const static uint16_t Hash_Length = 32;

public:
    CryptHash();
    virtual ~CryptHash();

    //output: hashed data
    bool Hash(uint8_t *output, const uint8_t *input_data, uint32_t input_len);
    bool Hash(uint8_t *output, const std::string &input);
    bool Hash(std::string &output, const std::string &input);

    static void ParseEncryptionSpec(uint32_t &len, uint64_t &seed,
        const std::wstring &spec_str);
    static RawString BuildCryptString(uint32_t len, uint64_t seed);
    static void BuildCryptString(std::string &str, uint32_t len, uint64_t seed);

    static void PrintHashValue(std::string &str, const uint8_t *hashed_data,
        size_t hash_len);

protected:
    void *core_data_ptr = nullptr;
};

#endif //ENABLE_CRYPT_HASH

} //end of namespace

