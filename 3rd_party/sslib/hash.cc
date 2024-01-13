#include "hash.h"
#include <cstring>
#include "random.h"
#include "macro.h"
#include "string.h"

#if defined ENABLE_CRYPT_HASH
#   if defined _WIN32 || defined _WIN64
#       include <Windows.h>
#       include <Wincrypt.h>
#   elif defined __APPLE__ || defined Macintosh
#       include <CommonCrypto/CommonDigest.h>
#   else
#       include <openssl/sha.h>
#       include <openssl/crypto.h> 
#   endif //def _WIN32
#endif //ENABLE_CRYPT_HASH

namespace sslib
{

#if defined ENABLE_CRYPT_HASH

CryptHash::CryptHash()
{
#if defined _WIN32 || defined _WIN64
    HCRYPTPROV *handle = new HCRYPTPROV;
    CryptAcquireContext(handle, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT);
    core_data_ptr = handle;
#elif defined __APPLE__ || defined Macintosh
    CC_SHA256_CTX *ctx = new CC_SHA256_CTX;
    core_data_ptr = ctx;
#else
    SHA256_CTX *ctx = new SHA256_CTX;
    core_data_ptr = ctx;
#endif
}

CryptHash::~CryptHash()
{
#if defined _WIN32 || defined _WIN64
    if (core_data_ptr != nullptr)
    {
        auto *handle = (HCRYPTPROV*)core_data_ptr;
        CryptReleaseContext(*handle, 0);
        delete handle;
    }
    core_data_ptr = nullptr;
#elif defined __APPLE__ || defined Macintosh
    if (core_data_ptr != nullptr)
    {
        CC_SHA256_CTX *ctx = (CC_SHA256_CTX*)core_data_ptr;
        delete ctx;
    }
    core_data_ptr = nullptr;
#else
    if (core_data_ptr != nullptr)
    {
        SHA256_CTX *ctx = (SHA256_CTX*)core_data_ptr;
        delete ctx;
    }
    core_data_ptr = nullptr;
#endif
}

bool CryptHash::Hash(uint8_t *output, const uint8_t *input_data, uint32_t input_len)
{
    bool ret = true;
#if defined _WIN32 || defined _WIN64
    HCRYPTPROV *prov_handle = (HCRYPTPROV*)core_data_ptr;
    HCRYPTHASH hHash;
    //ret = !!CryptCreateHash(m_hProv, CALG_MD5, 0, 0, &hHash);
    ret = !!CryptCreateHash(*prov_handle, CALG_SHA_256, 0, 0, &hHash);
    if (!ret) {
        return false;
    }

    ret = CryptHashData(hHash, (const BYTE*)input_data, input_len, 0) != FALSE;
    DWORD dwHashLen = 32;
    if (ret) {
        ret = CryptGetHashParam(hHash, HP_HASHVAL, output, &dwHashLen, 0) != FALSE;
    }

    CryptDestroyHash(hHash);
#elif defined __APPLE__ || defined Macintosh
    //unsigned char digest[SHA256_DIGEST_LENGTH];
    if (Hash_Length != CC_SHA256_DIGEST_LENGTH) {
        return false;
    }

    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    CC_SHA256_Update(&ctx, input_data, input_len);
    CC_SHA256_Final(output, &ctx);
#else
    //unsigned char digest[SHA256_DIGEST_LENGTH];
    if (Hash_Length != SHA256_DIGEST_LENGTH) {
        return false;
    }

    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input_data, input_len);
    SHA256_Final(output, &ctx);
#endif
    return ret;
}

bool CryptHash::Hash(uint8_t *output, const std::string &input)
{
    return Hash(output, (const uint8_t*)input.c_str(), (UInt32)input.size());
}

bool CryptHash::Hash(string &output, const string &input)
{
    uint8_t output[Hash_Length];
    bool ret = Hash(output, (const uint8_t*)input.c_str(), (UInt32)input.size());
    output.assign((const char*)output, Hash_Length);
    return ret;
}

//static
RawString CryptHash::BuildCryptString(uint32_t len, uint64_t seed)
{
    Random rng(seed);
    RawString rs;
    rs.New(len);
    Macro_RetIf(rs, rs.data == nullptr);

    uint32_t iData = 0;
    for (uint32_t iCh = 0; iCh < len; iCh += 4)
    {
        iData = (uint32_t)rng.NextInt32();
        memcpy(rs.data + iCh, (const char*)&iData, min((uint32_t)4, len - iCh));
    }

    return rs;
}

//static
void CryptHash::ParseEncryptionSpec(uint32_t &len, uint64_t &seed,
    const wstring &spec_str)
{
    len = 0;
    seed = 0;
    vector<wstring> encryption_tokens;
    WString::Split(spec_str, encryption_tokens, L"-");
    if (encryption_tokens.size() > 1)
    {
        for (auto &str : encryption_tokens) {
            WString::Trim(str);
        }

        len = (uint32_t)String::ToInt32(encryption_tokens[0]);
        for (size_t idx = 1; idx < encryption_tokens.size(); idx++)
        {
            seed = (seed << 16);
            seed |= (uint32_t)String::ToInt32(encryption_tokens[idx]);
        }
    }
}

//static
void CryptHash::BuildCryptString(std::string &str, uint32_t len, uint64_t seed)
{
    str.clear();
    RawString rs = BuildCryptString(len, seed);
    if (rs.size > 0 && rs.data != nullptr)
    {
        str.assign(rs.data, rs.size);
        rs.Delete();
    }
}

//static
void CryptHash::PrintHashValue(std::string &str, const uint8_t *hashed_data, size_t hash_len)
{
    char hex_digits[] = "0123456789ABCDEF";
    str.resize(0);
    for (size_t idx = 0; idx < hash_len; idx++)
    {
        str += hex_digits[((hashed_data[idx] & 0xF0) >> 4)];
        str += hex_digits[(hashed_data[idx] & 0x0F)];
        if (idx % 4 == 3 && idx != hash_len - 1) {
            str += '-';
        }
    }
}

#endif //ENABLE_CRYPT_HASH

} //end of namespace
