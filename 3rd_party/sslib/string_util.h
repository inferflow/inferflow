#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include "prime_types.h"
#include "number.h"
#include "macro.h"

#ifndef CP_UTF8
#define CP_UTF8 65001
#endif

namespace sslib
{

enum class CodePageId
{
    Unknown = 0,
    GBK = 936,
    Big5 = 950,
    CP1251 = 1251,
    CP1252 = 1252,
    Latin01 = 28591,
    Latin15 = 28605,
    Utf8 = CP_UTF8
};

struct CharOffsetMap
{
    std::vector<uint32_t> wcs_to_mbs;
    std::map<uint32_t, uint32_t> mbs_to_wcs;

    void Clear()
    {
        wcs_to_mbs.clear();
        mbs_to_wcs.clear();
    }
};

class CodePageData
{
public:
    bool Load(const std::string &file_path);

    bool ToUcs2(std::wstring &wstr, const char *str, uint32_t len = UINT32_MAX) const;
    bool ToUcs2(std::u16string &target, const char *source, uint32_t source_len = UINT32_MAX) const;
    bool FromUcs2(std::string &target, const wchar_t *source, uint32_t source_len = UINT32_MAX) const;
    bool FromUcs2(std::string &target, const char16_t *source, uint32_t source_len = UINT32_MAX) const;

    bool FromUcs2(std::string &target, CharOffsetMap *offset_map,
        const wchar_t *source, uint32_t source_len = UINT32_MAX) const;

protected:
    std::map<uint32_t, uint16_t> map1_, map2_;
};

class CodePageMap
{
public:
    CodePageMap();
    virtual ~CodePageMap();

    void Clear();
    bool Load(const std::string &data_dir);

    const CodePageData* Get(CodePageId cp) const;

protected:
    std::map<CodePageId, CodePageData*> map_data_;
};

class StringUtil
{
public:
    static bool Init(const std::string &data_dir);

    static size_t strlen16(const char16_t *str);
    static size_t StrLen16(const char16_t *str);
    static size_t strlen32(const char16_t *str);
    static size_t StrLen32(const char16_t *str);

    static bool WildcardMatch(const std::string &str, const std::string &pattern);
    static bool WildcardMatch(const char *str, const char *pattern);

    static bool Ucs2ToUtf8(std::string &target, const std::wstring &source);
    static bool Ucs2ToUtf8(std::string &target, CharOffsetMap *offset_map,
        const wchar_t *source, uint32_t source_len = UINT32_MAX);
    static bool Ucs2ToUtf8(std::string &target, const wchar_t *source, uint32_t source_len = UINT32_MAX);
    static bool Ucs2ToUtf8(std::string &target, const std::u16string &source);
    static bool Ucs2ToUtf8(std::string &target, const char16_t *source, uint32_t source_len = UINT32_MAX);

    static bool Utf8ToUcs2(std::wstring &target, const std::string &source,
        uint32_t max_big_char_count = 0);
    static bool Utf8ToUcs2(std::wstring &target, const char *source,
        uint32_t source_len = UINT32_MAX, uint32_t max_big_char_count = 0);
    static bool Utf8ToUcs2(std::u16string &target, const std::string &source,
        uint32_t max_big_char_count);
    static bool Utf8ToUcs2(std::u16string &target, const char *source,
        uint32_t source_len = UINT32_MAX, uint32_t max_big_char_count = 0);

    static bool Utf8ToWideStr(std::wstring &wstr, const std::string &str,
        uint32_t max_big_char_count = 0);
    static bool Utf8ToWideStr(std::wstring &wstr, const char *str,
        uint32_t len = UINT32_MAX, uint32_t max_big_char_count = 0);
    static std::wstring Utf8ToWideStr(const std::string &str,
        uint32_t max_big_char_count = 0);
    static std::wstring Utf8ToWideStr(const char *str, uint32_t len = UINT32_MAX,
        uint32_t max_big_char_count = 0);

    static bool ToUtf8(std::string &str, const std::wstring &wstr);
    static bool ToUtf8(std::string &str, const wchar_t *wstr, uint32_t len = UINT32_MAX);
    static std::string ToUtf8(const std::wstring &wstr);
    static std::string ToUtf8(const wchar_t *wstr, uint32_t len = UINT32_MAX);

    static std::wstring ConsoleEncodingToWideStr(const std::string &str);
    static std::string ToConsoleEncoding(const std::wstring &wstr);
    static std::string ToConsoleEncoding(const wchar_t *wstr, uint32_t len = UINT32_MAX);

    static bool ToWideStr(std::wstring &wstr, CodePageId source_code_page,
        const std::string &str, uint32_t max_big_char_count = 0);
    static bool ToWideStr(std::wstring &wstr, CodePageId source_code_page,
        const char *str, uint32_t len = UINT32_MAX,
        uint32_t max_big_char_count = 0);
    static std::wstring ToWideStr(CodePageId source_code_page,
        const std::string &str, uint32_t max_big_char_count = 0);
    static std::wstring ToWideStr(CodePageId source_code_page,
        const char *str, uint32_t len = UINT32_MAX,
        uint32_t max_big_char_count = 0);

    static bool ToUcs2(std::u16string &target, CodePageId source_code_page,
        const std::string &source, uint32_t max_big_char_count = 0);
    static bool ToUcs2(std::u16string &target, CodePageId source_code_page,
        const char *source, uint32_t source_len = UINT32_MAX,
        uint32_t max_big_char_count = 0);
    static std::u16string ToUcs2(CodePageId source_code_page,
        const std::string &source, uint32_t max_big_char_count = 0);
    static std::u16string ToUcs2(CodePageId source_code_page,
        const char *source, uint32_t source_len = UINT32_MAX,
        uint32_t max_big_char_count = 0);

    static bool ToMultibyteStr(std::string &str, CodePageId target_code_page,
        const std::wstring &wstr);
    static bool ToMultibyteStr(std::string &str, CharOffsetMap *offset_map,
        CodePageId target_code_page, const std::wstring &wstr);
    static bool ToMultibyteStr(std::string &str, CodePageId target_code_page,
        const wchar_t *wstr, uint32_t len = UINT32_MAX);
    static bool ToMultibyteStr(std::string &target, CodePageId target_code_page,
        const std::u16string &source);
    static bool ToMultibyteStr(std::string &target, CodePageId target_code_page,
        const char16_t *source, uint32_t source_len = UINT32_MAX);
    static std::string ToMultibyteStr(CodePageId target_code_page,
        const std::wstring &wstr);
    static std::string ToMultibyteStr(CodePageId target_code_page,
        const wchar_t *wstr, uint32_t len = UINT32_MAX);

    static bool LatinToUnicode(std::wstring &target, uint32_t part_id, const std::string &source);
    static bool LatinToUnicode(std::wstring &target, uint32_t part_id,
        const char *source, uint32_t source_len = UINT32_MAX);
    static std::wstring LatinToUnicode(uint32_t part_id, const std::string &source);
    static bool LatinToUcs2(std::u16string &target, uint32_t part_id, const std::string &source);
    static bool LatinToUcs2(std::u16string &target, uint32_t part_id,
        const char *source, uint32_t source_len = UINT32_MAX);

    static bool CP1252ToUnicode(std::wstring &wstr, const std::string &str);
    static bool CP1252ToUnicode(std::wstring &wstr, const char *str,
        uint32_t len = UINT32_MAX);
    static std::wstring CP1252ToUnicode(const std::string &str);
    static bool CP1252ToUcs2(std::u16string &target, const std::string &source);
    static bool CP1252ToUcs2(std::u16string &target, const char *source,
        uint32_t source_len = UINT32_MAX);

#ifdef _WIN32
    // Convert a wide string to a mutibyte string
    static bool WCS2MBS(
        const std::wstring &wstr,
        std::string& str,
        uint32_t code_page // The code page of multibyte
    );

    static bool WCS2MBS(
        const wchar_t *wcs,
        size_t wstr_len,
        std::string& str,
        uint32_t code_page // The code page of multibyte
    );

    // Convert a mutibyte string to a wide string
    static bool MBS2WCS(
        const std::string &str,
        std::wstring &wstr,
        uint32_t code_page // The code page of multibyte
    );

    static bool MBS2WCS(
        const char *str,
        size_t str_len,
        std::wstring &wstr,
        uint32_t code_page // The code page of multibyte
    );
#endif

protected:
    static CodePageMap code_page_map_ HiddenAttribute;
};

class CodePageConverter_GBK
{
public:
    static bool LoadData(const std::string &file_path);

    static bool ToWCS(std::wstring &wstr, const std::string &str);
    static bool ToWCS(std::wstring &wstr, const char *str, uint32_t len = UINT32_MAX);
    static std::wstring ToWCS(const std::string &str);
    static std::wstring ToWCS(const char *str, uint32_t len = UINT32_MAX);
    static bool ToMBS(std::string &str, const std::wstring &wstr);
    static bool ToMBS(std::string &str, const wchar_t *wstr, uint32_t len = UINT32_MAX);
    static std::string ToMBS(const std::wstring &wstr);
    static std::string ToMBS(const wchar_t *wstr, uint32_t len = UINT32_MAX);

private:
    static CodePageData data_ HiddenAttribute;
};

} //end of namespace
