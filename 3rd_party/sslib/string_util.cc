#include "string_util.h"
#include <cstring>
#include <sstream>
#include <cstdlib>
#ifdef _WIN32
#   include <Windows.h>
#   include <codecvt>
#   include <locale>
#endif //def _WIN32
#include "binary_file_stream.h"
#include "macro.h"
#include "log.h"

using namespace std;

namespace sslib
{

bool CodePageData::Load(const string &file_path)
{
    map1_.clear();
    map2_.clear();

    BinaryFileStream fs;
    bool ret = fs.OpenForRead(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to open the file: %s", file_path.c_str()));

    IBinaryStream &reader = fs;
    uint16_t count1 = 0, count2 = 0, c1 = 0, c2 = 0;
    reader.Read(count1);
    reader.Read(count2);
    for (uint16_t idx = 0; idx < count1; idx++)
    {
        reader.Read(c1);
        reader.Read(c2);
        map1_[c1] = c2;
        map2_[c2] = c1;
    }
    for (uint16_t idx = 0; idx < count2; idx++)
    {
        reader.Read(c2);
        reader.Read(c1);
        map2_[c2] = c1;
    }

    return ret;
}

bool CodePageData::ToUcs2(std::wstring &wstr, const char *str, uint32_t len) const
{
    if (str == nullptr || *str == '\0')
    {
        wstr.clear();
        return true;
    }

    if (len == UINT32_MAX) {
        len = (uint32_t)strlen(str);
    }

    bool ret = true;
    uint16_t c1 = 0;
    wstr.clear();
    for (uint32_t idx = 0; idx < len; idx++)
    {
        UInt8 b1 = str[idx];
        c1 = b1;
        if ((b1 & 0x80) != 0)
        {
            idx++;
            c1 = (c1 << 8) | (UInt8)str[idx];
        }

        auto iter = map1_.find(c1);
        if (iter == map1_.end())
        {
            ret = false;
            wstr += c1;
        }
        else
        {
            wstr += iter->second;
        }
    }

    return ret;
}

bool CodePageData::ToUcs2(std::u16string &target, const char *source, uint32_t source_len) const
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)strlen(source);
    }

    bool ret = true;
    uint16_t c1 = 0;
    for (uint32_t idx = 0; idx < source_len; idx++)
    {
        UInt8 b1 = source[idx];
        c1 = b1;
        if ((b1 & 0x80) != 0)
        {
            idx++;
            c1 = (c1 << 8) | (UInt8)source[idx];
        }

        auto iter = map1_.find(c1);
        if (iter == map1_.end())
        {
            ret = false;
            target += c1;
        }
        else
        {
            target += iter->second;
        }
    }

    return ret;
}

bool CodePageData::FromUcs2(std::string &target, const wchar_t *source, uint32_t source_len) const
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)wcslen(source);
    }

    bool ret = true;
    uint16_t c1 = 0, c2 = 0;
    for (uint32_t idx = 0; idx < source_len; idx++)
    {
        c2 = source[idx];
        auto iter = map2_.find(c2);
        if (iter == map2_.end())
        {
            ret = false;
            target += (char)0x3F;
        }
        else
        {
            c1 = iter->second;
            if (c1 <= 0xFF)
            {
                target += (char)c1;
            }
            else
            {
                target += (char)(c1 >> 8);
                target += (char)(c1 & 0xFF);
            }
        }
    }

    return ret;
}

bool CodePageData::FromUcs2(std::string &target, CharOffsetMap *offset_map,
    const wchar_t *source, uint32_t source_len) const
{
    target.clear();
    if (offset_map != nullptr) {
        offset_map->Clear();
    }
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)wcslen(source);
    }

    bool ret = true;
    uint16_t c1 = 0, c2 = 0;
    for (uint32_t idx = 0; idx < source_len; idx++)
    {
        if (offset_map != nullptr)
        {
            offset_map->wcs_to_mbs.push_back((uint32_t)target.size());
            offset_map->mbs_to_wcs[(uint32_t)target.size()] = idx;
        }

        c2 = source[idx];
        auto iter = map2_.find(c2);
        if (iter == map2_.end())
        {
            ret = false;
            target += (char)0x3F;
        }
        else
        {
            c1 = iter->second;
            if (c1 <= 0xFF)
            {
                target += (char)c1;
            }
            else
            {
                target += (char)(c1 >> 8);
                target += (char)(c1 & 0xFF);
            }
        }
    }

    return ret;
}

bool CodePageData::FromUcs2(std::string &target, const char16_t *source, uint32_t source_len) const
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)StringUtil::StrLen16(source);
    }

    bool ret = true;
    uint16_t c1 = 0, c2 = 0;
    for (uint32_t idx = 0; idx < source_len; idx++)
    {
        c2 = (uint16_t)source[idx];
        auto iter = map2_.find(c2);
        if (iter == map2_.end())
        {
            ret = false;
            target += (char)0x3F;
        }
        else
        {
            c1 = iter->second;
            if (c1 <= 0xFF)
            {
                target += (char)c1;
            }
            else
            {
                target += (char)(c1 >> 8);
                target += (char)(c1 & 0xFF);
            }
        }
    }

    return ret;
}

CodePageMap::CodePageMap()
{
}

CodePageMap::~CodePageMap()
{
    Clear();
}

void CodePageMap::Clear()
{
    for (auto iter = map_data_.begin(); iter != map_data_.end(); iter++)
    {
        if (iter->second != nullptr)
        {
            delete iter->second;
            iter->second = nullptr;
        }
    }
    map_data_.clear();
}

bool CodePageMap::Load(const string &data_dir)
{
    map<CodePageId, string> cp_map;
    cp_map[CodePageId::GBK] = "gbk";
    cp_map[CodePageId::Big5] = "big5";

    bool ret = true;
    string file_path;
    for (auto iter = cp_map.begin(); iter != cp_map.end(); iter++)
    {
        CodePageId code_page_id = iter->first;
        auto *cp_data = new CodePageData;
        file_path = data_dir + "code_cvt_" + iter->second + ".dat";
        ret = cp_data->Load(file_path);
        if (!ret)
        {
            delete cp_data;
            LogError("Failed to load the data of code page %u", (uint32_t)code_page_id);
            return false;
        }

        map_data_[code_page_id] = cp_data;
    }

    return ret;
}

const CodePageData* CodePageMap::Get(CodePageId cp) const
{
    auto iter = map_data_.find(cp);
    return iter != map_data_.end() ? iter->second : nullptr;
}

CodePageMap StringUtil::code_page_map_;

//static
bool StringUtil::Init(const string &data_dir)
{
    CodePageConverter_GBK::LoadData(data_dir + "code_cvt_gbk.dat");
    bool ret = code_page_map_.Load(data_dir);
    return ret;
}

size_t StringUtil::strlen16(const char16_t *str)
{
    return StrLen16(str);
}

//static
size_t StringUtil::StrLen16(const char16_t *str)
{
    if (str == nullptr) {
        return 0;
    }

    const char16_t *cursor = str;
    for (; *cursor != u'\0'; ++cursor) {
    }
    return cursor - str;
}

size_t StringUtil::strlen32(const char16_t *str)
{
    return StrLen32(str);
}

//static
size_t StringUtil::StrLen32(const char16_t *str)
{
    if (str == nullptr) {
        return 0;
    }

    const char16_t *cursor = str;
    for (; *cursor != U'\0'; ++cursor) {
    }
    return cursor - str;
}

//static
bool StringUtil::Utf8ToWideStr(std::wstring &out, const std::string &str,
    uint32_t max_big_char_count)
{
#ifdef _WIN32
    //return MBS2WCS(str, out, CP_UTF8);
    return Utf8ToUcs2(out, str, max_big_char_count);
#else
    return Utf8ToUcs2(out, str, max_big_char_count);
    //std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> convert;
    //out.assign((const wchar_t*)convert.from_bytes(str).c_str());
    //return true;
#endif
}

//static
bool StringUtil::Utf8ToWideStr(std::wstring &out, const char *str, uint32_t len,
    uint32_t max_big_char_count)
{
#ifdef _WIN32
    //return MBS2WCS(str, len, out, CP_UTF8);
    return Utf8ToUcs2(out, str, len, max_big_char_count);
#else
    return Utf8ToUcs2(out, str, len, max_big_char_count);
    //string src(str, len);
    //std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> convert;
    //out.assign((const wchar_t*)convert.from_bytes(src).c_str());
    //return true;
#endif
}

//static
wstring StringUtil::Utf8ToWideStr(const string &str,
    uint32_t max_big_char_count)
{
    wstring out;
    Utf8ToWideStr(out, str, max_big_char_count);
    return out;
}

//static
wstring StringUtil::Utf8ToWideStr(const char *str, uint32_t len,
    uint32_t max_big_char_count)
{
    wstring out;
    Utf8ToWideStr(out, str, len, max_big_char_count);
    return out;
}

//static
std::wstring StringUtil::ConsoleEncodingToWideStr(const std::string &str)
{
    wstring wstr;
#ifdef _WIN32
    MBS2WCS(str, wstr, CP_OEMCP);
#else
    Utf8ToUcs2(wstr, str);
#endif
    return wstr;
}

//static
bool StringUtil::ToUtf8(std::string &str, const std::wstring &wstr)
{
    return ToUtf8(str, wstr.c_str(), (uint32_t)wstr.length());
}

//static
bool StringUtil::ToUtf8(std::string &out, const wchar_t *wstr, uint32_t len)
{
#ifdef _WIN32
    return Ucs2ToUtf8(out, wstr, len);
#else
    return Ucs2ToUtf8(out, wstr, len);
#endif
}

//static
string StringUtil::ToUtf8(const wstring &wstr)
{
    string str;
    ToUtf8(str, wstr);
    return str;
}

//static
string StringUtil::ToUtf8(const wchar_t *wstr, uint32_t len)
{
    string str;
    ToUtf8(str, wstr, len);
    return str;
}

//static
std::string StringUtil::ToConsoleEncoding(const std::wstring &wstr)
{
#ifdef _WIN32
    string str;
    WCS2MBS(wstr, str, CP_OEMCP);
    return str;
#else
    return ToUtf8(wstr);
#endif
}

//static
std::string StringUtil::ToConsoleEncoding(const wchar_t *wstr, uint32_t len)
{
#ifdef _WIN32
    string str;
    WCS2MBS(wstr, len, str, CP_OEMCP);
    return str;
#else
    return ToUtf8(wstr, len);
#endif
}

//static
bool StringUtil::ToWideStr(std::wstring &wstr, CodePageId source_code_page,
    const std::string &str, uint32_t max_big_char_count)
{
    return ToWideStr(wstr, source_code_page, str.c_str(), (uint32_t)str.size(),
        max_big_char_count);
}

//static
bool StringUtil::ToWideStr(std::wstring &wstr, CodePageId source_code_page,
    const char *str, uint32_t len, uint32_t max_big_char_count)
{
    wstr.clear();

    switch (source_code_page)
    {
    case CodePageId::Utf8:
        return Utf8ToWideStr(wstr, str, len, max_big_char_count);
        break;
    case CodePageId::Latin01:
        return LatinToUnicode(wstr, 1, str, len);
        break;
    case CodePageId::Latin15:
        return LatinToUnicode(wstr, 15, str, len);
        break;
    case CodePageId::CP1252:
        return CP1252ToUnicode(wstr, str, len);
        break;
    default:
        break;
    }

    const CodePageData *cp_data = code_page_map_.Get(source_code_page);
    if (cp_data == nullptr) {
        return false;
    }

    return cp_data->ToUcs2(wstr, str, len);
}

//static
std::wstring StringUtil::ToWideStr(CodePageId source_code_page,
    const std::string &str, uint32_t max_big_char_count)
{
    wstring wstr;
    ToWideStr(wstr, source_code_page, str.c_str(), (uint32_t)str.size(), max_big_char_count);
    return wstr;
}

//static
std::wstring StringUtil::ToWideStr(CodePageId source_code_page,
    const char *str, uint32_t len, uint32_t max_big_char_count)
{
    wstring wstr;
    ToWideStr(wstr, source_code_page, str, len, max_big_char_count);
    return wstr;
}

//static
bool StringUtil::ToUcs2(std::u16string &target, CodePageId source_code_page,
    const std::string &source, uint32_t max_big_char_count)
{
    return ToUcs2(target, source_code_page, source.c_str(), (uint32_t)source.size(),
        max_big_char_count);
}

//static
bool StringUtil::ToUcs2(std::u16string &target, CodePageId source_code_page,
    const char *source, uint32_t source_len, uint32_t max_big_char_count)
{
    target.clear();
    switch (source_code_page)
    {
    case CodePageId::Utf8:
        return Utf8ToUcs2(target, source, source_len, max_big_char_count);
        break;
    case CodePageId::Latin01:
        return LatinToUcs2(target, 1, source, source_len);
        break;
    case CodePageId::Latin15:
        return LatinToUcs2(target, 15, source, source_len);
        break;
    case CodePageId::CP1252:
        return CP1252ToUcs2(target, source, source_len);
        break;
    default:
        break;
    }

    const CodePageData *cp_data = code_page_map_.Get(source_code_page);
    if (cp_data == nullptr) {
        return false;
    }

    return cp_data->ToUcs2(target, source, source_len);
}

//static
std::u16string StringUtil::ToUcs2(CodePageId source_code_page,
    const std::string &str, uint32_t max_big_char_count)
{
    u16string wstr;
    ToUcs2(wstr, source_code_page, str.c_str(), (uint32_t)str.size(), max_big_char_count);
    return wstr;
}

//static
std::u16string StringUtil::ToUcs2(CodePageId source_code_page,
    const char *source, uint32_t source_len, uint32_t max_big_char_count)
{
    u16string wstr;
    ToUcs2(wstr, source_code_page, source, source_len, max_big_char_count);
    return wstr;
}

//static
bool StringUtil::ToMultibyteStr(std::string &str, CodePageId target_code_page,
    const std::wstring &wstr)
{
    return ToMultibyteStr(str, target_code_page, wstr.c_str(), (uint32_t)wstr.size());
}

//static
bool StringUtil::ToMultibyteStr(std::string &str, CharOffsetMap *offset_map,
    CodePageId target_code_page, const std::wstring &wstr)
{
    str.clear();

    switch (target_code_page)
    {
    case CodePageId::Utf8:
        return Ucs2ToUtf8(str, offset_map, wstr.c_str(), (uint32_t)wstr.size());
        break;
    default:
        break;
    }

    const CodePageData *cp_data = code_page_map_.Get(target_code_page);
    if (cp_data == nullptr) {
        return false;
    }

    return cp_data->FromUcs2(str, offset_map, wstr.c_str(), (uint32_t)wstr.size());
}

//static
bool StringUtil::ToMultibyteStr(std::string &str, CodePageId target_code_page,
    const wchar_t *wstr, uint32_t len)
{
    str.clear();

    switch (target_code_page)
    {
    case CodePageId::Utf8:
        return ToUtf8(str, wstr, len);
        break;
    default:
        break;
    }

    const CodePageData *cp_data = code_page_map_.Get(target_code_page);
    if (cp_data == nullptr) {
        return false;
    }

    return cp_data->FromUcs2(str, wstr, len);
}

//static
bool StringUtil::ToMultibyteStr(std::string &target,
    CodePageId target_code_page,
    const std::u16string &source)
{
    return ToMultibyteStr(target, target_code_page,
        source.c_str(), (uint32_t)source.size());
}

//static
bool StringUtil::ToMultibyteStr(std::string &target,
    CodePageId target_code_page,
    const char16_t *source, uint32_t source_len)
{
    target.clear();

    switch (target_code_page)
    {
    case CodePageId::Utf8:
        return Ucs2ToUtf8(target, source, source_len);
        break;
    default:
        break;
    }

    const CodePageData *cp_data = code_page_map_.Get(target_code_page);
    if (cp_data == nullptr) {
        return false;
    }

    return cp_data->FromUcs2(target, source, source_len);
}

//static
std::string StringUtil::ToMultibyteStr(CodePageId target_code_page,
    const std::wstring &wstr)
{
    string str;
    ToMultibyteStr(str, target_code_page, wstr.c_str(), (uint32_t)wstr.size());
    return str;
}

//static
std::string StringUtil::ToMultibyteStr(CodePageId target_code_page,
    const wchar_t *wstr, uint32_t len)
{
    string str;
    ToMultibyteStr(str, target_code_page, wstr, len);
    return str;
}

//static
bool StringUtil::LatinToUnicode(wstring &wstr, uint32_t part_id, const string &str)
{
    return LatinToUnicode(wstr, part_id, str.c_str(), (uint32_t)str.size());
}

//static
bool StringUtil::LatinToUcs2(std::u16string &target, uint32_t part_id, const std::string &source)
{
    return LatinToUcs2(target, part_id, source.c_str(), (uint32_t)source.size());
}

//static
bool StringUtil::LatinToUnicode(std::wstring &wstr, uint32_t part_id,
    const char *str, uint32_t len)
{
    wstr.clear();
    if (str == nullptr || len == 0) {
        return true;
    }

    if (len == UINT32_MAX) {
        len = (uint32_t)strlen(str);
    }

    wstr.reserve(len);
    for (uint32_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        uint16_t ch = (uint16_t)(UInt8)str[ch_idx];
        if (part_id == 15)
        {
            switch (ch)
            {
            case 0x0A4:
                ch = 0x20AC;
                break;
            case 0x0A6:
                ch = 0x0160;
                break;
            case 0x0A8:
                ch = 0x0161;
                break;
            case 0x0B4:
                ch = 0x017D;
                break;
            case 0x0B8:
                ch = 0x017E;
                break;
            case 0x0BC:
                ch = 0x0152;
                break;
            case 0x0BD:
                ch = 0x0153;
                break;
            case 0x0BE:
                ch = 0x0178;
                break;
            default:
                break;
            }
        }

        wstr += (wchar_t)ch;
    }

    return true;
}

//static
bool StringUtil::LatinToUcs2(std::u16string &target, uint32_t part_id,
    const char *source, uint32_t source_len)
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)strlen(source);
    }

    target.reserve(source_len);
    for (uint32_t ch_idx = 0; ch_idx < source_len; ch_idx++)
    {
        uint16_t ch = (uint16_t)(UInt8)source[ch_idx];
        if (part_id == 15)
        {
            switch (ch)
            {
            case 0x0A4:
                ch = 0x20AC;
                break;
            case 0x0A6:
                ch = 0x0160;
                break;
            case 0x0A8:
                ch = 0x0161;
                break;
            case 0x0B4:
                ch = 0x017D;
                break;
            case 0x0B8:
                ch = 0x017E;
                break;
            case 0x0BC:
                ch = 0x0152;
                break;
            case 0x0BD:
                ch = 0x0153;
                break;
            case 0x0BE:
                ch = 0x0178;
                break;
            default:
                break;
            }
        }

        target += (char16_t)ch;
    }

    return true;
}

wstring StringUtil::LatinToUnicode(uint32_t part_id, const string &str)
{
    wstring wstr;
    LatinToUnicode(wstr, part_id, str.c_str(), (uint32_t)str.size());
    return wstr;
}

bool StringUtil::CP1252ToUnicode(std::wstring &wstr, const std::string &str)
{
    return CP1252ToUnicode(wstr, str.c_str(), (uint32_t)str.size());
}

//static
bool StringUtil::CP1252ToUcs2(std::u16string &target, const std::string &source)
{
    return CP1252ToUcs2(target, source.c_str(), (uint32_t)target.size());
}

bool StringUtil::CP1252ToUnicode(std::wstring &wstr, const char *str, uint32_t len)
{
    wstr.clear();
    if (str == nullptr || len == 0) {
        return true;
    }

    if (len == UINT32_MAX) {
        len = (uint32_t)strlen(str);
    }

    wstr.reserve(len);
    for (uint32_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        uint16_t ch = (uint16_t)(UInt8)str[ch_idx];
        switch (ch)
        {
        case 0x080:
            ch = 0x20AC;
            break;
        case 0x082:
            ch = 0x201A;
            break;
        case 0x083:
            ch = 0x0192;
            break;
        case 0x084:
            ch = 0x201E;
            break;
        case 0x085:
            ch = 0x2026;
            break;
        case 0x086:
            ch = 0x2020;
            break;
        case 0x087:
            ch = 0x2021;
            break;
        case 0x088:
            ch = 0x02C6;
            break;
        case 0x089:
            ch = 0x2030;
            break;
        case 0x08A:
            ch = 0x0160;
            break;
        case 0x08B:
            ch = 0x2039;
            break;
        case 0x08C:
            ch = 0x0152;
            break;
        case 0x08E:
            ch = 0x017D;
            break;
        case 0x091:
            ch = 0x2018;
            break;
        case 0x092:
            ch = 0x2019;
            break;
        case 0x093:
            ch = 0x201C;
            break;
        case 0x094:
            ch = 0x201D;
            break;
        case 0x095:
            ch = 0x2022;
            break;
        case 0x096:
            ch = 0x2013;
            break;
        case 0x097:
            ch = 0x2014;
            break;
        case 0x098:
            ch = 0x02DC;
            break;
        case 0x099:
            ch = 0x2122;
            break;
        case 0x09A:
            ch = 0x0161;
            break;
        case 0x09B:
            ch = 0x203A;
            break;
        case 0x09C:
            ch = 0x0153;
            break;
        case 0x09E:
            ch = 0x017E;
            break;
        case 0x09F:
            ch = 0x0178;
            break;
        default:
            break;
        }

        wstr += (wchar_t)ch;
    }

    return true;
}

//static
bool StringUtil::CP1252ToUcs2(std::u16string &wstr, const char *str, uint32_t len)
{
    wstr.clear();
    if (str == nullptr || len == 0) {
        return true;
    }

    if (len == UINT32_MAX) {
        len = (uint32_t)strlen(str);
    }

    wstr.reserve(len);
    for (uint32_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        uint16_t ch = (uint16_t)(UInt8)str[ch_idx];
        switch (ch)
        {
        case 0x080:
            ch = 0x20AC;
            break;
        case 0x082:
            ch = 0x201A;
            break;
        case 0x083:
            ch = 0x0192;
            break;
        case 0x084:
            ch = 0x201E;
            break;
        case 0x085:
            ch = 0x2026;
            break;
        case 0x086:
            ch = 0x2020;
            break;
        case 0x087:
            ch = 0x2021;
            break;
        case 0x088:
            ch = 0x02C6;
            break;
        case 0x089:
            ch = 0x2030;
            break;
        case 0x08A:
            ch = 0x0160;
            break;
        case 0x08B:
            ch = 0x2039;
            break;
        case 0x08C:
            ch = 0x0152;
            break;
        case 0x08E:
            ch = 0x017D;
            break;
        case 0x091:
            ch = 0x2018;
            break;
        case 0x092:
            ch = 0x2019;
            break;
        case 0x093:
            ch = 0x201C;
            break;
        case 0x094:
            ch = 0x201D;
            break;
        case 0x095:
            ch = 0x2022;
            break;
        case 0x096:
            ch = 0x2013;
            break;
        case 0x097:
            ch = 0x2014;
            break;
        case 0x098:
            ch = 0x02DC;
            break;
        case 0x099:
            ch = 0x2122;
            break;
        case 0x09A:
            ch = 0x0161;
            break;
        case 0x09B:
            ch = 0x203A;
            break;
        case 0x09C:
            ch = 0x0153;
            break;
        case 0x09E:
            ch = 0x017E;
            break;
        case 0x09F:
            ch = 0x0178;
            break;
        default:
            break;
        }

        wstr += (char16_t)ch;
    }

    return true;
}

std::wstring StringUtil::CP1252ToUnicode(const std::string &str)
{
    wstring wstr;
    CP1252ToUnicode(wstr, str);
    return wstr;
}

bool StringUtil::Ucs2ToUtf8(std::string &str, const std::wstring &wstr)
{
    return Ucs2ToUtf8(str, wstr.c_str(), (uint32_t)wstr.size());
}

//static
bool StringUtil::Ucs2ToUtf8(std::string &target, CharOffsetMap *offset_map,
    const wchar_t *source, uint32_t source_len)
{
    target.clear();
    if (offset_map != nullptr) {
        offset_map->Clear();
    }
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)wcslen(source);
    }

    target.reserve(source_len);
    for (uint32_t ch_idx = 0; ch_idx < source_len; ch_idx++)
    {
        if (offset_map != nullptr)
        {
            offset_map->wcs_to_mbs.push_back((uint32_t)target.size());
            offset_map->mbs_to_wcs[(uint32_t)target.size()] = ch_idx;
        }

        uint16_t ch = (uint16_t)source[ch_idx];
        if (ch <= 0x7F)
        {
            target += (char)ch;
        }
        else if (ch <= 0x7FF)
        {
            target += (char)(0xC0 | (ch >> 6));
            target += (char)(0x80 | (ch & 0x3F));
        }
        else
        {
            target += (char)(0xE0 | (ch >> 12));
            target += (char)(0x80 | ((ch >> 6) & 0x3F));
            target += (char)(0x80 | (ch & 0x3F));
        }
    }

    return true;
}

bool StringUtil::Ucs2ToUtf8(std::string &target, const wchar_t *source, uint32_t source_len)
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)wcslen(source);
    }

    target.reserve(source_len);
    for (uint32_t ch_idx = 0; ch_idx < source_len; ch_idx++)
    {
        uint16_t ch = (uint16_t)source[ch_idx];
        if (ch <= 0x7F)
        {
            target += (char)ch;
        }
        else if (ch <= 0x7FF)
        {
            target += (char)(0xC0 | (ch >> 6));
            target += (char)(0x80 | (ch & 0x3F));
        }
        else
        {
            target += (char)(0xE0 | (ch >> 12));
            target += (char)(0x80 | ((ch >> 6) & 0x3F));
            target += (char)(0x80 | (ch & 0x3F));
        }
    }

    return true;
}

bool StringUtil::Ucs2ToUtf8(std::string &target, const std::u16string &source)
{
    return Ucs2ToUtf8(target, source.c_str(), (uint32_t)source.size());
}

bool StringUtil::Ucs2ToUtf8(std::string &target, const char16_t *source, uint32_t source_len)
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)StrLen16(source);
    }

    target.reserve(source_len);
    for (uint32_t ch_idx = 0; ch_idx < source_len; ch_idx++)
    {
        uint16_t ch = (uint16_t)source[ch_idx];
        if (ch <= 0x7F)
        {
            target += (char)ch;
        }
        else if (ch <= 0x7FF)
        {
            target += (char)(0xC0 | (ch >> 6));
            target += (char)(0x80 | (ch & 0x3F));
        }
        else
        {
            target += (char)(0xE0 | (ch >> 12));
            target += (char)(0x80 | ((ch >> 6) & 0x3F));
            target += (char)(0x80 | (ch & 0x3F));
        }
    }

    return true;
}

bool StringUtil::Utf8ToUcs2(wstring &target, const string &source,
    uint32_t max_big_char_count)
{
    return Utf8ToUcs2(target, source.c_str(), (uint32_t)source.size(), max_big_char_count);
}

bool StringUtil::Utf8ToUcs2(wstring &target, const char *source,
    uint32_t source_len, uint32_t max_big_char_count)
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    bool has_error = false;
    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)strlen(source);
    }

    target.reserve(source_len);
    int r = 0;
    uint32_t big_char_count = 0;
    uint32_t current_wchar = 0;
    for (uint32_t ch_idx = 0; ch_idx < source_len; ch_idx++)
    {
        UInt8 ch = (UInt8)source[ch_idx];
        if ((ch & 0x80) == 0)
        {
            target += (wchar_t)ch;
            if (r > 0) {
                has_error = true;
            }
            r = 0;
        }
        else if ((ch & 0x40) == 0)
        {
            if (r <= 0)
            {
                has_error = true;
            }
            else
            {
                r--;
                current_wchar <<= 6;
                current_wchar |= (ch & 0x3F);
                if (r == 0)
                {
                    if (current_wchar >= 0xFFFF)
                    {
                        current_wchar = ' ';
                        big_char_count++;
                    }

                    if (big_char_count > max_big_char_count) {
                        has_error = true;
                    }

                    target += (wchar_t)current_wchar;
                }
            }
        }
        else if ((ch & 0x20) == 0)
        {
            current_wchar = ch & 0x1F;
            if (r != 0) {
                has_error = true;
            }
            r = 1;
        }
        else if ((ch & 0x10) == 0)
        {
            current_wchar = ch & 0x0F;
            if (r != 0) {
                has_error = true;
            }
            r = 2;
        }
        else if ((ch & 0x08) == 0)
        {
            current_wchar = ch & 0x07;
            if (r != 0) {
                has_error = true;
            }
            r = 3;
        }
        else
        {
            has_error = true;
        }
    }

    return !has_error;
}

bool StringUtil::Utf8ToUcs2(std::u16string &target, const std::string &source,
    uint32_t max_big_char_count)
{
    return Utf8ToUcs2(target, source.c_str(), (uint32_t)source.size(), max_big_char_count);
}

bool StringUtil::Utf8ToUcs2(std::u16string &target, const char *source,
    uint32_t source_len, uint32_t max_big_char_count)
{
    target.clear();
    if (source == nullptr || source_len == 0) {
        return true;
    }

    bool has_error = false;
    if (source_len == UINT32_MAX) {
        source_len = (uint32_t)strlen(source);
    }

    target.reserve(source_len);
    int r = 0;
    uint32_t big_char_count = 0;
    uint32_t current_wchar = 0;
    for (uint32_t ch_idx = 0; ch_idx < source_len; ch_idx++)
    {
        UInt8 ch = (UInt8)source[ch_idx];
        if ((ch & 0x80) == 0)
        {
            target += (wchar_t)ch;
            if (r > 0) {
                has_error = true;
            }
            r = 0;
        }
        else if ((ch & 0x40) == 0)
        {
            if (r <= 0)
            {
                has_error = true;
            }
            else
            {
                r--;
                current_wchar <<= 6;
                current_wchar |= (ch & 0x3F);
                if (r == 0)
                {
                    if (current_wchar >= 0xFFFF)
                    {
                        current_wchar = ' ';
                        big_char_count++;
                    }

                    if (big_char_count > max_big_char_count) {
                        has_error = true;
                    }

                    target += (uint16_t)current_wchar;
                }
            }
        }
        else if ((ch & 0x20) == 0)
        {
            current_wchar = ch & 0x1F;
            if (r != 0) {
                has_error = true;
            }
            r = 1;
        }
        else if ((ch & 0x10) == 0)
        {
            current_wchar = ch & 0x0F;
            if (r != 0) {
                has_error = true;
            }
            r = 2;
        }
        else if ((ch & 0x08) == 0)
        {
            current_wchar = ch & 0x07;
            if (r != 0) {
                has_error = true;
            }
            r = 3;
        }
        else
        {
            has_error = true;
        }
    }

    return !has_error;
}

#ifdef _WIN32
bool StringUtil::MBS2WCS(const string &str, wstring &wstr, UINT code_page)
{
    return MBS2WCS(str.c_str(), str.size(), wstr, code_page);
}

bool StringUtil::MBS2WCS(const char *pStr, size_t str_len, wstring &wstr, UINT code_page)
{
    if (str_len == 0)
    {
        wstr.clear();
        return true;
    }

    // first test how buffer is long
    size_t cch_new_len = MultiByteToWideChar(
        code_page,
        // Windows XP and later: MB_ERR_INVALID_CHARS is the only dwFlags
        // value supported by Code page 65001 (UTF-8). 
        0,
        (LPCSTR)pStr,
        static_cast<int>(str_len),
        nullptr,
        0 // cause MultiByteToWideChar return required len
    );

    if (cch_new_len == 0)
    {
        return false;
    }

    wstr.resize(cch_new_len);

    cch_new_len = MultiByteToWideChar(
        code_page,
        // Windows XP and later: MB_ERR_INVALID_CHARS is the only dwFlags
        // value supported by Code page 65001 (UTF-8). 
        0,
        (LPCSTR)pStr,
        static_cast<int>(str_len),
        // cast constness off. Safe because this function will not write 
        // longer than capacity
        (LPWSTR)wstr.c_str(),
        // TODO: cast here
        static_cast<int>(wstr.size())
    );

    if (0 == cch_new_len)
    {
        return false;
    }
    //wstr.push_back(L'\0');
    return true;
}

bool StringUtil::WCS2MBS(const wstring &wstr, string &str, UINT code_page)
{
    return WCS2MBS(wstr.c_str(), wstr.size(), str, code_page);
}

// do UNICODE -> MultiBytes convertion
bool StringUtil::WCS2MBS(const wchar_t *wcs, size_t wstr_len, string &str, UINT code_page)
{
    if (wstr_len == 0)
    {
        str.clear();
        return true;
    }

    // first test how buffer is long
    size_t cch_new_len = WideCharToMultiByte(
        code_page,
        // Windows XP and later: MB_ERR_INVALID_CHARS is the only dwFlags
        // value supported by Code page 65001 (UTF-8). 
        0,
        (LPCWSTR)wcs,
        static_cast<int>(wstr_len),
        nullptr,
        0, // cause MultiByteToWideChar return required len
        nullptr,
        nullptr
    );

    if (cch_new_len == 0)
    {
        return false;
    }

    str.resize(cch_new_len);

    cch_new_len = WideCharToMultiByte(
        code_page,
        0,
        (LPCWSTR)wcs,
        static_cast<int>(wstr_len),
        (LPSTR)str.c_str(),
        static_cast<int>(str.size()),
        nullptr,
        nullptr
    );

    if (0 == cch_new_len)
    {
        return false;
    }
    //str.push_back('\0');

    return true;
}
#endif //_WIN32

////////////////////////////////////////////////////////////////////////////////
// class CodePageConverter_GBK

//static
CodePageData CodePageConverter_GBK::data_;

//static
bool CodePageConverter_GBK::LoadData(const string &file_path)
{
    return data_.Load(file_path);
}

//static
bool CodePageConverter_GBK::ToWCS(std::wstring &wstr, const std::string &str)
{
    return data_.ToUcs2(wstr, str.c_str(), (uint32_t)str.size());
}

//static
bool CodePageConverter_GBK::ToWCS(std::wstring &wstr, const char *str, uint32_t len)
{
    return data_.ToUcs2(wstr, str, len);
}

//static
std::wstring CodePageConverter_GBK::ToWCS(const std::string &str)
{
    wstring out;
    ToWCS(out, str);
    return out;
}

//static
std::wstring CodePageConverter_GBK::ToWCS(const char *str, uint32_t len)
{
    wstring out;
    data_.ToUcs2(out, str, len);
    return out;
}

//static
bool CodePageConverter_GBK::ToMBS(std::string &str, const std::wstring &wstr)
{
    return data_.FromUcs2(str, wstr.c_str(), (uint32_t)wstr.size());
}

//static
bool CodePageConverter_GBK::ToMBS(std::string &str, const wchar_t *wstr, uint32_t len)
{
    return data_.FromUcs2(str, wstr, len);
}

//static
std::string CodePageConverter_GBK::ToMBS(const std::wstring &wstr)
{
    string out_str;
    ToMBS(out_str, wstr);
    return out_str;
}

//static
std::string CodePageConverter_GBK::ToMBS(const wchar_t *wstr, uint32_t len)
{
    string out_str;
    data_.FromUcs2(out_str, wstr, len);
    return out_str;
}

//static
bool StringUtil::WildcardMatch(const string &str, const string &pattern)
{
    return WildcardMatch(str.c_str(), pattern.c_str());
}

//static
bool StringUtil::WildcardMatch(const char *str, const char *pattern)
{
    size_t len = strlen(str);
    for (; *pattern != '\0'; pattern++)
    {
        switch (*pattern)
        {
        case '?':
            if (*str == '\0') {
                return false;
            }
            str++;
            break;
        case '*':
            if (pattern[1] == '\0') {
                return true;
            }
            for (size_t i = 0; i < len; i++)
            {
                if (WildcardMatch(pattern + 1, str + i)) {
                    return true;
                }
            }
            return false;
        default:
            if (*str != *pattern) {
                return false;
            }
            str++;
        }
    }

    return *str == '\0';
}

} //end of namespace
