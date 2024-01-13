#pragma once

#include <string>
#include <map>
#include "prime_types.h"
#include "string.h"
#include "string_util.h"

namespace sslib
{

using namespace std;

enum class HttpHeaderFieldId
{
    //general header fields
    Connection, TransferEncoding, Date,
    //request header fields
    Accept, AcceptCharset, AcceptEncoding, ContentEncoding, Host, UserAgent,
    //response header fields
    Server,
    //entity header fields
    ContentLength, ContentType, ContentLanguage
};

typedef map<string, HttpHeaderFieldId, StrLessNoCase> HttpHeaderFieldMap;

struct HttpHeaderData
{
    string charset;
    string content_type;
    string content_language;
    string content_encoding;
    int ret_code = 0;

    void Clear()
    {
        charset.clear();
        content_type.clear();
        content_language.clear();
        content_encoding.clear();
        ret_code = 0;
    }
};

class HttpUtility
{
public:
    static std::string UrlEncode(const std::string &url);
    static bool UrlEncode(std::string &target_url, const char *url,
        uint32_t url_len = UINT32_MAX);
    static std::string UrlDecode(const std::string &url);
    static bool UrlDecode(std::string &target_url, const char *url,
        uint32_t url_len = UINT32_MAX);
    static std::wstring UrlDecode(const std::wstring &url);
    static bool UrlDecode(std::wstring &target_url, const wchar_t *url,
        uint32_t url_len = UINT32_MAX);

    static void InitHttpFieldMap(HttpHeaderFieldMap &field_map);
    static void InitCharsetMap(map<string, CodePageId> &charset_map);

    static void ParseHttpHeader(HttpHeaderData &hhd, const string &header_str,
        const UnicodeTable &uct, const HttpHeaderFieldMap &field_map);
    static void GetCharsetAndLanguageFromContent(HttpHeaderData &hhd,
        const string &content_str);
};

} //end of namespace
