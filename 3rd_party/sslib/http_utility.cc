#include "http_utility.h"
#include <vector>
#include "string_util.h"

namespace sslib
{

using namespace std;

//static
std::string HttpUtility::UrlEncode(const string &url)
{
    string target_url;
    UrlEncode(target_url, url.c_str(), (uint32_t)url.size());
    return target_url;
}

//static
bool HttpUtility::UrlEncode(string &target_url, const char *url, uint32_t url_len)
{
    target_url.clear();
    for (uint32_t idx = 0; idx < url_len; idx++)
    {
        UInt8 ch = (UInt8)url[idx];
        if ((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
        {
            target_url += ch;
        }
        else
        {
            UInt8 d1 = ch >> 4, d2 = (ch & 0x0F);
            target_url += "%";
            target_url += (d1 <= 9 ? '0' + d1 : 'A' + (d1 - 10));
            target_url += (d2 <= 9 ? '0' + d2 : 'A' + (d2 - 10));
        }
    }

    return true;
}

//static
std::string HttpUtility::UrlDecode(const string &url)
{
    string target_url;
    UrlDecode(target_url, url.c_str(), (uint32_t)url.size());
    return target_url;
}

//static
bool HttpUtility::UrlDecode(string &target_url, const char *url, uint32_t url_len)
{
    target_url.clear();
    if (url == nullptr || url_len == 0) {
        return true;
    }

    char seq_buf[3], ch = ' ';
    uint32_t seq_len = 0, seq_value = 0;
    uint32_t digit = 0;
    for (uint32_t char_idx = 0; char_idx < url_len && url[char_idx] != '\0'; char_idx++)
    {
        ch = url[char_idx];
        if (ch == '%')
        {
            if (seq_len > 0)
            {
                target_url.append(seq_buf, seq_len);
                seq_len = 0;
                seq_value = 0;
            }
            seq_buf[seq_len++] = ch;
        }
        else
        {
            if (seq_len == 0)
            {
                target_url += ch;
            }
            else
            {
                if (ch >= '0' && ch <= '9') {
                    digit = (uint32_t)(ch - '0');
                }
                else if (ch >= 'a' && ch <= 'f') {
                    digit = 10 + (uint32_t)(ch - 'a');
                }
                else if (ch >= 'A' && ch <= 'F') {
                    digit = 10 + (uint32_t)(ch - 'A');
                }
                else {
                    digit = 0x10;
                }

                if (digit <= 0x0F)
                {
                    seq_buf[seq_len++] = ch;
                    seq_value = ((seq_value << 4) | digit);
                    if (seq_len == 3)
                    {
                        target_url += (char)seq_value;
                        seq_len = 0;
                        seq_value = 0;
                    }
                }
                else
                {
                    target_url.append(seq_buf, seq_len);
                    seq_len = 0;
                    seq_value = 0;
                }
            }
        }
    }

    if (seq_len > 0) {
        target_url.append(seq_buf, seq_len);
    }

    return true;
}

//static
std::wstring HttpUtility::UrlDecode(const wstring &url)
{
    wstring target_url;
    UrlDecode(target_url, url.c_str());
    return target_url;
}

//static
bool HttpUtility::UrlDecode(wstring &target_url, const wchar_t *url, uint32_t url_len)
{
    target_url.clear();
    if (url == nullptr || url_len == 0) {
        return true;
    }

    wchar_t seq_buf[3], ch;
    uint32_t seq_len = 0, seq_value = 0;
    uint32_t digit = 0;
    for (uint32_t char_idx = 0; char_idx < url_len && url[char_idx] != L'\0'; char_idx++)
    {
        ch = url[char_idx];
        if (ch == L'%')
        {
            if (seq_len > 0)
            {
                target_url.append(seq_buf, seq_len);
                seq_len = 0;
                seq_value = 0;
            }
            seq_buf[seq_len++] = ch;
        }
        else
        {
            if (seq_len == 0)
            {
                target_url += ch;
            }
            else
            {
                if (ch >= L'0' && ch <= L'9') {
                    digit = (uint32_t)(ch - L'0');
                }
                else if (ch >= L'a' && ch <= L'f') {
                    digit = 10 + (uint32_t)(ch - L'a');
                }
                else if (ch >= L'A' && ch <= L'F') {
                    digit = 10 + (uint32_t)(ch - L'A');
                }
                else {
                    digit = 0x10;
                }

                if (digit <= 0x0F)
                {
                    seq_buf[seq_len++] = ch;
                    seq_value = ((seq_value << 4) | digit);
                    if (seq_len == 3)
                    {
                        target_url += (char)seq_value;
                        seq_len = 0;
                        seq_value = 0;
                    }
                }
                else
                {
                    target_url.append(seq_buf, seq_len);
                    seq_len = 0;
                    seq_value = 0;
                }
            }
        }
    }

    if (seq_len > 0) {
        target_url.append(seq_buf, seq_len);
    }

    return true;
}

//static
void HttpUtility::InitHttpFieldMap(HttpHeaderFieldMap &field_map)
{
    field_map.clear();
    field_map["connection"] = HttpHeaderFieldId::Connection;
    field_map["transfer-encoding"] = HttpHeaderFieldId::TransferEncoding;

    field_map["content-length"] = HttpHeaderFieldId::ContentLength;
    field_map["content-type"] = HttpHeaderFieldId::ContentType;
    field_map["content-language"] = HttpHeaderFieldId::ContentLanguage;
    field_map["Content-Encoding"] = HttpHeaderFieldId::ContentEncoding;
}

//static
void HttpUtility::InitCharsetMap(map<string, CodePageId> &charset_map)
{
    charset_map.clear();
    charset_map["utf-8"] = CodePageId::Utf8;
    charset_map["utf8"] = CodePageId::Utf8;
    charset_map["gbk"] = CodePageId::GBK;
    charset_map["gb2312"] = CodePageId::GBK;
    charset_map["gbk2312"] = CodePageId::GBK;
    charset_map["big5"] = CodePageId::Big5;
    charset_map["cp1252"] = CodePageId::CP1252;
    charset_map["windows-1252"] = CodePageId::CP1252;
    charset_map["iso-8859-1"] = CodePageId::Latin01;
    charset_map["iso-8859-15"] = CodePageId::Latin15;
}

//static
void HttpUtility::ParseHttpHeader(HttpHeaderData &hhd, const string &header_str,
    const UnicodeTable &uct, const HttpHeaderFieldMap &field_map)
{
    hhd.Clear();
    string charset_indicator = "charset=";

    vector<string> row_list;
    String::Split(header_str, row_list, "\n");

    if (!row_list.empty())
    {
        const string &row0 = row_list[0];
        size_t pos1 = row0.find(' ');
        size_t pos2 = pos1 == string::npos ? pos1 : row0.find(' ', pos1 + 1);
        if (pos2 != string::npos && pos2 > pos1)
        {
            string ret_code_str = row0.substr(pos1 + 1, pos2 - pos1 - 1);
            hhd.ret_code = String::ToInt32(ret_code_str);
        }
    }

    string attr_name, attr_value;
    for (auto &row_str : row_list)
    {
        String::Trim(row_str);
        string::size_type offset = row_str.find(":");
        if (offset == string::npos) {
            continue;
        }

        attr_name = row_str.substr(0, offset);
        auto attr_iter = field_map.find(attr_name);
        if (attr_iter == field_map.end()) {
            continue;
        }

        attr_value = row_str.substr(offset + 1);
        String::Trim(attr_value);

        switch (attr_iter->second)
        {
        case HttpHeaderFieldId::ContentType:
            offset = attr_value.find_first_of(" ;\"");
            hhd.content_type = offset != string::npos ? attr_value.substr(0, offset) : attr_value;
            String::Trim(hhd.content_type);

            offset = attr_value.find(charset_indicator);
            if (offset != string::npos)
            {
                auto new_start = offset + charset_indicator.size();
                auto offset2 = attr_value.find_first_of(";,", new_start);
                if (offset2 == string::npos) {
                    hhd.charset = attr_value.substr(new_start);
                }
                else {
                    hhd.charset = attr_value.substr(new_start, offset2 - new_start);
                }
                String::Trim(hhd.charset);
            }
            break;
        case HttpHeaderFieldId::ContentLanguage:
            hhd.content_language = attr_value;
            break;
        case HttpHeaderFieldId::ContentEncoding:
            hhd.content_encoding = attr_value;
            break;
        default:
            break;
        }
    }

    String::MakeLower(hhd.charset, uct);
    String::MakeLower(hhd.content_type, uct);
    String::MakeLower(hhd.content_language, uct);
}

//static
void HttpUtility::GetCharsetAndLanguageFromContent(HttpHeaderData &hhd,
    const string &content_str)
{
    hhd.Clear();
    string charset_indicator = "charset=";
    String lower_content(content_str);
    lower_content.MakeLower();

    size_t start_pos = 0;
    size_t pos1 = 0, pos2 = 0, pos3 = 0;
    while (start_pos < lower_content.size())
    {
        pos1 = lower_content.find("<meta", start_pos);
        if (pos1 == string::npos) {
            return;
        }

        pos2 = lower_content.find(charset_indicator, pos1);
        pos3 = lower_content.find(">", pos1);
        if (pos3 != string::npos && pos2 < pos3) {
            break;
        }

        //next meta tag
        start_pos = pos1 + 1;
    }

    size_t pos4 = lower_content.find("<body", pos1);
    if (pos3 != string::npos && pos4 != string::npos && pos2 < pos3 && pos3 < pos4)
    {
        pos2 += charset_indicator.size();
        if (lower_content[pos2] == '\"') {
            pos2++;
        }
        size_t pos5 = lower_content.find_first_of(";'\" />", pos2);
        if (pos5 != string::npos)
        {
            hhd.charset = lower_content.substr(pos2, pos5 - pos2);
        }
    }
}

} //end of namespace
