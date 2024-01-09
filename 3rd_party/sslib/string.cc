#include "string.h"
#include "string_util.h"
#include <cwctype>

using namespace std;

namespace sslib
{

bool StrLessNoCase::operator () (const std::string &str1, const std::string &str2) const
{
    return strcasecmp(str1.c_str(), str2.c_str()) < 0;
}

bool StrLessNoCase::operator () (const char *sz1, const char *sz2) const
{
    return strcasecmp(sz1, sz2) < 0;
}

bool WStrLessNoCase::operator () (const std::wstring &str1, const std::wstring &str2) const
{
    return wcscasecmp(str1.c_str(), str2.c_str()) < 0;
}

bool WStrLessNoCase::operator () (const wchar_t *sz1, const wchar_t *sz2) const
{
    return wcscasecmp(sz1, sz2) < 0;
}

String::String() : string()
{
}

String::String(const char *ptr, string::size_type count, const allocator<char>& _Al)
	: string(ptr, count, _Al)
{
}

String::String(const char *ptr, const allocator<char> &_Al)
	: string(ptr, _Al)
{
}

String::String(const string& str, string::size_type roff, string::size_type count)
	: string(str, roff, count)
{
}

String::String(const string::size_type count, string::value_type ch)
	: string(count, ch)
{
}

String::~String()
{
}

String &String::TrimLeft(const string &target_str)
{
	string::size_type pos = find_first_not_of(target_str);
	if(pos != string::npos)
	{
		this->assign(this->substr(pos));
	}
	else this->clear();

	return (*this);
}

String &String::TrimRight(const string &target_str)
{
	string::size_type pos = find_last_not_of(target_str);
	if(pos != string::npos)
	{
		this->assign(this->substr(0, pos+1));
	}
	else this->clear();

	return (*this);
}

String &String::Trim(const string &target_str)
{
	TrimLeft(target_str);
	TrimRight(target_str);
	return (*this);
}

String &String::MakeLower()
{
    size_t len = size();
    for(size_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        if((*this)[ch_idx] >= 'A' && (*this)[ch_idx] <= 'Z') {
            (*this)[ch_idx] = (*this)[ch_idx] - 'A' + 'a';
        }
    }

	return(*this);
}

String &String::MakeUpper()
{
	size_t len = size();
    for(size_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        if((*this)[ch_idx] >= 'a' && (*this)[ch_idx] <= 'z') {
            (*this)[ch_idx] = (*this)[ch_idx] - 'a' + 'A';
        }
    }

	return(*this);
}

// Replace in the string the first occurrence of the specified substring with another specified string
String& String::ReplaceOne(const char *from, const char *to)
{
	string::size_type idx = find(from);
	if(idx != string::npos)
	{
		replace(idx, strlen(from), to);
	}

	return *this;
}

// Replace in the string all occurrences of the specified substring with another specified string
String& String::ReplaceAll(const char *from, const char *to)
{
    ReplaceAll(*this, from, to);
    return *this;
}

// Break the string into tokens
void String::Tokenize(vector<String> &tokens, const string &delims) const
{
	tokens.clear();

	string item_str;
	string::size_type begin_idx, end_idx;

	begin_idx = find_first_not_of(delims);
	while(begin_idx != string::npos ) //while beginning of a token found
	{
		//search end of the token
		end_idx = find_first_of(delims, begin_idx);

		//
        if (end_idx == string::npos) {
            item_str = substr(begin_idx, end_idx);
        }
        else {
            item_str = substr(begin_idx, end_idx - begin_idx);
        }
		tokens.push_back(item_str);

		//
		begin_idx = find_first_not_of(delims, end_idx);
	}
}

void String::Split(std::vector<string> &tokens, const std::string &delims, const StrSplitOptions *options)
{
    Split(*this, tokens, delims, options);
}

void String::TokenizeTabled(vector<String>& tokens, const string& delims) const
{
	tokens.clear();

	string item_str;
	string::size_type begin_idx,end_idx;

	begin_idx = 0;
	while(begin_idx != string::npos && begin_idx< length()) //while beginning of a token found
	{
		//search end of the token
		end_idx = find_first_of(delims, begin_idx);

		//
		if(end_idx == string::npos)
			item_str = substr(begin_idx, end_idx);
		else item_str = substr(begin_idx, end_idx-begin_idx);
		tokens.push_back(item_str);

		//
		begin_idx =end_idx+1;
	}
}

#ifdef _WIN32
//static
bool String::ChangeCodePage(string &str,
    uint32_t src_code_page, uint32_t target_code_page)
{
    wstring wstr;
    bool ret = StringUtil::MBS2WCS(str, wstr, src_code_page);
    ret = ret && StringUtil::WCS2MBS(wstr, str, target_code_page);
    return ret;
}

//static
bool String::ChangeCodePage(string &output_str, const char *input_str,
    uint32_t src_code_page, uint32_t target_code_page)
{
    wstring wstr;
    bool ret = StringUtil::MBS2WCS(input_str, strlen(input_str), wstr, src_code_page);
    ret = ret && StringUtil::WCS2MBS(wstr, output_str, target_code_page);
    return ret;
}
#endif //def _WIN32

//static
string& String::TrimLeft(std::string &str, const std::string &target_str)
{
    string::size_type pos = str.find_first_not_of(target_str);
    if(pos != string::npos) {
        str.assign(str.substr(pos));
    }
    else {
        str.clear();
    }

    return str;
}

//static
string& String::TrimRight(std::string &str, const std::string &target_str)
{
    string::size_type pos = str.find_last_not_of(target_str);
    if(pos != string::npos) {
        str.assign(str.substr(0, pos+1));
    }
    else {
        str.clear();
    }

    return str;
}

//static
string& String::Trim(std::string &str, const std::string &target_str)
{
    TrimLeft(str, target_str);
    TrimRight(str, target_str);
    return str;
}

//static
void String::MakeLower(std::string &str, const UnicodeTable &char_table)
{
    wstring wstr;
    StringUtil::Utf8ToWideStr(wstr, str);
    WString::MakeLower(wstr, char_table);
    StringUtil::ToUtf8(str, wstr);
}

//static
std::string String::MakeLower(const char *str, const UnicodeTable &char_table)
{
    wstring wstr;
    StringUtil::Utf8ToWideStr(wstr, str, (UInt32)strlen(str));
    WString::MakeLower(wstr, char_table);
    string ret_str;
    StringUtil::ToUtf8(ret_str, wstr);
    return ret_str;
}

//static
void String::MakeUpper(std::string &str, const UnicodeTable &char_table)
{
    wstring wstr;
    StringUtil::Utf8ToWideStr(wstr, str);
    WString::MakeUpper(wstr, char_table);
    StringUtil::ToUtf8(str, wstr);
}

//static
std::string String::MakeUpper(const char *str, const UnicodeTable &char_table)
{
    wstring wstr;
    StringUtil::Utf8ToWideStr(wstr, str, (UInt32)strlen(str));
    WString::MakeUpper(wstr, char_table);
    string ret_str;
    StringUtil::ToUtf8(ret_str, wstr);
    return ret_str;
}

//static
int String::CaseCmp(const char *str1, const char *str2)
{
    return strcasecmp(str1, str2);
}

//static
int String::CaseCmp(const string &str1, const char *str2)
{
    return strcasecmp(str1.c_str(), str2);
}

//static
int String::CaseCmp(const string &str1, const string &str2)
{
    return strcasecmp(str1.c_str(), str2.c_str());
}

//static
bool String::Itoa(string &str, Int32 value, int radix)
{
    return Itoa(str, (Int64)value, radix);
}

//static
bool String::Itoa(string &str, UInt32 value, int radix)
{
    return Itoa(str, (uint64_t)value, radix);
}

//static
bool String::Itoa(string &str, Int64 value, int radix)
{
    str.clear();
    if (radix < 2 || radix > 36) {
        return false;
    }

    static char dig[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    bool isNeg = false;

    if (radix == 10 && value < 0) {
        value = -value;
        isNeg = true;
    }

    uint64_t v = value;
    do
    {
        str += dig[v % radix];
        v /= radix;
    } while (v != 0);

    if (isNeg) {
        str += '-';
    }

    int n = (int)str.size();
    for (int idx = 0; idx < n / 2; idx++)
    {
        char ch = str[idx];
        ch = str[idx];
        str[idx] = str[n - idx - 1];
        str[n - idx - 1] = ch;
    }

    return true;
}

//static
bool String::Itoa(string &str, uint64_t value, int radix)
{
    str.clear();
    if (radix < 2 || radix > 36) {
        return false;
    }

    static char dig[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    uint64_t v = value;
    do
    {
        str += dig[v % radix];
        v /= radix;
    } while (v != 0);

    int n = (int)str.size();
    for (int idx = 0; idx < n / 2; idx++)
    {
        char ch = str[idx];
        ch = str[idx];
        str[idx] = str[n - idx - 1];
        str[n - idx - 1] = ch;
    }

    return true;
}

//static
void String::ReplaceOne(string &str, const char *from, const char *to)
{
	string::size_type idx = str.find(from);
	if(idx != string::npos)
	{
		str.replace(idx, strlen(from), to);
	}
}

//static
void String::ReplaceOne(std::string &str, const string &from, const string &to)
{
    return ReplaceOne(str, from.c_str(), to.c_str());
}

//static
void String::ReplaceAll(std::string &str, const char *from, const char *to)
{
    string::size_type idx = 0, begin_pos = 0;
    while (idx != string::npos)
    {
        idx = str.find(from, begin_pos);
        if (idx == string::npos) {
            break;
        }

        str.replace(idx, strlen(from), to);
        begin_pos = idx + strlen(to);
    }
}

//static
void String::ReplaceAll(std::string &str, const string &from, const string &to)
{
    return ReplaceAll(str, from.c_str(), to.c_str());
}

void String::Tokenize(const string &str, std::vector<std::string>& tokens, const std::string& delims)
{
	tokens.clear();

	string item_str;
	string::size_type begin_idx, end_idx;

	begin_idx = str.find_first_not_of(delims);
	while(begin_idx != string::npos) //while beginning of a token found
	{
		//search end of the token
		end_idx = str.find_first_of(delims, begin_idx);

		//
		if(end_idx == string::npos)
			item_str = str.substr(begin_idx, end_idx);
		else item_str = str.substr(begin_idx, end_idx-begin_idx);
		tokens.push_back(item_str);

		//
		begin_idx = str.find_first_not_of(delims, end_idx);
	}
}

//static
void String::Split(const std::string &str, std::vector<string> &tokens, const std::string &delims,
    const StrSplitOptions *options)
{
    StrSplitOptions opt;
    if(options == nullptr) {
        options = &opt;
    }

    tokens.clear();
    if(str.empty()) {
        return;
    }

    size_t len = str.size();
    size_t term_start = 0;
    for(size_t ch_idx = 0; ch_idx <= len; ch_idx++)
    {
        char ch = ch_idx < len ? str[ch_idx] : ' ';
        bool is_found = ch_idx < len ? delims.find(ch) != string::npos : true;
        if(is_found)
        {
            if(!options->m_bSkipEmptyItems || ch_idx > term_start)
            {
                tokens.push_back(str.substr(term_start, ch_idx - term_start));
            }
            term_start = ch_idx + 1;
        }
    }
}

WString::WString() : wstring()
{
}

WString::WString(const wchar_t *ptr, wstring::size_type count, const allocator<wchar_t> &_Al)
	: wstring(ptr, count, _Al)
{
}

WString::WString(const wchar_t *ptr, const allocator<wchar_t> &_Al)
	: wstring(ptr, _Al)
{
}

WString::WString(const wstring &str, wstring::size_type roff, wstring::size_type count)
	: wstring(str, roff, count)
{
}

WString::WString(const wstring::size_type count, wstring::value_type ch)
	: wstring(count, ch)
{
}

WString::~WString()
{
}

WString &WString::TrimLeft(const wstring &target_str)
{
	wstring::size_type pos = find_first_not_of(target_str);
	if(pos != wstring::npos)
	{
		this->assign(this->substr(pos));
	}
	else this->clear();

	return (*this);
}

WString &WString::TrimRight(const wstring &target_str)
{
	wstring::size_type pos = find_last_not_of(target_str);
	if(pos != wstring::npos)
	{
		this->assign(this->substr(0, pos+1));
	}
	else this->clear();

	return (*this);
}

WString &WString::Trim(const wstring &target_str)
{
	TrimLeft(target_str);
	TrimRight(target_str);
	return (*this);
}

WString &WString::MakeLower()
{
    size_t len = size();
    for (size_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        (*this)[ch_idx] = towlower((*this)[ch_idx]);
    }

    return(*this);
}

WString &WString::MakeUpper()
{
    size_t len = size();
    for (size_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        (*this)[ch_idx] = towupper((*this)[ch_idx]);
    }

    return(*this);
}

/** Replace in the wstring the first occurrence of the specified substring with another specified string */
WString& WString::ReplaceOne(const wchar_t *from, const wchar_t *to)
{
	wstring::size_type idx = find(from);
	if(idx != wstring::npos)
	{
		replace(idx, wcslen(from), to);
	}

	return *this;
}

// Replace in the string all occurrences of the specified substring with another specified string
WString& WString::ReplaceAll(const wchar_t *from, const wchar_t *to)
{
    ReplaceAll(*this, from, to);
    return *this;
}

// Break the string into tokens
void WString::Tokenize(vector<WString> &tokens, const wstring &delims) const
{
	tokens.clear();

	wstring item_str;
	wstring::size_type begin_idx, end_idx;

	begin_idx = find_first_not_of(delims);
	while(begin_idx != wstring::npos) //while beginning of a token found
	{
		//search end of the token
		end_idx = find_first_of(delims, begin_idx);

		//
        if (end_idx == wstring::npos) {
            item_str = substr(begin_idx, end_idx);
        }
        else {
            item_str = substr(begin_idx, end_idx - begin_idx);
        }
		tokens.push_back(item_str);

		//
		begin_idx = find_first_not_of(delims, end_idx);
	}
}

void WString::Split(std::vector<wstring> &tokens, const std::wstring &delims,
    const StrSplitOptions *options)
{
    Split(*this, tokens, delims, options);
}

//static
bool WString::HasLetter(const wstring &str, const UnicodeTable &uct)
{
    for (size_t idx = 0; idx < str.size(); idx++)
    {
        const auto &uci = uct.Get(str[idx]);
        if (uci.prime_type == UctPrime::Letter) {
            return true;
        }
    }

    return false;
}

//static
bool WString::HasLetter(const wchar_t *wstr, const UnicodeTable &char_table)
{
    for(; *wstr != 0; wstr++)
    {
        wchar_t wch = wstr[0];
        if(char_table.Get(wch).prime_type == UctPrime::Letter) {
            return true;
        }
    }

    return false;
}

//static
bool WString::HasLetterOrDigit(const wstring &str, const UnicodeTable &uct)
{
    bool has_letter_or_digit = false;
    for (size_t idx = 0; idx < str.size(); idx++)
    {
        const auto &uci = uct.Get(str[idx]);
        if (uci.prime_type == UctPrime::Letter || uci.prime_type == UctPrime::Number)
        {
            has_letter_or_digit = true;
            break;
        }
    }
    return has_letter_or_digit;
}

//static
bool WString::HasLetterOrDigit(const wchar_t *str, const UnicodeTable &uct)
{
    if (str == nullptr) {
        return false;
    }

    bool has_letter_or_digit = false;
    for (const wchar_t *ptr = str; *ptr != L'\0'; ptr++)
    {
        const auto &uci = uct.Get(*ptr);
        if (uci.prime_type == UctPrime::Letter || uci.prime_type == UctPrime::Number) {
            has_letter_or_digit = true;
            break;
        }
    }
    return has_letter_or_digit;
}

//static
std::wstring& WString::TrimLeft(std::wstring &str, const std::wstring &target_str)
{
    wstring::size_type pos = str.find_first_not_of(target_str);
    if(pos != wstring::npos) {
        str = str.substr(pos);
    }
    else {
        str.clear();
    }

    return str;
}

//static
std::wstring& WString::TrimRight(std::wstring &str, const std::wstring &target_str)
{
    wstring::size_type pos = str.find_last_not_of(target_str);
    if(pos != wstring::npos) {
        str.resize(pos+1);
    }
    else {
        str.clear();
    }

    return str;
}

//static
std::wstring& WString::Trim(std::wstring &str, const std::wstring &target_str)
{
    TrimLeft(str, target_str);
    return TrimRight(str, target_str);
}

//static
std::wstring& WString::TrimSeparatorsLeft(std::wstring &str, const UnicodeTable &uct)
{
    size_t new_start = 0;
    for (size_t idx = 0; idx < str.size(); idx++)
    {
        const auto &uci = uct.Get(str[idx]);
        if (uci.prime_type != UctPrime::Separator) {
            break;
        }

        new_start = idx + 1;
    }

    if (new_start > 0)
    {
        if (new_start < str.size()) {
            str = str.substr(new_start);
        }
        else {
            str.clear();
        }
    }

    return str;
}

//static
std::wstring& WString::TrimSeparatorsRight(std::wstring &str, const UnicodeTable &uct)
{
    if (str.empty()) {
        return str;
    }

    int len = (int)str.size();
    int new_end = len;
    for (int idx = len; idx >= 0; idx--)
    {
        const auto &uci = uct.Get(str[idx]);
        if (uci.prime_type != UctPrime::Separator) {
            break;
        }

        new_end = idx;
    }

    if (new_end < len)
    {
        str = str.substr(0, new_end);
    }

    return str;
}

//static
std::wstring& WString::TrimSeparators(std::wstring &str, const UnicodeTable &uct)
{
    TrimSeparatorsLeft(str, uct);
    return TrimSeparatorsRight(str, uct);
}

//static
void WString::MakeLower(std::wstring &wstr, const UnicodeTable &char_table)
{
    for(int idx = (int)wstr.size() - 1; idx >= 0; idx--) {
        wstr[idx] = char_table.Get(wstr[idx]).lower_case;
    }
}

//static
void WString::MakeUpper(std::wstring &wstr, const UnicodeTable &char_table)
{
    for(int idx = (int)wstr.size() - 1; idx >= 0; idx--) {
        wstr[idx] = char_table.Get(wstr[idx]).upper_case;
    }
}

//static
uint32_t WString::HashCodeNoCaseUInt32(const std::wstring &wstr, const UnicodeTable &char_table)
{
    return HashCodeNoCaseUInt32(wstr.c_str(), (uint32_t)wstr.size(), char_table);
}

//static
uint32_t WString::HashCodeNoCaseUInt32(const wchar_t *wcs, uint32_t len,
    const UnicodeTable &char_table)
{
    uint32_t result = 0;
    for(uint32_t idx = 0; idx < len && (wcs[idx] != L'\0' || len != UINT32_MAX); idx++)
    {
        result = 31 * result + char_table.Get(wcs[idx]).lower_case;
    }
    return result;
}

//static
uint32_t WString::HashCodeNoCaseUInt32(const wchar_t *wcs, const UnicodeTable &char_table)
{
    return HashCodeNoCaseUInt32(wcs, UINT32_MAX, char_table);
}

//static
void WString::ReplaceOne(wstring &str, const wchar_t *from, const wchar_t *to)
{
	wstring::size_type idx = str.find_first_of(from);
	if(idx != wstring::npos)
	{
		str.replace(idx, wcslen(from), to);
	}
}

//static
void WString::ReplaceOne(std::wstring &str,
    const wstring &from, const wstring &to)
{
    ReplaceOne(str, from.c_str(), to.c_str());
}

//static
void WString::ReplaceAll(std::wstring &str, const wchar_t *from, const wchar_t *to)
{
    wstring::size_type idx = 0, begin_pos = 0;
    while(idx != wstring::npos)
    {
        idx = str.find(from, begin_pos);
        if (idx == wstring::npos) {
            break;
        }

        str.replace(idx, wcslen(from), to);
        begin_pos = idx + wcslen(to);
    }
}

//static
void WString::ReplaceAll(std::wstring &str,
    const wstring &from, const wstring &to)
{
    ReplaceAll(str, from.c_str(), to.c_str());
}

//static
int WString::CaseCmp(const wchar_t *lhs, const wchar_t *rhs)
{
    if (lhs == nullptr) {
        return rhs == nullptr ? 0 : -1;
    }
    else if (rhs == nullptr) {
        return 1;
    }
    return wcscasecmp(lhs, rhs);
}

//static
int WString::CaseCmp(const wstring &lhs, const wchar_t *rhs)
{
    if (rhs == nullptr) {
        return 1;
    }
    return wcscasecmp(lhs.c_str(), rhs);
}

//static
int WString::CaseCmp(const wstring &lhs, const wstring &rhs)
{
    return wcscasecmp(lhs.c_str(), rhs.c_str());
}

void WString::Tokenize(const wstring &str, std::vector<std::wstring> &tokens, const std::wstring &delims)
{
	tokens.clear();
	wstring item_str;
	wstring::size_type begin_idx, end_idx;

	begin_idx = str.find_first_not_of(delims);
	while(begin_idx != wstring::npos) //while beginning of a token found
	{
		//search end of the token
		end_idx = str.find_first_of(delims, begin_idx);

		//
        if (end_idx == wstring::npos) {
            item_str = str.substr(begin_idx, end_idx);
        }
        else {
            item_str = str.substr(begin_idx, end_idx - begin_idx);
        }
		tokens.push_back(item_str);

		//
		begin_idx = str.find_first_not_of(delims, end_idx);
	}
}

//static
void WString::Split(const std::wstring &str, std::vector<wstring> &tokens, const std::wstring &delims,
    const StrSplitOptions *options)
{
    StrSplitOptions opt;
    if(options == nullptr) {
        options = &opt;
    }

    tokens.clear();
    if(str.empty()) {
        return;
    }

    size_t len = str.size();
    size_t term_start = 0;
    for(size_t ch_idx = 0; ch_idx <= len; ch_idx++)
    {
        wchar_t wch = ch_idx < len ? str[ch_idx] : L' ';
        bool is_found = ch_idx < len ? delims.find(wch) != wstring::npos : true;
        if(is_found)
        {
            if(!options->m_bSkipEmptyItems || ch_idx > term_start)
            {
                tokens.push_back(str.substr(term_start, ch_idx - term_start));
            }
            term_start = ch_idx + 1;
        }
    }
}

} //end of namespace
