#pragma once

#include <string>
#include <cstring>
#include <vector>
#include "prime_types.h"
#include "unicode_table.h"

namespace sslib
{

using std::string;
using std::wstring;

struct StrLess
{
    bool operator () (const std::string &str1, const std::string &str2) const {
        return strcmp(str1.c_str(), str2.c_str()) < 0;
    }
    bool operator () (const char *sz1, const char *sz2) const {
        return strcmp(sz1, sz2) < 0;
    }
};

struct StrLessNoCase
{
    bool operator () (const std::string &str1, const std::string &str2) const;
    bool operator () (const char *sz1, const char *sz2) const;
};
typedef StrLessNoCase NullTerminatedStringIgnoreCaseLess;

struct WStrLess
{
    bool operator () (const std::wstring &str1, const std::wstring &str2) const {
        return wcscmp(str1.c_str(), str2.c_str()) < 0;
    }
    bool operator () (const wchar_t *sz1, const wchar_t *sz2) const {
        return wcscmp(sz1, sz2) < 0;
    }
};
struct WStrLessNoCase
{
    bool operator () (const std::wstring &str1, const std::wstring &str2) const;
    bool operator () (const wchar_t *sz1, const wchar_t *sz2) const;
};

struct StrSplitOptions
{
    bool m_bSkipEmptyItems;

    StrSplitOptions() {
        m_bSkipEmptyItems = false;
    }
};

class String : public std::string
{
public:
	String();
	String(const char *ptr, std::string::size_type count, const std::allocator<char> &_Al = std::allocator<char>());
	String(const char *ptr, const std::allocator<char> &_Al = std::allocator<char>());
	String(const std::string &str, std::string::size_type roff = 0, std::string::size_type count = std::string::npos);
	String(const std::string::size_type count, std::string::value_type ch);
	virtual ~String();

	//Replace in the string the first occurrence of the specified substring with another specified string
	String& ReplaceOne(const char *from, const char *to);
	String& ReplaceOne(const std::string &from, const std::string &to) {
		return ReplaceOne(from.c_str(), to.c_str());
	}
	//Replace in the string all occurrences of the specified substring with another specified string
	String& ReplaceAll(const char *from, const char *to);
	String& ReplaceAll(const std::string &from, const std::string &to) {
		return ReplaceAll(from.c_str(), to.c_str());
	}

	String &TrimLeft(const std::string &target=" \r\n\t");
	String &TrimRight(const std::string &target=" \r\n\t");
	String &Trim(const std::string &target=" \r\n\t");

	String &MakeLower();
	String &MakeUpper();

	//Break the string into tokens
	void Tokenize(std::vector<String> &tokens, const std::string &delims) const;
    void Split(std::vector<std::string> &tokens, const std::string &delims,
        const StrSplitOptions *options = nullptr);

	//insert a null string if the delims is the first char
	void TokenizeTabled(std::vector<String> &tokens, const std::string &delims) const;

#ifdef _WIN32
    static bool ChangeCodePage(std::string &str,
        uint32_t src_code_page, uint32_t target_code_page);
    static bool ChangeCodePage(std::string &strOutput, const char *szInput,
        uint32_t src_code_page, uint32_t target_code_page);
#endif //def _WIN32

//Static members
public:
    static bool IsNullOrEmpty(const char *str) {
        return str == nullptr || str[0] == '\0';
    }

    static std::string& TrimLeft(std::string &str, const std::string &target=" \r\n\t");
    static std::string& TrimRight(std::string &str, const std::string &target=" \r\n\t");
    static std::string& Trim(std::string &str, const std::string &target=" \r\n\t");

    static void MakeLower(std::string &str, const UnicodeTable &char_table);
    static std::string MakeLower(const char *str, const UnicodeTable &char_table);
    static void MakeUpper(std::string &str, const UnicodeTable &char_table);
    static std::string MakeUpper(const char *str, const UnicodeTable &char_table);

    static void ReplaceOne(std::string &str, const char *from, const char *to);
    static void ReplaceOne(std::string &str, const string &from, const string &to);
    static void ReplaceAll(std::string &str, const char *from, const char *to);
    static void ReplaceAll(std::string &str, const string &from, const string &to);

	static int StrCmp(const char *str1, const char *str2) { return strcmp(str1, str2); }
    static int CaseCmp(const char *str1, const char *str2);
    static int CaseCmp(const string &str1, const char *str2);
    static int CaseCmp(const string &str1, const string &str2);

    static bool Itoa(std::string &str, Int32 value, int radix);
    static bool Itoa(std::string &str, UInt32 value, int radix);
    static bool Itoa(std::string &str, Int64 value, int radix);
    static bool Itoa(std::string &str, UInt64 value, int radix);

    static int32_t ToInt32(const std::string &str)
    {
        return atoi(str.c_str());
    }

    static int32_t ToInt32(const std::wstring &str)
    {
#       ifdef _WIN32
        return (int32_t)_wtoi(str.c_str());
#       else
        return (int32_t)wcstol(str.c_str(), 0, 10);
#       endif
    }

    static int64_t ToInt64(const std::string &str)
    {
#       ifdef _WIN32
        return (int64_t)_atoi64(str.c_str());
#       else
        return (int64_t)strtoll(str.c_str(), 0, 10);
#       endif
    }

    static int64_t ToInt64(const std::wstring &str)
    {
#       ifdef _WIN32
        return (int64_t)_wtoi64(str.c_str());
#       else
        return (int64_t)wcstoll(str.c_str(), 0, 10);
#       endif
    }

    static float ToFloat(const std::string &str)
    {
        char *end_str = nullptr;
        return std::strtof(str.c_str(), &end_str);
    }

    static float ToFloat(const std::wstring &str)
    {
        wchar_t *end_str = nullptr;
        return std::wcstof(str.c_str(), &end_str);
    }

    static double ToDouble(const std::string &str)
    {
        char *end_str = nullptr;
        return std::strtod(str.c_str(), &end_str);
    }

    static double ToDouble(const std::wstring &str)
    {
        wchar_t *end_str = nullptr;
        return std::wcstod(str.c_str(), &end_str);
    }

	static void Tokenize(const std::string &str, std::vector<std::string> &tokens, const std::string &delims);
    static void Split(const std::string &str, std::vector<std::string> &tokens, const std::string &delims,
        const StrSplitOptions *options = nullptr);
}; //class String

class WString : public std::wstring
{
public:
    WString();
    WString(const wchar_t *ptr, std::wstring::size_type count, const std::allocator<wchar_t>& _Al = std::allocator<wchar_t>());
    WString(const wchar_t *ptr, const std::allocator<wchar_t> &_Al = std::allocator<wchar_t>());
    WString(const std::wstring &str, std::string::size_type roff = 0, std::wstring::size_type count = std::wstring::npos);
    WString(const std::wstring::size_type count, std::wstring::value_type ch);
	virtual ~WString();

	//Replace in the string the first occurrence of the specified substring with another specified string
    WString& ReplaceOne(const wchar_t *from, const wchar_t *to);
    WString& ReplaceOne(const std::wstring &from, const std::wstring &to) {
		return ReplaceOne(from.c_str(), to.c_str());
	}
	//Replace in the string all occurrences of the specified substring with another specified string
	WString& ReplaceAll(const wchar_t *from, const wchar_t *to);
	WString& ReplaceAll(const std::wstring &from, const std::wstring &to) {
		return ReplaceAll(from.c_str(), to.c_str());
	}

    WString &TrimLeft(const std::wstring &target=L" \r\n\t");
    WString &TrimRight(const std::wstring &target=L" \r\n\t");
    WString &Trim(const std::wstring &target=L" \r\n\t");

    WString &MakeLower();
    WString &MakeUpper();

    //Break the string into tokens
    void Tokenize(std::vector<WString> &tokens, const std::wstring &delims) const;
    void Split(std::vector<std::wstring> &tokens, const std::wstring &delims,
        const StrSplitOptions *options=nullptr);

//Static members
public:
    static bool IsNullOrEmpty(const wchar_t *wstr) {
        return wstr == nullptr || wstr[0] == L'\0';
    }

    static bool HasLetter(const wstring &str, const UnicodeTable &uct);
    static bool HasLetter(const wchar_t *wstr, const UnicodeTable &uct);
    static bool HasLetterOrDigit(const wstring &str, const UnicodeTable &uct);
    static bool HasLetterOrDigit(const wchar_t *str, const UnicodeTable &uct);

    static std::wstring& TrimLeft(std::wstring &str, const std::wstring &target = L" \r\n\t");
    static std::wstring& TrimRight(std::wstring &str, const std::wstring &target = L" \r\n\t");
    static std::wstring& Trim(std::wstring &str, const std::wstring &target = L" \r\n\t");

    static std::wstring& TrimSeparatorsLeft(std::wstring &str, const UnicodeTable &uct);
    static std::wstring& TrimSeparatorsRight(std::wstring &str, const UnicodeTable &uct);
    static std::wstring& TrimSeparators(std::wstring &str, const UnicodeTable &uct);

    static void MakeLower(std::wstring &lower, const UnicodeTable &char_table);
    static void MakeUpper(std::wstring &lower, const UnicodeTable &char_table);

    static uint32_t HashCodeNoCaseUInt32(const std::wstring &wstr, const UnicodeTable &char_table);
    static uint32_t HashCodeNoCaseUInt32(const wchar_t *wcs, uint32_t len, const UnicodeTable &char_table);
    static uint32_t HashCodeNoCaseUInt32(const wchar_t *wcs, const UnicodeTable &char_table);

    static void ReplaceOne(std::wstring &str, const wchar_t *from, const wchar_t *to);
    static void ReplaceOne(std::wstring &str, const wstring &from, const wstring &to);
    static void ReplaceAll(std::wstring &str, const wchar_t *from, const wchar_t *to);
    static void ReplaceAll(std::wstring &str, const wstring &from, const wstring &to);

    static int Cmp(const wchar_t *lhs, const wchar_t *rhs)
    {
        if (lhs == nullptr) {
            return rhs == nullptr ? 0 : -1;
        }
        else if (rhs == nullptr) {
            return 1;
        }

        return wcscmp(lhs, rhs);
    }

    static int CaseCmp(const wchar_t *lhs, const wchar_t *rhs);
    static int CaseCmp(const wstring &lhs, const wchar_t *rhs);
    static int CaseCmp(const wstring &lhs, const wstring &rhs);

    static void Tokenize(const std::wstring &str, std::vector<std::wstring> &tokens, const std::wstring &delims);
    static void Split(const std::wstring &str, std::vector<std::wstring> &tokens, const std::wstring &delims,
        const StrSplitOptions *pOptions=nullptr);
}; //class WString

} //end of namespace
