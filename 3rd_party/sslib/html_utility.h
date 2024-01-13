#pragma once

#include <string>
#include "unicode_table.h"
#include "sub_str_lookup_tree.h"

namespace sslib
{

using namespace std;

class HtmlUtility
{
public:
    bool Init();

    wstring HtmlTextDecoding(const wstring &src_text) const;
    void HtmlTextDecoding(wstring &target_text, const wchar_t *src_text, size_t src_len) const;
    void HtmlTextDecoding(wstring &target_text, const wstring &src_text) const;
    static void AppendText(wstring &target_text, const wstring &src_text);

protected:
    SubStrLookupTree html_code_trans_tree_;

protected:
    static int HexDigitValue(wchar_t ch)
    {
        if (ch >= L'0' && ch <= L'9') {
            return ch - L'0';
        }
        if (ch >= L'a' && ch <= L'f') {
            return 10 + (ch - L'a');
        }
        if (ch >= L'A' && ch <= L'F') {
            return 10 + (ch - L'A');
        }
        return -1;
    }

    static int DigitValue(wchar_t ch)
    {
        return ch >= L'0' && ch <= L'9' ? ch - L'0' : -1;
    }

protected:
    void InitSeparators();

    void InitAsciiSymbols();
    void InitLatin1SupplementSymbols();
    void InitMathSymbols();
    void InitMiscSymbols();

    void InitLatinLetters();
    void InitGreekLetters();
};

} //end of namespace
