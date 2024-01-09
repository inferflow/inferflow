#include "html_utility.h"
#include <vector>
#include "log.h"

using namespace std;

namespace sslib
{

bool HtmlUtility::Init()
{
    bool ret = true;

    //references:
    //    http://www.okelmann.com/homepage/unicode.htm
    //    https://en.wikipedia.org/wiki/List_of_XML_and_HTML_character_entity_references

    InitSeparators();
    InitAsciiSymbols();
    InitLatin1SupplementSymbols();
    InitMathSymbols();
    InitMiscSymbols();
    InitLatinLetters();
    InitGreekLetters();

    html_code_trans_tree_.AddString(L"&sect", L"§");
    html_code_trans_tree_.AddString(L"&hellip", L"…");
    html_code_trans_tree_.AddString(L"&acute", L"´");
    html_code_trans_tree_.AddString(L"&DiacriticalAcute", L"´");

    html_code_trans_tree_.AddString(L"&bdquo", L"„");
    html_code_trans_tree_.AddString(L"&ldquo", L"“");
    html_code_trans_tree_.AddString(L"&rdquo", L"”");
    html_code_trans_tree_.AddString(L"&lsquo", L"‘");
    html_code_trans_tree_.AddString(L"&rsquo", L"’");
    html_code_trans_tree_.AddString(L"&lsaquo", L"‹"); //U+2039
    html_code_trans_tree_.AddString(L"&rsaquo", L"›");

    html_code_trans_tree_.AddString(L"&bull", L"•");
    //html_code_trans_tree_.AddString(L"&middot", L"·");
    html_code_trans_tree_.AddString(L"&sdot", L"⋅"); //U+22C5 (8901)
    html_code_trans_tree_.AddString(L"&mdash", L"—");
    html_code_trans_tree_.AddString(L"&ndash", L"–");

    html_code_trans_tree_.AddString(L"&euro", L"\x20\xAC", 0x20AC);

    html_code_trans_tree_.AddString(L"&trade", L"\x21\x22", 0x2122);

    html_code_trans_tree_.AddString(L"&divide", L"\xF7", 0xF7);

    html_code_trans_tree_.AddString(L"&loz", L"\x25\xCA", 0x25CA);

    html_code_trans_tree_.AddString(L"&larr", L"←"); //U+2190
    html_code_trans_tree_.AddString(L"&leftarrow", L"←"); //U+2190
    html_code_trans_tree_.AddString(L"&LeftArrow", L"←"); //U+2190
    html_code_trans_tree_.AddString(L"&slarr", L"←"); //U+2190
    html_code_trans_tree_.AddString(L"&ShortLeftArrow", L"←"); //U+2190
    html_code_trans_tree_.AddString(L"&uarr", L"\x21\x91", 0x2191);
    html_code_trans_tree_.AddString(L"&rarr", L"→"); //0x2192
    html_code_trans_tree_.AddString(L"&rightarrow", L"→"); //0x2192
    html_code_trans_tree_.AddString(L"&RightArrow,", L"→"); //0x2192
    html_code_trans_tree_.AddString(L"&srarr", L"→"); //0x2192
    html_code_trans_tree_.AddString(L"&ShortRightArrow", L"→"); //0x2192
    html_code_trans_tree_.AddString(L"&darr", L"↓"); //0x2193
    html_code_trans_tree_.AddString(L"&downarrow", L"↓"); //0x2193
    html_code_trans_tree_.AddString(L"&DownArrow", L"↓"); //0x2193
    html_code_trans_tree_.AddString(L"&ShortDownArrow", L"↓"); //0x2193
    html_code_trans_tree_.AddString(L"&harr", L"\x21\x94", 0x2194);

    return ret;
}

wstring HtmlUtility::HtmlTextDecoding(const std::wstring &src_text) const
{
    wstring ret_str;
    HtmlTextDecoding(ret_str, src_text.c_str(), src_text.size());
    return ret_str;
}

void HtmlUtility::HtmlTextDecoding(wstring &target_text, const wstring &src_text) const
{
    HtmlTextDecoding(target_text, src_text.c_str(), src_text.size());
}

void HtmlUtility::HtmlTextDecoding(wstring &target_text, const wchar_t *src_text, size_t src_len) const
{
    target_text.clear();
    const SubStrLookupTree *html_code_tree = nullptr;
    wstring sub_str;

    bool be_skip_semicolon = false, be_escape_mode = false, be_digits = false, is_hex = false;
    int digits_value = 0, digit_val = 0;
    for (size_t char_idx = 0; char_idx <= src_len; char_idx++)
    {
        wchar_t wch = char_idx < src_len ? src_text[char_idx] : L' ';
        const SubStrLookupTree *ret_tree_ptr = html_code_tree != nullptr
            ? html_code_tree->Find(wch) : nullptr;
        //LogKeyInfo("wch: %lc", wch);
        //if (ret_tree_ptr == nullptr)
        //{
        //    if (html_code_tree == nullptr) {
        //        LogWarning("null html code tree");
        //    }
        //    else {
        //        LogKeyInfo(L"html code tree child count: %u", html_code_tree->ChildCount());
        //    }
        //}

        if (ret_tree_ptr != nullptr && char_idx < src_len)
        {
            if (char_idx < src_len) {
                sub_str += wch;
            }
            html_code_tree = ret_tree_ptr;
        }
        else //ret_tree_ptr == nullptr
        {
            be_skip_semicolon = false;
            if (be_digits)
            {
                digit_val = is_hex ? HexDigitValue(wch) : DigitValue(wch);
                if (digit_val >= 0)
                {
                    digits_value *= (is_hex ? 16 : 10);
                    digits_value += digit_val;
                    if (digits_value > 65535) {
                        digits_value = -1;
                    }
                    continue;
                }
                else
                {
                    target_text += (digits_value >= 0 ? (wchar_t)digits_value : L' ');
                    be_digits = false;
                    is_hex = false;
                    be_skip_semicolon = true;
                }
            }
            else if (html_code_tree != nullptr)
            {
                if (html_code_tree->int_value() <= 0xFFFF)
                {
                    target_text += html_code_tree->str_value();
                    //LogKeyInfo(L"target so far: %ls", target_text.c_str());
                    be_skip_semicolon = true;
                }
                else
                {
                    target_text += sub_str;
                }
            }

            html_code_tree = nullptr;
            sub_str.clear();

            if (char_idx >= src_len) {
                break;
            }

            be_escape_mode = false;
            if (wch == L'&' && char_idx + 1 < src_len)
            {
                //LogKeyInfo("Find &");
                if (src_text[char_idx + 1] == L'#')
                {
                    digit_val = -1;
                    if (char_idx + 2 < src_len)
                    {
                        wchar_t ch2 = src_text[char_idx + 2];
                        is_hex = ch2 == L'x' || ch2 == L'X';
                        if (is_hex && char_idx + 3 < src_len)
                        {
                            wchar_t ch3 = src_text[char_idx + 3];
                            digit_val = HexDigitValue(ch3);
                        }
                        else
                        {
                            digit_val = DigitValue(ch2);
                        }
                    }

                    if (digit_val >= 0)
                    {
                        digits_value = digit_val;
                        be_escape_mode = true;
                        be_digits = true;
                        char_idx += (is_hex ? 3 : 2);
                    }
                }
                else
                {
                    sub_str += wch;
                    //LogKeyInfo("root child count: %u", html_code_trans_tree_.ChildCount());
                    html_code_tree = html_code_trans_tree_.Find(wch);
                    be_escape_mode = true;
                }
            }

            if (!be_escape_mode && !(wch == L';' && be_skip_semicolon))
            {
                target_text += wch;
            }
        }
    }

    //if (target_text.find(L"&nbsp") != wstring::npos)
    //{
    //    wstring input_str(src_text, src_len);
    //    LogWarning(L"having nbsp: %ls", input_str.c_str());
    //}
}

//static
void HtmlUtility::AppendText(std::wstring &target_text, const std::wstring &src_text)
{
    for (size_t char_idx = 0; char_idx < src_text.size(); char_idx++)
    {
        wchar_t wch = src_text[char_idx];

        //
        if (wch == L'\x13') {
            wch = L'-';
        }

        bool is_space_type = wch == L' ' || wch == L'\n' || wch == L'\r' || wch == L'\t';
        bool is_space_end = !target_text.empty() && target_text[target_text.size() - 1] == L' ';

        if (is_space_type)
        {
            if (!is_space_end) {
                target_text += L' ';
            }
        }
        else
        {
            target_text += wch;
        }
    }
}

void HtmlUtility::InitSeparators()
{
    html_code_trans_tree_.AddString(L"&nbsp", L"\xA0", 0xA0);
    html_code_trans_tree_.AddString(L"&NonBreakingSpace", L"\xA0", 0xA0);
    html_code_trans_tree_.AddString(L"&ensp", L"\x20\x02", 0x2002);
    html_code_trans_tree_.AddString(L"&emsp", L"\x20\x03", 0x2003);
    html_code_trans_tree_.AddString(L"&thinsp", L"\x20\x09", 0x2009);
}

void HtmlUtility::InitAsciiSymbols()
{
    const wchar_t *html_entity_array[][3] =
    {
        {L"&excl", L"!", L"U+0021 (33)"},
        {L"&quot", L"\"", L"U+0022 (34)"},
        {L"&QUOT", L"\"", L"U+0022 (34)"},
        {L"&num", L"#", L"U+0023 (35)"},
        {L"&dollar", L"$", L"U+0024 (36)"},
        {L"&percnt", L"%", L"U+0025 (37)"},
        {L"&amp", L"&", L"U+0026 (38)"},
        {L"&AMP", L"&", L"U+0026 (38)"},
        {L"&apos", L"'", L"U+0027 (39)"},
        {L"&lpar", L"(", L"U+0028 (40)"},
        {L"&rpar", L")", L"U+0029 (41)"},
        {L"&ast", L"*", L"U+002A (42)"},
        {L"&midast", L"*", L"U+002A (42)"},
        {L"&add", L"+", L"U+002B (43)"},
        {L"&plus", L"+", L"U+002B (43)"},
        {L"&comma", L",", L"U+002C (44)"},
        {L"&period", L".", L"U+002E (46)"},
        {L"&sol", L"/", L"U+002F (47)"}, //solidus
        {L"&colon", L":", L"U+003A (58)"},
        {L"&semi", L";", L"U+003B (59)"},
        {L"&lt", L"<", L"U+003C (60)"},
        {L"&LT", L"<", L"U+003C (60)"},
        {L"&equal", L"=", L"U+003D (61)"},
        {L"&gt", L">", L"U+003E (62)"},
        {L"&GT", L">", L"U+003E (62)"},
        {L"&quest", L"?", L"U+003F (63)"},
        {L"&commat", L"@", L"U+0040 (64)"},
        {L"&lsqb", L"[", L"U+005B (91)"},
        {L"&lbrack", L"[", L"U+005B (91)"},
        {L"&bsol", L"\\", L"U+005C(92)"},
        {L"&rsqb", L"]", L"U+005D (93)"},
        {L"&rbrack", L"]", L"U+005D (93)"},
        {L"&Hat", L"^", L"U+005E (94)"}, //circumflex accent (hat)
        {L"&lowbar", L"_", L"U+005F (95)"},
        {L"&grave", L"`", L"U+0060 (96)"},
        {L"&DiacriticalGrave", L"`", L"U+0060 (96)"},
        {L"&lcub", L"{", L"U+007B (123)"},
        {L"&lbrace", L"{", L"U+007B (123)"},
        {L"&verbar", L"|", L"U+007C (124)"},
        {L"&vert", L"|", L"U+007C (124)"},
        {L"&VerticalLine", L"|", L"U+007C (124)"},
        {L"&rcub", L"}", L"U+007D (125)"},
        {L"&rbrace", L"}", L"U+007D (125)"},
    };

    int entity_count = (int)sizeof(html_entity_array) / sizeof(const wchar_t*) / 3;
    //LogKeyInfo("Ascii symbol count: %d", entity_count);
    for (int idx = 0; idx < entity_count; idx++)
    {
        const auto entity_info = html_entity_array[idx];
        html_code_trans_tree_.AddString(entity_info[0], entity_info[1]);
    }
}

void HtmlUtility::InitLatin1SupplementSymbols()
{
    const wchar_t *html_entity_array[][3] =
    {
        {L"&iexcl", L"¡", L"U+00A1 (161)"},
        {L"&cent", L"¢", L"U+00A2 (162)"},
        {L"&pound", L"£", L"U+00A3 (163)"},
        {L"&curren", L"¤", L"U+00A4 (164)"},
        {L"&yen", L"¥", L"U+00A5 (165)"},
        {L"&brvbar", L"¦", L"U+00A6 (166)"},
        {L"&sect", L"§", L"U+00A7 (167)"},
        {L"&Dot", L"¨", L"U+00A8 (168)"},
        {L"&die", L"¨", L"U+00A8 (168)"},
        {L"&DoubleDot", L"¨", L"U+00A8 (168)"},
        {L"&uml", L"¨", L"U+00A8 (168)"},
        {L"&copy", L"©", L"U+00A9 (169)"},
        {L"&COPY", L"©", L"U+00A9 (169)"},
        {L"&ordf", L"ª", L"U+00AA (170)"},
        {L"&laquo", L"«", L"U+00AB (171)"},
        {L"&not", L"¬", L"U+00AC (172)"},
        {L"&shy", L" ", L"U+00AD (173)"}, //soft hyphen (discretionary hyphen)
        {L"&reg", L"®", L"U+00AE (174)"},
        {L"&circledR", L"®", L"U+00AE (174)"},
        {L"&REG", L"®", L"U+00AE (174)"},
        {L"&macr", L"¯", L"U+00AF (175)"},
        {L"&OverBar", L"¯", L"U+00AF (175)"},
        {L"&strns", L"¯", L"U+00AF (175)"},
        {L"&deg", L"°", L"U+00B0 (176)"},
        {L"&plusmn", L"±", L"U+00B1 (177)"},
        {L"&pm", L"±", L"U+00B1 (177)"},
        {L"&PlusMinus", L"±", L"U+00B1 (177)"},
        {L"&sup2", L"²", L"U+00B2 (178)"},
        {L"&sup3", L"³", L"U+00B3 (179)"},
        {L"&acute", L"´", L"U+00B4 (180)"},
        {L"&DiacriticalAcute", L"´", L"U+00B4 (180)"},
        {L"&micro", L"µ", L"U+00B5 (181)"},
        {L"&para", L"¶", L"U+00B6 (182)"},
        {L"&middot", L"·", L"U+00B7 (183)"},
        {L"&centerdot", L"·", L"U+00B7 (183)"},
        {L"&CenterDot", L"·", L"U+00B7 (183)"},
        {L"&cedil", L"¸", L"U+00B8 (184)"},
        {L"&Cedilla", L"¸", L"U+00B8 (184)"},
        {L"&sup1", L"¹", L"U+00B9 (185)"},
        {L"&ordm", L"º", L"U+00BA (186)"},
        {L"&raquo", L"»", L"U+00BB (187)"},
        {L"&frac14", L"¼", L"U+00BC (188)"},
        {L"&frac12", L"½", L"U+00BD (189)"},
        {L"&half", L"½", L"U+00BD (189)"},
        {L"&frac34", L"¾", L"U+00BE (190)"},
        {L"&iquest", L"¿", L"U+00BF (191)"},
    };

    int entity_count = (int)sizeof(html_entity_array) / sizeof(const wchar_t*) / 3;
    //LogKeyInfo("Latin-1 supplement symbol count: %d", entity_count);
    for (int idx = 0; idx < entity_count; idx++)
    {
        const auto entity_info = html_entity_array[idx];
        html_code_trans_tree_.AddString(entity_info[0], entity_info[1]);
    }
}

void HtmlUtility::InitMathSymbols()
{
    const wchar_t *html_entity_array[][3] =
    {
        {L"&forall", L"∀", L"U+2200 (8704)"},
        {L"&ForAll", L"∀", L"U+2200 (8704)"},
        {L"&comp", L"∁", L"U+2201 (8705)"},
        {L"&complement", L"∁", L"U+2201 (8705)"},
        {L"&part", L"∂", L"U+2202 (8706)"}, //partial differential
        {L"&PartialD", L"∂", L"U+2202 (8706)"},
        {L"&exist", L"∃", L"U+2203 (8707)"}, //there exists
        {L"&Exists", L"∃", L"U+2203 (8707)"},
        {L"&nexist", L"∄", L"U+2204 (8708)"}, //does not exist
        {L"&NotExists", L"∄", L"U+2204 (8708)"},
        {L"&nexists", L"∄", L"U+2204 (8708)"},
        {L"&empty", L"∅", L"U+2205 (8709)"},
        {L"&emptyset", L"∅", L"U+2205 (8709)"},
        {L"&emptyv", L"∅", L"U+2205 (8709)"},
        {L"&varnothing", L"∅", L"U+2205 (8709)"},
        {L"&nabla", L"∇", L"U+2207 (8711)"}, //del or nabla (vector differential operator)
        {L"&Del", L"∇", L"U+2207 (8711)"},
        {L"&isin", L"∈", L"U+2208 (8712)"},
        {L"&isinv", L"∈", L"U+2208 (8712)"},
        {L"&Element", L"∈", L"U+2208 (8712)"},
        {L"&in", L"∈", L"U+2208 (8712)"},
        {L"&notin", L"∉", L"U+2209 (8713)"},
        {L"&NotElement", L"∉", L"U+2209 (8713)"},
        {L"&notinva", L"∉", L"U+2209 (8713)"},
        {L"&ni", L"∋", L"U+220B (8715)"},
        {L"&niv", L"∋", L"U+220B (8715)"},
        {L"&ReverseElement", L"∋", L"U+220B (8715)"},
        {L"&SuchThat", L"∋", L"U+220B (8715)"},
        {L"&notni", L"∌", L"U+220C (8716)"},
        {L"&notniva", L"∌", L"U+220C (8716)"},
        {L"&NotReverseElement", L"∌", L"U+220C (8716)"},
        {L"&prod", L"∏", L"U+220F (8719)"},
        {L"&Product", L"∏", L"U+220F (8719)"},
        {L"&coprod", L"∐", L"U+2210 (8720)"},
        {L"&Coproduct", L"∐", L"U+2210 (8720)"},
        {L"&sum", L"∑", L"U+2211 (8721)"},
        {L"&Sum", L"∑", L"U+2211 (8721)"},
        {L"&minus", L"−", L"U+2212 (8722)"},
        {L"&mnplus", L"∓", L"U+2213 (8723)"},
        {L"&mp", L"∓", L"U+2213 (8723)"},
        {L"&MinusPlus", L"∓", L"U+2213 (8723)"},
        {L"&plusdo", L"∔", L"U+2214 (8724)"},
        {L"&dotplus", L"∔", L"U+2214 (8724)"},
        {L"&setmn", L"∖", L"U+2216 (8726)"},
        {L"&setminus", L"∖", L"U+2216 (8726)"},
        {L"&Backslash", L"∖", L"U+2216 (8726)"},
        {L"&ssetmn", L"∖", L"U+2216 (8726)"},
        {L"&smallsetminus", L"∖", L"U+2216 (8726)"},
        {L"&lowast", L"∗", L"U+2217 (8727)"},
        {L"&compfn", L"∘", L"U+2218 (8728)"},
        {L"&SmallCircle", L"∘", L"U+2218 (8728)"},
        {L"&radic", L"√", L"U+221A (8730)"},
        {L"&Sqrt", L"√", L"U+221A (8730)"},
        {L"&prop", L"∝", L"U+221D (8733)"},
        {L"&propto", L"∝", L"U+221D (8733)"},
        {L"&Proportional", L"∝", L"U+221D (8733)"},
        {L"&vprop", L"∝", L"U+221D (8733)"},
        {L"&varpropto", L"∝", L"U+221D (8733)"},
        {L"&infin", L"∞", L"U+221E (8734)"},
        {L"&angrt", L"∟", L"U+221F (8735)"},
        {L"&ang", L"∠", L"U+2220 (8736)"},
        {L"&angle", L"∠", L"U+2220 (8736)"},
        {L"&angmsd", L"∡", L"U+2221 (8737)"},
        {L"&measuredangle", L"∡", L"U+2221 (8737)"},
        {L"&angsph", L"∡", L"U+2222 (8738)"},
        {L"&mid", L"∣", L"U+2223 (8739)"},
        {L"&VerticalBar", L"∣", L"U+2223 (8739)"},
        {L"&smid", L"∣", L"U+2223 (8739)"},
        {L"&shortmid", L"∣", L"U+2223 (8739)"},
        {L"&nmid", L"∤", L"U+2224 (8740)"},
        {L"&NotVerticalBar", L"∤", L"U+2224 (8740)"},
        {L"&nsmid", L"∤", L"U+2224 (8740)"},
        {L"&nshortmid", L"∤", L"U+2224 (8740)"},
        {L"&par", L"∥", L"U+2225 (8741)"},
        {L"&parallel", L"∥", L"U+2225 (8741)"},
        {L"&DoubleVerticalBar", L"∥", L"U+2225 (8741)"},
        {L"&spar", L"∥", L"U+2225 (8741)"},
        {L"&shortparallel", L"∥", L"U+2225 (8741)"},
        {L"&npar", L"∦", L"U+2226 (8742)"},
        {L"&nparallel", L"∦", L"U+2226 (8742)"},
        {L"&NotDoubleVerticalBar", L"∦", L"U+2226 (8742)"},
        {L"&nspar", L"∦", L"U+2226 (8742)"},
        {L"&nshortparallel", L"∦", L"U+2226 (8742)"},
        {L"&and", L"∧", L"U+2227 (8743)"},
        {L"&wedge", L"∧", L"U+2227 (8743)"},
        {L"&or", L"∨", L"U+2228 (8744)"},
        {L"&vee", L"∨", L"U+2228 (8744)"},
        {L"&cap", L"∩", L"U+2229 (8745)"},
        {L"&cup", L"∪", L"U+222A (8746)"},
        {L"&int", L"∫", L"U+222B (8747)"},
        {L"&Integral", L"∫", L"U+222B (8747)"},
        {L"&Int", L"∬", L"U+222C (8748)"},
        {L"&tint", L"∭", L"U+222D (8749)"},
        {L"&iiint", L"∭", L"U+222D (8749)"},
        {L"&conint", L"∮", L"U+222E (8750)"},
        {L"&oint", L"∮", L"U+222E (8750)"},
        {L"&ContourIntegral", L"∮", L"U+222E (8750)"},
        {L"&Conint", L"∯", L"U+222F (8751)"},
        {L"&DoubleContourIntegral", L"∯", L"U+222F (8751)"},
        {L"&Cconint", L"∰", L"U+2230 (8752)"},
        {L"&cwint", L"∱", L"U+2231 (8753)"},
        {L"&cwconint", L"∲", L"U+2232 (8754)"},
        {L"&ClockwiseContourIntegral", L"∲", L"U+2232 (8754)"},
        {L"&awconint", L"∲", L"U+2233 (8755)"},
        {L"&CounterClockwiseContourIntegral", L"∲", L"U+2233 (8755)"},
        {L"&there4", L"∴", L"U+2234 (8756)"},
        {L"&therefore", L"∴", L"U+2234 (8756)"},
        {L"&Therefore", L"∴", L"U+2234 (8756)"},
        {L"&becaus", L"∵", L"U+2235 (8757)"},
        {L"&because", L"∵", L"U+2235 (8757)"},
        {L"&Because", L"∵", L"U+2235 (8757)"},
        {L"&ratio", L"∶", L"U+2236 (8758)"},
        {L"&Colon", L"∷", L"U+2237 (8759)"},
        {L"&Proportion", L"∷", L"U+2237 (8759)"},
        {L"&minusd", L"∸", L"U+2238 (8760)"},
        {L"&dotminus", L"∸", L"U+2238 (8760)"},
        {L"&mDDot", L"∺", L"U+223A (8762)"},
        {L"&homtht", L"∻", L"U+223B (8763)"},
        { L"&sim", L"∼", L"U+223C (8764)" },
        { L"&Tilde", L"∼", L"U+223C (8764)" },
        { L"&thksim", L"∼", L"U+223C (8764)" },
        { L"&thicksim", L"∼", L"U+223C (8764)" },
        { L"&bsim", L"∽", L"U+223D (8765)" },
        { L"&backsim", L"∽", L"U+223D (8765)" },
        { L"&ac", L"∾", L"U+223E (8766)" },
        { L"&mstpos", L"∾", L"U+223E (8766)" },
        { L"&acd", L"∿", L"U+223F (8767)" },
        { L"&wreath", L"≀", L"U+2240 (8768)" },
        { L"&VerticalTilde", L"≀", L"U+2240 (8768)" },
        { L"&wr", L"≀", L"U+2240 (8768)" },
        { L"&nsim", L"≁", L"U+2241 (8769)" },
        { L"&NotTilde", L"≁", L"U+2241 (8769)" },
        { L"&esim", L"≂", L"U+2242 (8770)" },
        { L"&EqualTilde", L"≂", L"U+2242 (8770)" },
        { L"&eqsim", L"≂", L"U+2242 (8770)" },
        { L"&sime", L"≃", L"U+2243 (8771)" },
        { L"&TildeEqual", L"≃", L"U+2243 (8771)" },
        { L"&simeq", L"≃", L"U+2243 (8771)" },
        { L"&nsime", L"≄", L"U+2244 (8772)" },
        { L"&nsimeq", L"≄", L"U+2244 (8772)" },
        { L"&NotTildeEqual", L"≄", L"U+2244 (8772)" },
        { L"&cong", L"≅", L"U+2245 (8773)" },
        { L"&TildeFullEqual", L"≅", L"U+2245 (8773)" },
        { L"&simne", L"≆", L"U+2246 (8774)" },
        { L"&ncong", L"≇", L"U+2247 (8775)" },
        { L"&NotTildeFullEqual", L"≇", L"U+2247 (8775)" },
        { L"&asymp", L"≈", L"U+2248 (8776)" },
        { L"&ap", L"≈", L"U+2248 (8776)" },
        { L"&TildeTilde", L"≈", L"U+2248 (8776)" },
        { L"&approx", L"≈", L"U+2248 (8776)" },
        { L"&thkap", L"≈", L"U+2248 (8776)" },
        { L"&thickapprox", L"≈", L"U+2248 (8776)" },
        { L"&nap", L"≉", L"U+2249 (8777)" },
        { L"&NotTildeTilde", L"≉", L"U+2249 (8777)" },
        { L"&napprox", L"≉", L"U+2249 (8777)" },
        { L"&ape", L"≊", L"U+224A (8778)" },
        { L"&approxeq", L"≊", L"U+224A (8778)" },
        { L"&apid", L"≋", L"U+224B (8779)" },
        { L"&bcong", L"≌", L"U+224C (8780)" },
        { L"&backcong", L"≌", L"U+224C (8780)" },
        { L"&asympeq", L"≍", L"U+224D (8781)" },
        { L"&CupCap", L"≍", L"U+224D (8781)" },
        { L"&bump", L"≎", L"U+224E (8782)" },
        { L"&HumpDownHump", L"≎", L"U+224E (8782)" },
        { L"&Bumpeq", L"≎", L"U+224E (8782)" },
        { L"&bumpe", L"≏", L"U+224F (8783)" },
        { L"&HumpEqual", L"≏", L"U+224F (8783)" },
        { L"&Humpeq", L"≏", L"U+224F (8783)" },
        { L"&esdot", L"≐", L"U+2250 (8784)" },
        { L"&DotEqual", L"≐", L"U+2250 (8784)" },
        { L"&doteq", L"≐", L"U+2250 (8784)" },
        { L"&edot", L"≑", L"U+2251 (8785)" },
        { L"&doteqdot", L"≑", L"U+2251 (8785)" },
        { L"&efdot", L"≒", L"U+2252 (8786)" },
        { L"&fallingdotseq", L"≒", L"U+2252 (8786)" },
        { L"&erdot", L"≓", L"U+2253 (8787)" },
        { L"&risingdotseq", L"≓", L"U+2253 (8787)" },
        { L"&colone", L"≔", L"U+2254 (8788)" },
        { L"&coloneq", L"≔", L"U+2254 (8788)" },
        { L"&Assign", L"≔", L"U+2254 (8788)" },
        { L"&ecolon", L"≕", L"U+2255 (8789)" },
        { L"&eqcolon", L"≕", L"U+2255 (8789)" },
        { L"&ecir", L"≖", L"U+2256 (8790)" },
        { L"&eqcirc", L"≖", L"U+2256 (8790)" },
        { L"&cire", L"≗", L"U+2257 (8791)" },
        { L"&circeq", L"≗", L"U+2257 (8791)" },
        { L"&wedgeq", L"≙", L"U+2259 (8793)" },
        { L"&veeeq", L"≚", L"U+225A (8794)" },
        { L"&trie", L"≜", L"U+225C (8796)" },
        { L"&triangleq", L"≜", L"U+225C (8796)" },
        { L"&equest", L"≟", L"U+225F (8799)" },
        { L"&questeq", L"≟", L"U+225F (8799)" },
        { L"&ne", L"≠", L"U+2260 (8800)" },
        { L"&NotEqual", L"≠", L"U+2260 (8800)" },
        { L"&equiv", L"≡", L"U+2261 (8801)" },
        { L"&NotCongruent", L"≢", L"U+2262 (8802)" },
        { L"&le", L"≤", L"U+2264 (8804)" },
        { L"&LessEqual", L"≤", L"U+2264 (8804)" },
        { L"&leq", L"≤", L"U+2264 (8804)" },
        { L"&ge", L"≥", L"U+2265 (8805)" },
        { L"&GreaterEqual", L"≥", L"U+2265 (8805)" },
        { L"&geq", L"≥", L"U+2265 (8805)" },
        { L"&lE", L"≦", L"U+2266 (8806)" },
        { L"&LessFullEqual", L"≦", L"U+2266 (8806)" },
        { L"&leqq", L"≦", L"U+2266 (8806)" },
        { L"&gE", L"≧", L"U+2267 (8807)" },
        { L"&GreaterFullEqual", L"≧", L"U+2267 (8807)" },
        { L"&geqq", L"≧", L"U+2267 (8807)" },
        { L"&lnE", L"≨", L"U+2268 (8808)" },
        { L"&lneqq", L"≨", L"U+2268 (8808)" },
        { L"&gnE", L"≩", L"U+2269 (8809)" },
        { L"&gneqq", L"≩", L"U+2269 (8809)" },
        { L"&Lt", L"≪", L"U+226A (8810)" },
        { L"&NestedLessLess", L"≪", L"U+226A (8810)" },
        { L"&ll", L"≪", L"U+226A (8810)" },
        { L"&Gt", L"≫", L"U+226B (8811)" },
        { L"&NestedGreaterGreater", L"≫", L"U+226B (8811)" },
        { L"&gg", L"≫", L"U+226B (8811)" },
        { L"&twixt", L"≬", L"U+226C (8812)" },
        { L"&between", L"≬", L"U+226C (8812)" },
        { L"&NotCupCap", L"≭", L"U+226D (8813)" },
        { L"&nlt", L"≮", L"U+226E (8814)" },
        { L"&NotLess", L"≮", L"U+226E (8814)" },
        { L"&nless", L"≮", L"U+226E (8814)" },
        { L"&ngt", L"≯", L"U+226F (8815)" },
        { L"&NotGreater", L"≯", L"U+226F (8815)" },
        { L"&ngtr", L"≯", L"U+226F (8815)" },
        { L"&nle", L"≰", L"U+2270 (8816)" },
        { L"&NotLessEqual", L"≰", L"U+2270 (8816)" },
        { L"&nleq", L"≰", L"U+2270 (8816)" },
        { L"&nge", L"≱", L"U+2271 (8817)" },
        { L"&NotGreaterEqual", L"≱", L"U+2271 (8817)" },
        { L"&ngeq", L"≱", L"U+2271 (8817)" },
        { L"&lsim", L"≲", L"U+2272 (8818)" },
        { L"&LessTilde", L"≲", L"U+2272 (8818)" },
        { L"&lesssim", L"≲", L"U+2272 (8818)" },
        { L"&gsim", L"≳", L"U+2273 (8819)" },
        { L"&GreaterTilde", L"≳", L"U+2273 (8819)" },
        { L"&gtrsimsim", L"≳", L"U+2273 (8819)" },
        { L"&nlsim", L"≴", L"U+2274 (8820)" },
        { L"&NotLessTilde", L"≴", L"U+2274 (8820)" },
        { L"&ngsim", L"≵", L"U+2275 (8821)" },
        { L"&NotGreaterTilde", L"≵", L"U+2275 (8821)" },
        { L"&lg", L"≶", L"U+2276 (8822)" },
        { L"&lessgtr", L"≶", L"U+2276 (8822)" },
        { L"&LessGreater", L"≶", L"U+2276 (8822)" },
        { L"&gl", L"≷", L"U+2277 (8823)" },
        { L"&gtrless", L"≷", L"U+2277 (8823)" },
        { L"&GreaterLess", L"≷", L"U+2277 (8823)" },
        { L"&ntlg", L"≸", L"U+2278 (8824)" },
        { L"&NotLessGreater", L"≸", L"U+2278 (8824)" },
        { L"&ntgl", L"≹", L"U+2279 (8825)" },
        { L"&GreaterLess", L"≹", L"U+2279 (8825)" },
        { L"&pr", L"≺", L"U+227A (8826)" },
        { L"&Precedes", L"≺", L"U+227A (8826)" },
        { L"&prec", L"≺", L"U+227A (8826)" },
        { L"&sc", L"≻", L"U+227B (8827)" },
        { L"&Succeeds", L"≻", L"U+227B (8827)" },
        { L"&succ", L"≻", L"U+227B (8827)" },
        { L"&prcue", L"≼", L"U+227C (8828)" },
        { L"&PrecedesSlantEqual", L"≼", L"U+227C (8828)" },
        { L"&preccurlyeq", L"≼", L"U+227C (8828)" },
        { L"&sccue", L"≽", L"U+227D (8829)" },
        { L"&SucceedsSlantEqual", L"≽", L"U+227D (8829)" },
        { L"&succcurlyeq", L"≽", L"U+227D (8829)" },
        { L"&prsim", L"≾", L"U+227E (8830)" },
        { L"&precsim", L"≾", L"U+227E (8830)" },
        { L"&PrecedesTilde", L"≾", L"U+227E (8830)" },
        { L"&sccue", L"≿", L"U+227F (8831)" },
        { L"&SucceedsSlantEqual", L"≿", L"U+227F (8831)" },
        { L"&succcurlyeq", L"≿", L"U+227F (8831)" },
        { L"&npr", L"⊀", L"U+2280 (8832)" },
        { L"&nprec", L"⊀", L"U+2280 (8832)" },
        { L"&NotPrecedes", L"⊀", L"U+2280 (8832)" },
        { L"&nsc", L"⊁", L"U+2281 (8833)" },
        { L"&nsucc", L"⊁", L"U+2281 (8833)" },
        { L"&NotSucceeds", L"⊁", L"U+2281 (8833)" },
        { L"&sub", L"⊂", L"U+2282 (8834)" },
        { L"&subset", L"⊂", L"U+2282 (8834)" },
        { L"&sup", L"⊃", L"U+2283 (8835)" },
        { L"&supset", L"⊃", L"U+2283 (8835)" },
        { L"&Superset", L"⊃", L"U+2283 (8835)" },
        { L"&nsub", L"⊄", L"U+2284 (8836)" },
        { L"&nsup", L"⊅", L"U+2285 (8837)" },
    };

    int entity_count = (int)sizeof(html_entity_array) / sizeof(const wchar_t*) / 3;
    //LogKeyInfo("Math symbol count: %d", entity_count);
    for (int idx = 0; idx < entity_count; idx++)
    {
        const auto entity_info = html_entity_array[idx];
        html_code_trans_tree_.AddString(entity_info[0], entity_info[1]);
    }
}

void HtmlUtility::InitMiscSymbols()
{
    const wchar_t *html_entity_array[][3] =
    {
        {L"&permil", L"‰", L"U+2030 (8240)"},
        {L"&pertenk", L"‱", L"U+2031 (8241)"},
        {L"&prime", L"′", L"U+2032 (8242)"},
        {L"&Prime", L"″", L"U+2033 (8243)"},
        {L"&tprime", L"‴", L"U+2034 (8244)"},
        {L"&bprime", L"‵", L"U+2035 (8245)"},
        {L"&backprime", L"‵", L"U+2035 (8245)"},
        {L"&lsaquo", L"‹", L"U+2039 (8249)"},
        {L"&rsaquo", L"›", L"U+203A (8250)"},

        {L"&oplus", L"⊕", L"U+2295 (8853)"},
        {L"&CirclePlus", L"⊕", L"U+2295 (8853)"},
        {L"&ominus", L"⊖", L"U+2296 (8854)"},
        {L"&CircleMinus", L"⊖", L"U+2296 (8854)"},
        {L"&otimes", L"⊗", L"U+2297 (8855)"},
        {L"&CircleTimes", L"⊗", L"U+2297 (8855)"},
        {L"&osol", L"⊘", L"U+2298 (8856)"},

        {L"&perp", L"⊥", L"U+22A5 (8869)"},
        {L"&bottom", L"⊥", L"U+22A5 (8869)"},
        {L"&bot", L"⊥", L"U+22A5 (8869)"},
        {L"&UpTee", L"⊥", L"U+22A5 (8869)"},

        {L"&lceil", L"⌈", L"U+2308 (8968)"},
        {L"&LeftCeiling", L"⌈", L"U+2308 (8968)"},
        {L"&rceil", L"⌉", L"U+2309 (8969)"},
        {L"&RightCeiling", L"⌉", L"U+2309 (8969)"},
        {L"&lfloor", L"⌊", L"U+230A (8970)"},
        {L"&LeftFloor", L"⌊", L"U+230A (8970)"},
        {L"&rfloor", L"⌋", L"U+230B (8971)"},
        {L"&RightFloor", L"⌋", L"U+230B (8971)"},

        {L"&starf", L"★", L"U+2605 (9733)"},
        {L"&bigstar", L"★", L"U+2605 (9733)"},
        {L"&star", L"☆", L"U+2606 (9734)"},
        {L"&phone", L"☎", L"U+260E (9742)"},
        {L"&female", L"♀", L"U+2640 (9792)"},
        {L"&male", L"♂", L"U+2642 (9794)"},
        {L"&spades", L"♠", L"U+2660 (9824)"},
        {L"&spadesuit", L"♠", L"U+2660 (9824)"},
        {L"&clubs", L"♣", L"U+2663 (9827)"},
        {L"&clubsuit", L"♣", L"U+2663 (9827)"},
        {L"&hearts", L"♥", L"U+2665 (9829)"},
        {L"&heartsuit", L"♥", L"U+2665 (9829)"},
        {L"&diams", L"♦", L"U+2666 (9830)"},
        {L"&diamondsuit", L"♦", L"U+2666 (9830)"},
    };

    int entity_count = (int)sizeof(html_entity_array) / sizeof(const wchar_t*) / 3;
    //LogKeyInfo("Misc symbol count: %d", symbol_count);
    for (int idx = 0; idx < entity_count; idx++)
    {
        const auto entity_info = html_entity_array[idx];
        html_code_trans_tree_.AddString(entity_info[0], entity_info[1]);
    }
}

void HtmlUtility::InitLatinLetters()
{
    const wchar_t *html_entity_array[][3] =
    {
        {L"&Agrave", L"À", L"U+00C0 (192)"},
        {L"&Aacute", L"Á", L"U+00C1 (193)"},
        {L"&Acirc", L"Â", L"U+00C2 (194)"},
        {L"&Atilde", L"Ã", L"U+00C3 (195)"},
        {L"&Auml", L"Ä", L"U+00C4 (196)"},
        {L"&Aring", L"Å", L"U+00C5 (197)"},
        {L"&AElig", L"Æ", L"U+00C6 (198)"},
        {L"&Ccedil", L"Ç", L"U+00C7 (199)"},
        {L"&Egrave", L"È", L"U+00C8 (200)"},
        {L"&Eacute", L"É", L"U+00C9 (201)"},
        {L"&Ecirc", L"Ê", L"U+00CA (202)"},
        {L"&Euml", L"Ë", L"U+00CB (203)"},
        {L"&Igrave", L"Ì", L"U+00CC (204)"},
        {L"&Iacute", L"Í", L"U+00CD (205)"},
        {L"&Icirc", L"Î", L"U+00CE (206)"},
        {L"&Iuml", L"Ï", L"U+00CF (207)"},
        {L"&ETH", L"Ð", L"U+00D0 (208)"},
        {L"&Ntilde", L"Ñ", L"U+00D1 (209)"},
        {L"&Ograve", L"Ò", L"U+00D2 (210)"},
        {L"&Oacute", L"Ó", L"U+00D3 (211)"},
        {L"&Ocirc", L"Ô", L"U+00D4 (212)"},
        {L"&Otilde", L"Õ", L"U+00D5 (213)"},
        {L"&Ouml", L"Ö", L"U+00D6 (214)"},
        {L"&times", L"×", L"U+00D7 (215)"},
        {L"&Oslash", L"Ø", L"U+00D8 (216)"},
        {L"&Ugrave", L"Ù", L"U+00D9 (217)"},
        {L"&Uacute", L"Ú", L"U+00DA (218)"},
        {L"&Ucirc", L"Û", L"U+00DB (219)"},
        {L"&Uuml", L"Ü", L"U+00DC (220)"},
        {L"&Yacute", L"Ý", L"U+00DD (221)"},
        {L"&THORN", L"Þ", L"U+00DE (222)"},
        {L"&szlig", L"ß", L"U+00DF (223)"},
        {L"&agrave", L"à", L"U+00E0 (224)"},
        {L"&aacute", L"á", L"U+00E1 (225)"},
        {L"&acirc", L"â", L"U+00E2 (226)"},
        {L"&atilde", L"ã", L"U+00E3 (227)"},
        {L"&auml", L"ä", L"U+00E4 (228)"},
        {L"&aring", L"å", L"U+00E5 (229)"},
        {L"&aelig", L"æ", L"U+00E6 (230)"},
        {L"&ccedil", L"ç", L"U+00E7 (231)"},
        {L"&egrave", L"è", L"U+00E8 (232)"},
        {L"&eacute", L"é", L"U+00E9 (233)"},
        {L"&ecirc", L"ê", L"U+00EA (234)"},
        {L"&euml", L"ë", L"U+00EB (235)"},
        {L"&igrave", L"ì", L"U+00EC (236)"},
        {L"&iacute", L"í", L"U+00ED (237)"},
        {L"&icirc", L"î", L"U+00EE (238)"},
        {L"&iuml", L"ï", L"U+00EF (239)"},
        {L"&eth", L"ð", L"U+00F0 (240)"},
        {L"&ntilde", L"ñ", L"U+00F1 (241)"},
        {L"&ograve", L"ò", L"U+00F2 (242)"},
        {L"&oacute", L"ó", L"U+00F3 (243)"},
        {L"&ocirc", L"ô", L"U+00F4 (244)"},
        {L"&otilde", L"õ", L"U+00F5 (245)"},
        {L"&ouml", L"ö", L"U+00F6 (246)"},
        {L"&oslash", L"ø", L"U+00F8 (248)"},
        {L"&ugrave", L"ù", L"U+00F9 (249)"},
        {L"&uacute", L"ú", L"U+00FA (250)"},
        {L"&ucirc", L"û", L"U+00FB (251)"},
        {L"&uuml", L"ü", L"U+00FC (252)"},
        {L"&yacute", L"ý", L"U+00FD (253)"},
        {L"&thorn", L"þ", L"U+00FE (254)"},
        {L"&yuml", L"ÿ", L"U+00FF (255)"},

        {L"&Scaron", L"Š", L"U+0160 (352)"},
        {L"&scaron", L"š", L"U+0161 (353)"}
    };

    int entity_count = (int)sizeof(html_entity_array) / sizeof(const wchar_t*) / 3;
    //LogKeyInfo("Latin letter count: %d", entity_count);
    for (int idx = 0; idx < entity_count; idx++)
    {
        const auto entity_info = html_entity_array[idx];
        html_code_trans_tree_.AddString(entity_info[0], entity_info[1]);
    }
}

void HtmlUtility::InitGreekLetters()
{
    const wchar_t *html_entity_array[][3] =
    {
        {L"&Alpha", L"Α", L"U+0391 (913)"},
        {L"&Beta", L"Β", L"U+0392 (914)"},
        {L"&Gamma", L"Γ", L"U+0393 (915)"},
        {L"&Delta", L"Δ", L"U+0394 (916)"},
        {L"&Epsilon", L"Ε", L"U+0395 (917)"},
        {L"&Zeta", L"Ζ", L"U+0396 (918)"},
        {L"&Eta", L"Η", L"U+0397 (919)"},
        {L"&Theta", L"Θ", L"U+0398 (920)"},
        {L"&Iota", L"Ι", L"U+0399 (921)"},
        {L"&Kappa", L"Κ", L"U+039A (922)"},
        {L"&Lambda", L"Λ", L"U+039B (923)"},
        {L"&Mu", L"Μ", L"U+039C (924)"},
        {L"&Nu", L"Ν", L"U+039D (925)"},
        {L"&Xi", L"Ξ", L"U+039E (926)"},
        {L"&Omicron", L"Ο", L"U+039F (927)"},
        {L"&Pi", L"Π", L"U+03A0 (928)"},
        {L"&Rho", L"Ρ", L"U+03A1 (929)"},
        {L"&Sigma", L"Σ", L"U+03A3 (931)"},
        {L"&Tau", L"Τ", L"U+03A4 (932)"},
        {L"&Upsilon", L"Υ", L"U+03A5 (933)"},
        {L"&Phi", L"Φ", L"U+03A6 (934)"},
        {L"&Chi", L"Χ", L"U+03A7 (935)"},
        {L"&Psi", L"Ψ", L"U+03A8 (936)"},
        {L"&Omega", L"Ω", L"U+03A9 (937)"},
        {L"&alpha", L"α", L"U+03B1 (945)"},
        {L"&beta", L"β", L"U+03B2 (946)"},
        {L"&gamma", L"γ", L"U+03B3 (947)"},
        {L"&delta", L"δ", L"U+03B4 (948)"},
        {L"&epsilon", L"ε", L"U+03B5 (949)"},
        {L"&epsiv", L"ε", L"U+03B5 (949)"},
        {L"&varepsilon", L"ε", L"U+03B5 (949)"},
        {L"&zeta", L"ζ", L"U+03B6 (950)"},
        {L"&eta", L"η", L"U+03B7 (951)"},
        {L"&theta", L"θ", L"U+03B8 (952)"},
        {L"&iota", L"ι", L"U+03B9 (953)"},
        {L"&kappa", L"κ", L"U+03BA (954)"},
        {L"&lambda", L"λ", L"U+03BB (955)"},
        {L"&mu", L"μ", L"U+03BC (956)"},
        {L"&nu", L"ν", L"U+03BD (957)"},
        {L"&xi", L"ξ", L"U+03BE (958)"},
        {L"&omicron", L"ο", L"U+03BF (959)"},
        {L"&pi", L"π", L"U+03C0 (960)"},
        {L"&rho", L"ρ", L"U+03C1 (961)"},
        {L"&sigmaf", L"ς", L"U+03C2 (962)"},
        {L"&sigmav", L"ς", L"U+03C2 (962)"},
        {L"&varsigma", L"ς", L"U+03C2 (962)"},
        {L"&sigma", L"σ", L"U+03C3 (963)"},
        {L"&tau", L"τ", L"U+03C4 (964)"},
        {L"&upsilon", L"υ", L"U+03C5 (965)"},
        {L"&upsi", L"υ", L"U+03C5 (965)"},
        {L"&phi", L"φ", L"U+03C6 (966)"},
        {L"&chi", L"χ", L"U+03C7 (967)"},
        {L"&psi", L"ψ", L"U+03C8 (968)"},
        {L"&omega", L"ω", L"U+03C9 (969)"},
        {L"&thetasym", L"ϑ", L"U+03D1 (977)"},
        {L"&thetav", L"ϑ", L"U+03D1 (977)"},
        {L"&vartheta", L"ϑ", L"U+03D1 (977)"},
        {L"&upsih", L"ϒ", L"U+03D2 (978)"},
        {L"&Upsi", L"ϒ", L"U+03D2 (978)"},
        {L"&straightphi", L"ϕ", L"U+03D5 (981)"},
        {L"&piv", L"ϖ", L"U+03D6 (982)"},
        {L"&varpi", L"ϖ", L"U+03D6 (982)"},
    };

    int entity_count = (int)sizeof(html_entity_array) / sizeof(const wchar_t*) / 3;
    //LogKeyInfo("Greek letter count: %d", entity_count);
    for (int idx = 0; idx < entity_count; idx++)
    {
        const auto entity_info = html_entity_array[idx];
        html_code_trans_tree_.AddString(entity_info[0], entity_info[1]);
    }
}

} //end of namespace
