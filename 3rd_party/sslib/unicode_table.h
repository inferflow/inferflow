#pragma once

#include "prime_types.h"

namespace sslib
{

struct UcRange
{
    enum
    {
        Unknown = 0,
        Ascii,
        Latin,
        Greek,
        Cyrillic,
        Arabic,
        GeneralPunct,
        Currency,
        CJKIdeographs,
        PrivateUse,
    };
};

//prime type
struct UctPrime
{
    enum
    {
        Unknown = 0,
        Letter,
        Number,
        Separator,
        Symbol,     //symbols and punctuation
        Mark,
        Other
    };
};

struct UctLetter
{
    enum
    {
        LatinBasic      = 0x0000,
        LatinExt        = 0x0001,
        Greek           = 0x0002,
        Cyrillic        = 0x0003,
        Arabic          = 0x0004,
        CJK             = 0x0005, //Kanji
        JP              = 0x0006, //Hiragana and Katakana
        KR              = 0x0007, //Hangul (Korean)
        MaskLang        = 0xFFFF
    };

    enum
    {
        LowerCase       = 0x010000,
        UpperCase       = 0x020000,
        TitleCase       = 0x030000,
        Modifier        = 0x040000,
        Other           = 0xFF0000
    };
};

struct UctNumber
{
    enum
    {
        Basic       = 0x0000,
        MaskLang    = 0xFFFF
    };

    enum
    {
        Digit       = 0x010000,
        Letter      = 0x020000,
        Others      = 0x030000,
        FullWidth   = 0x100000
    };
};

//Symbols and Punctuation
struct UctSymbol
{
    enum
    {
        Basic           = 0x00,
        CJK             = 0x03,
        Arabic          = 0x04,
        MaskLang        = 0xFF,

        OpenBracket     = 0x0100,
        CloseBracket    = 0x0200,
        Bracket         = 0x0300,
        Quotes          = 0x0400,
        DashLike        = 0x0800,   //dashes, hyphens, underscore, minus, etc
        LetterLike      = 0x1000,   //letter-like
    };

    enum
    {
        FullWidth   = 0x100000
    };
};

struct UctSeparator
{
    enum
    {
        Space = 0,
        Line,
        Paragraph
    };
};

#pragma pack(push, 1)
struct UnicodeCharInfo
{
    uint8_t range;
    uint8_t prime_type;
    uint32_t sub_type;
    wchar_t lower_case;
    wchar_t upper_case;
    wchar_t norm;
    uint32_t ext_data;

    UnicodeCharInfo()
    {
        range = UcRange::Unknown;
        SetType(UctPrime::Unknown, 0);
        SetVariantForms(L' ', L' ', L' ');
        ext_data = 0;
    }

    bool IsAsciiLetter() const
    {
        return range == UcRange::Ascii && prime_type == UctPrime::Letter;
    }

    bool IsAsciiDigit() const
    {
        return range == UcRange::Ascii && prime_type == UctPrime::Number;
    }

    bool IsAsciiLetterOrDigit() const
    {
        return range == UcRange::Ascii && (prime_type == UctPrime::Letter || prime_type == UctPrime::Number);
    }

    bool IsGreekLetter() const
    {
        return range == UcRange::Greek && prime_type == UctPrime::Letter;
    }

    bool IsSeparator() const
    {
        return prime_type == UctPrime::Separator;
    }

    void Set(uint8_t p_range, uint8_t p_prim_type, uint32_t p_sub_type,
        uint16_t p_lower_case, uint16_t p_upper_case, uint16_t p_norm)
    {
        range = p_range;
        SetType(p_prim_type, p_sub_type);
        SetVariantForms((wchar_t)p_lower_case, (wchar_t)p_upper_case, (wchar_t)p_norm);
    }

    void SetType(uint8_t p_prim_type, uint32_t p_sub_type)
    {
        prime_type = p_prim_type;
        sub_type = p_sub_type;
    }

    void SetVariantForms(wchar_t p_lower_case, wchar_t p_upper_case, wchar_t p_norm)
    {
        lower_case = p_lower_case;
        upper_case = p_upper_case;
        norm = p_norm;
    }

    void SetSymbol(uint32_t p_sub_type, wchar_t p_norm)
    {
        prime_type = UctPrime::Symbol; sub_type = p_sub_type;
        norm = p_norm;
    }
};
#pragma pack(pop)

class UnicodeTable
{
public:
    UnicodeTable();
    virtual ~UnicodeTable();

    bool Init();

    //
    inline const UnicodeCharInfo& Get(wchar_t wch) const {
        return char_array_[(uint16_t)wch];
    }
    UnicodeCharInfo& Get(wchar_t wch) {
        return char_array_[(uint16_t)wch];
    }

    //
    bool IsAsciiLetter(wchar_t wch) const {
        return IsAsciiLetter(char_array_[(uint16_t)wch]);
    }
    bool IsAsciiDigit(wchar_t wch) const {
        return IsAsciiDigit(char_array_[(uint16_t)wch]);
    }

    //
    static bool IsAsciiLetter(const UnicodeCharInfo &uci) {
        return uci.prime_type == UctPrime::Letter && (uci.sub_type & UctLetter::MaskLang) == UctLetter::LatinBasic;
    }
    static bool IsAsciiDigit(const UnicodeCharInfo &uci) {
        return uci.prime_type == UctPrime::Number && uci.sub_type == UctNumber::Basic;
    }
    static bool IsSpace(const UnicodeCharInfo &uci) {
        return uci.prime_type == UctPrime::Separator && uci.sub_type == UctSeparator::Space;
    }

protected:
    UnicodeCharInfo *char_array_;

protected:
    void SetupLetters();
    void SetupNumbers();
    void SetupSeparators();
    void SetupSymbols();
    void SetupOtherChars();
};

} //end of namespace
