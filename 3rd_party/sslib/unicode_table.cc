#include "unicode_table.h"
#include "log.h"

namespace sslib
{

UnicodeTable::UnicodeTable()
{
    char_array_ = new UnicodeCharInfo[65536];
}

UnicodeTable::~UnicodeTable()
{
    if (char_array_ != nullptr)
    {
        delete[] char_array_;
        char_array_ = nullptr;
    }
}

bool UnicodeTable::Init()
{
    bool ret = true;
    for (uint32_t idx = 0; idx < 65536; idx++)
    {
        char_array_[idx].range = UcRange::Unknown;
        char_array_[idx].SetType(UctPrime::Unknown, 0);
        char_array_[idx].SetVariantForms((wchar_t)idx, (wchar_t)idx, (wchar_t)idx);
    }

    //ranges
    //TODO

    //types
    SetupLetters();
    SetupNumbers();
    SetupSeparators();
    SetupSymbols();
    SetupOtherChars();

    return ret;
}

void UnicodeTable::SetupLetters()
{
    //Basic Latin
    for (uint16_t char_idx = 1; char_idx <= 26; char_idx++)
    {
        char_array_[0x0040 + char_idx].Set(UcRange::Ascii, UctPrime::Letter, UctLetter::LatinBasic | UctLetter::UpperCase, 0x0060 + char_idx, 0x0040 + char_idx, 0x0040 + char_idx);
        char_array_[0x0060 + char_idx].Set(UcRange::Ascii, UctPrime::Letter, UctLetter::LatinBasic | UctLetter::LowerCase, 0x0060 + char_idx, 0x0040 + char_idx, 0x0060 + char_idx);
    }

    //0x00C0 ~ 0x00DF
    uint32_t lower_ext_latin = UctLetter::LatinExt | UctLetter::LowerCase;
    uint32_t upper_ext_latin = UctLetter::LatinExt | UctLetter::UpperCase;
    char_array_[0x00C0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E0, 0x00C0, 0x0041);
    char_array_[0x00C1].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E1, 0x00C1, 0x0041);
    char_array_[0x00C2].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E2, 0x00C2, 0x0041);
    char_array_[0x00C3].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E3, 0x00C3, 0x0041);
    char_array_[0x00C4].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E4, 0x00C4, 0x0041); //check
    char_array_[0x00C5].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E5, 0x00C5, 0x0041);
    char_array_[0x00C6].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E6, 0x00C6, 0x00C6);
    char_array_[0x00C7].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E7, 0x00C7, 0x0043);
    char_array_[0x00C8].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E8, 0x00C8, 0x0045);
    char_array_[0x00C9].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00E9, 0x00C9, 0x0045);
    char_array_[0x00CA].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00EA, 0x00CA, 0x0045);
    char_array_[0x00CB].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00EB, 0x00CB, 0x0045);
    char_array_[0x00CC].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00EC, 0x00CC, 0x0049);
    char_array_[0x00CD].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00ED, 0x00CD, 0x0049);
    char_array_[0x00CE].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00EE, 0x00CE, 0x0049);
    char_array_[0x00CF].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00EF, 0x00CF, 0x0049);
    char_array_[0x00D0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F0, 0x00D0, 0x00D0);
    char_array_[0x00D1].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F1, 0x00D1, 0x004E);
    char_array_[0x00D2].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F2, 0x00D2, 0x004F);
    char_array_[0x00D3].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F3, 0x00D3, 0x004F);
    char_array_[0x00D4].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F4, 0x00D4, 0x004F);
    char_array_[0x00D5].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F5, 0x00D5, 0x004F);
    char_array_[0x00D6].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F6, 0x00D6, 0x004F); //check
    char_array_[0x00D8].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F8, 0x00D8, 0x00D8);
    char_array_[0x00D9].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00F9, 0x00D9, 0x0055);
    char_array_[0x00DA].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00FA, 0x00DA, 0x0055);
    char_array_[0x00DB].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00FB, 0x00DB, 0x0055);
    char_array_[0x00DC].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00FC, 0x00DC, 0x0055); //check
    char_array_[0x00DD].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00FD, 0x00DD, 0x0059); //check
    char_array_[0x00DE].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00FE, 0x00DE, 0x00DE);
    char_array_[0x00DF].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00DF, 0x00DF, 0x00DF);

    //0x00E0 ~ 0x00FF
    char_array_[0x00E0].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E0, 0x00C0, 0x0061);
    char_array_[0x00E1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E1, 0x00C1, 0x0061);
    char_array_[0x00E2].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E2, 0x00C2, 0x0061);
    char_array_[0x00E3].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E3, 0x00C3, 0x0061);
    char_array_[0x00E4].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E4, 0x00C4, 0x0061); //check
    char_array_[0x00E5].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E5, 0x00C5, 0x0061);
    char_array_[0x00E6].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E6, 0x00C6, 0x00E6);
    char_array_[0x00E7].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E7, 0x00C7, 0x0063);
    char_array_[0x00E8].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E8, 0x00C8, 0x0065);
    char_array_[0x00E9].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00E9, 0x00C9, 0x0065);
    char_array_[0x00EA].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00EA, 0x00CA, 0x0065);
    char_array_[0x00EB].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00EB, 0x00CB, 0x0065);
    char_array_[0x00EC].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00EC, 0x00CC, 0x0069);
    char_array_[0x00ED].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00ED, 0x00CD, 0x0069);
    char_array_[0x00EE].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00EE, 0x00CE, 0x0069);
    char_array_[0x00EF].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00EF, 0x00CF, 0x0069);
    char_array_[0x00F0].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F0, 0x00D0, 0x00F0);
    char_array_[0x00F1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F1, 0x00D1, 0x006E);
    char_array_[0x00F2].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F2, 0x00D2, 0x006F);
    char_array_[0x00F3].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F3, 0x00D3, 0x006F);
    char_array_[0x00F4].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F4, 0x00D4, 0x006F);
    char_array_[0x00F5].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F5, 0x00D5, 0x006F);
    char_array_[0x00F6].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F6, 0x00D6, 0x006F); //check
    char_array_[0x00F8].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F8, 0x00D8, 0x00F8);
    char_array_[0x00F9].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00F9, 0x00D9, 0x0075);
    char_array_[0x00FA].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00FA, 0x00DA, 0x0075);
    char_array_[0x00FB].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00FB, 0x00DB, 0x0075);
    char_array_[0x00FC].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00FC, 0x00DC, 0x0075); //check
    char_array_[0x00FD].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00FD, 0x00DD, 0x0079); //check
    char_array_[0x00FE].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00FE, 0x00DE, 0x00FE);
    char_array_[0x00FF].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x00DF, 0x0178, 0x00FF);

    //0x0100 ~ 0x017E
    for (uint16_t char_idx = 0x0100; char_idx < 0x0106; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0041);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0061);
    }
    for (uint16_t char_idx = 0x0106; char_idx < 0x010E; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0043);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0063);
    }
    for (uint16_t char_idx = 0x010E; char_idx < 0x0112; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0044);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0064);
    }
    for (uint16_t char_idx = 0x0112; char_idx < 0x011C; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0045);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0065);
    }
    for (uint16_t char_idx = 0x011C; char_idx < 0x0124; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0047);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0067);
    }
    for (uint16_t char_idx = 0x0124; char_idx < 0x0128; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0048);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0068);
    }
    for (uint16_t char_idx = 0x0128; char_idx < 0x0132; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0049);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0069);
    }
    char_array_[0x0132].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0133, 0x0132, 0x0132);
    char_array_[0x0133].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0133, 0x0132, 0x0133);
    char_array_[0x0134].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0135, 0x0134, 0x004A);
    char_array_[0x0135].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0135, 0x0134, 0x006A);
    char_array_[0x0136].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0137, 0x0136, 0x004B);
    char_array_[0x0137].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0137, 0x0136, 0x006B);
    char_array_[0x0138].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0138, 0x0138, 0x0138);
    for (uint16_t char_idx = 0x0139; char_idx < 0x0143; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x004C);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x006C);
    }
    for (uint16_t char_idx = 0x0143; char_idx < 0x014A; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x004E);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x006E);
    }
    char_array_[0x014A].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x014B, 0x014A, 0x014A);
    char_array_[0x014B].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x014B, 0x014A, 0x014B);
    for (uint16_t char_idx = 0x014C; char_idx < 0x0152; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x004F);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x006F);
    }
    char_array_[0x0152].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0153, 0x0152, 0x0152);
    char_array_[0x0153].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0153, 0x0152, 0x0153);
    for (uint16_t char_idx = 0x0154; char_idx < 0x015A; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0052);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0072);
    }
    for (uint16_t char_idx = 0x015A; char_idx < 0x0162; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0053);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0073);
    }
    for (uint16_t char_idx = 0x0162; char_idx < 0x0168; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0054);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0074);
    }
    for (uint16_t char_idx = 0x0168; char_idx < 0x0174; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0055);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0075);
    }
    for (uint16_t char_idx = 0x0168; char_idx < 0x0174; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x0055);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x0075);
    }
    char_array_[0x0174].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0175, 0x0174, 0x0057);
    char_array_[0x0175].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0175, 0x0174, 0x0077);
    char_array_[0x0176].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0177, 0x0176, 0x0059);
    char_array_[0x0177].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0177, 0x0176, 0x0079);
    char_array_[0x0178].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x00FF, 0x0178, 0x0178); //check
    for (uint16_t char_idx = 0x0179; char_idx < 0x017F; char_idx += 2) {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, 0x005A);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, 0x007A);
    }

    char_array_[0x017F].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x017F, 0x017F, L's');

    /// Latin Extended-B (todo: improve)
    for (uint16_t char_idx = 0x0180; char_idx < 0x01FF; char_idx++) {
        char_array_[char_idx].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx, char_idx, char_idx);
    }

    char_array_[0x0180].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0180, 0x0180, L'b');
    char_array_[0x0181].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0181, 0x0181, L'B');
    char_array_[0x0182].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0182, 0x0182, L'B');
    char_array_[0x0183].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0183, 0x0183, L'b');
    char_array_[0x0184].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0184, 0x0184, L'B');
    char_array_[0x0185].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0185, 0x0185, L'b');
    char_array_[0x0186].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0186, 0x0186, 0x0186);
    char_array_[0x0187].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0187, 0x0187, L'C');
    char_array_[0x0188].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0188, 0x0188, L'c');
    char_array_[0x0189].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0189, 0x0189, L'D');
    char_array_[0x018A].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x018A, 0x018A, L'D');
    char_array_[0x018B].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x018B, 0x018B, L'D');
    char_array_[0x018C].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x018C, 0x018C, L'd');

    for (uint16_t char_idx = 0x01CD; char_idx <= 0xDC; char_idx++)
    {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, char_idx);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, char_idx + 1);
    }

    char_array_[0x01CD].norm = L'A';
    char_array_[0x01CE].norm = L'a';
    char_array_[0x01CF].norm = L'I';
    char_array_[0x01D0].norm = L'i';
    char_array_[0x01D1].norm = L'O';
    char_array_[0x01D2].norm = L'o';
    char_array_[0x01D3].norm = L'U';
    char_array_[0x01D4].norm = L'u';

    for (uint16_t char_idx = 0x01DE; char_idx <= 0xEF; char_idx++)
    {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, char_idx);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, char_idx + 1);
    }

    for (uint16_t char_idx = 0x0200; char_idx < 0x024F; char_idx += 2)
    {
        char_array_[char_idx + 0].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, char_idx + 1, char_idx, char_idx);
        char_array_[char_idx + 1].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx + 1, char_idx, char_idx + 1);
    }

    for (uint16_t char_idx = 0x0234; char_idx <= 0x0239; char_idx++) {
        char_array_[char_idx].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, char_idx, char_idx, char_idx);
    }

    char_array_[0x023A].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x023A, 0x023A, L'A');
    char_array_[0x023B].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x023C, 0x023B, L'C');
    char_array_[0x023C].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x023C, 0x023B, L'c');
    char_array_[0x023D].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x023D, 0x023D, L'L');
    char_array_[0x023E].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x023E, 0x023E, L'T');
    char_array_[0x023F].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x023F, 0x023F, L's');

    char_array_[0x0240].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0240, 0x0240, L'z');
    char_array_[0x0241].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0242, 0x0241, 0x0241);
    char_array_[0x0242].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0x0242, 0x0241, 0x0242);
    char_array_[0x0243].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0243, 0x0243, 0x0243);
    char_array_[0x0244].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0244, 0x0244, 0x0244);
    char_array_[0x0245].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0x0245, 0x0245, 0x0245);

    /// Greek
    char_array_[0x0386].Set(UcRange::Greek, UctPrime::Letter, UctLetter::Greek, 0x0386, 0x0386, 0x0386);
    for (uint16_t char_idx = 0x0388; char_idx <= 0x0390; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::Greek, UctPrime::Letter, UctLetter::Greek, char_idx, char_idx, char_idx);
    }
    for (uint16_t char_idx = 0x0391; char_idx <= 0x03A9; char_idx++) {
        char_array_[char_idx].Set(UcRange::Greek, UctPrime::Letter, UctLetter::Greek | UctLetter::UpperCase, char_idx, char_idx, char_idx);
    }
    for (uint16_t char_idx = 0x03AA; char_idx <= 0x03B0; char_idx++) {
        char_array_[char_idx].Set(UcRange::Greek, UctPrime::Letter, UctLetter::Greek, char_idx, char_idx, char_idx);
    }
    for (uint16_t char_idx = 0x03B1; char_idx <= 0x03C9; char_idx++) {
        char_array_[char_idx].Set(UcRange::Greek, UctPrime::Letter, UctLetter::Greek | UctLetter::LowerCase, char_idx, char_idx, char_idx);
    }
    for (uint16_t char_idx = 0x03CA; char_idx <= 0x03F3; char_idx++) {
        char_array_[char_idx].Set(UcRange::Greek, UctPrime::Letter, UctLetter::Greek, char_idx, char_idx, char_idx);
    }

    //Fullwidth ASCII letters
    for (uint16_t char_idx = 1; char_idx <= 26; char_idx++)
    {
        char_array_[0xFF20 + char_idx].Set(UcRange::Latin, UctPrime::Letter, upper_ext_latin, 0xFF40 + char_idx, 0xFF20 + char_idx, 0x0040 + char_idx);
        char_array_[0xFF40 + char_idx].Set(UcRange::Latin, UctPrime::Letter, lower_ext_latin, 0xFF40 + char_idx, 0xFF20 + char_idx, 0x0060 + char_idx);
    }

    //Cyrillic
    for (UInt16 char_idx = 0x0400; char_idx <= 0x04FF; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::Cyrillic, UctPrime::Letter, UctLetter::Cyrillic, char_idx, char_idx, char_idx);
    }

    //Arabic letters
    for (UInt16 char_idx = 0x0620; char_idx <= 0x064A; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::Arabic, UctPrime::Letter, UctLetter::Arabic, char_idx, char_idx, char_idx);
    }
    for (UInt16 char_idx = 0x066E; char_idx <= 0x06D3; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::Arabic, UctPrime::Letter, UctLetter::Arabic, char_idx, char_idx, char_idx);
    }

    //Japanese Hiragana (3040-309F) and Katakana (30A0-30FF)
    for (uint16_t char_idx = 0x3040; char_idx <= 0x309F; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::Unknown, UctPrime::Letter, UctLetter::JP, char_idx, char_idx, char_idx);
    }
    for (uint16_t char_idx = 0x30A0; char_idx <= 0x30FF; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::Unknown, UctPrime::Letter, UctLetter::JP, char_idx, char_idx, char_idx);
    }

    //CJK Unified Ideographs (4E00-9FAF)
    for (uint16_t char_idx = 0x4E00; char_idx <= 0x9FAF; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::CJKIdeographs, UctPrime::Letter, UctLetter::CJK, char_idx, char_idx, char_idx);
    }

    //Hangul (Korean): AC00-D7A3, 1100-11FF, 3130-318F, FFA0-FFDF, ...
    for (uint16_t char_idx = 0xAC00; char_idx <= 0xD7A3; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::Unknown, UctPrime::Letter, UctLetter::KR, char_idx, char_idx, char_idx);
    }

    //CJK Unified Ideographs Extension A (3400-4DBF)
    for (uint16_t char_idx = 0x3400; char_idx <= 0x4DBF; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::CJKIdeographs, UctPrime::Letter, UctLetter::CJK, char_idx, char_idx, char_idx);
    }

    //CJK Unified Ideographs, Extension B (20000-215FF)
    //TODO
}

void UnicodeTable::SetupNumbers()
{
    for (uint16_t char_idx = 0; char_idx <= 9; char_idx++)
    {
        char_array_[0x0030 + char_idx].Set(UcRange::Ascii, UctPrime::Number,
            UctNumber::Basic | UctNumber::Digit,
            0x0030 + char_idx, 0x0030 + char_idx, 0x0030 + char_idx);
        char_array_[0xFF10 + char_idx].Set(UcRange::Ascii, UctPrime::Number,
            UctNumber::Basic | UctNumber::Digit | UctNumber::FullWidth,
            0xFF10 + char_idx, 0xFF10 + char_idx, 0x0030 + char_idx);
    }
}

void UnicodeTable::SetupSeparators()
{
    char_array_[0x0020].Set(UcRange::Ascii, UctPrime::Separator, UctSeparator::Space, 0x0020, 0x0020, 0x0020);
    char_array_[0x00A0].Set(UcRange::Latin, UctPrime::Separator, UctSeparator::Space, 0x00A0, 0x00A0, 0x0020);
    char_array_[L'\t'].Set(UcRange::Ascii, UctPrime::Separator, UctSeparator::Space, L'\t', L'\t', L'\t');
    char_array_[L'\r'].Set(UcRange::Ascii, UctPrime::Separator, UctSeparator::Line, L'\r', L'\r', L'\r');
    char_array_[L'\n'].Set(UcRange::Ascii, UctPrime::Separator, UctSeparator::Line, L'\n', L'\n', L'\n');

    for (uint16_t char_idx = 0x2000; char_idx <= 0x200B; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::GeneralPunct, UctPrime::Separator,
            UctSeparator::Space, char_idx, char_idx, 0x0020);
    }

    char_array_[0x3000].Set(UcRange::GeneralPunct, UctPrime::Separator,
        UctSeparator::Space, 0x3000, 0x3000, 0x0020);
}

void UnicodeTable::SetupSymbols()
{
    //ASCII
    for (uint16_t ascii_idx = 0x21; ascii_idx <= 0x7E; ascii_idx++)
    {
        UnicodeCharInfo &uci = char_array_[ascii_idx];
        if (uci.prime_type != UctPrime::Letter && uci.prime_type != UctPrime::Number)
        {
            uci.Set(UcRange::Ascii, UctPrime::Symbol, UctSymbol::Basic, ascii_idx, ascii_idx, ascii_idx);
        }
    }

    char_array_[L'\"'].sub_type |= UctSymbol::Quotes;
    char_array_[L'\''].sub_type |= UctSymbol::Quotes;
    char_array_[L'-'].sub_type |= UctSymbol::DashLike;
    char_array_[L'_'].sub_type |= UctSymbol::DashLike;
    char_array_[L'~'].sub_type |= UctSymbol::DashLike;
    char_array_[L'('].sub_type |= UctSymbol::OpenBracket;
    char_array_[L')'].sub_type |= UctSymbol::CloseBracket;
    char_array_[L'<'].sub_type |= UctSymbol::OpenBracket;
    char_array_[L'>'].sub_type |= UctSymbol::CloseBracket;
    char_array_[L'['].sub_type |= UctSymbol::OpenBracket;
    char_array_[L']'].sub_type |= UctSymbol::CloseBracket;
    char_array_[L'{'].sub_type |= UctSymbol::OpenBracket;
    char_array_[L'}'].sub_type |= UctSymbol::CloseBracket;

    //symbols in the ASCII extended character sets
    for (uint16_t char_idx = 0xA1; char_idx <= 0xBF; char_idx++)
    {
        UnicodeCharInfo &uci = char_array_[char_idx];
        uci.Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic, char_idx, char_idx, char_idx);
    }

    char_array_[0x00B7].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic,
        0x00B7, 0x00B7, 0x00B7); //middle dot
    char_array_[0x00D7].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic,
        0x00D7, 0x00D7, 0x00D7); //multiplication sign
    char_array_[0x00F7].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic,
        0x00F7, 0x00F7, 0x00F7); //division sign
    //TODO: add others (refer to http://ascii-table.com/ascii-extended-pc-list.php)

    //fullwidth
    for (uint16_t char_idx = 0x01; char_idx <= 0x5E; char_idx++)
    {
        UnicodeCharInfo &uci = char_array_[0xFF00 + char_idx];
        const UnicodeCharInfo &uci_ascii = char_array_[0x0020 + char_idx];
        if (uci.prime_type != UctPrime::Letter && uci.prime_type != UctPrime::Number)
        {
            uci.Set(UcRange::GeneralPunct, UctPrime::Symbol, uci_ascii.sub_type | UctSymbol::FullWidth,
                0xFF00 + char_idx, 0xFF00 + char_idx, 0x0020 + char_idx);
        }
    }

    for (uint16_t char_idx = 0xFFC0; char_idx <= 0xFFEE; char_idx++)
    {
        char_array_[char_idx].SetSymbol(UctSymbol::Basic, char_idx);
    }

    for (uint16_t char_idx = 0x2010; char_idx <= 0x2013; char_idx++) {
        char_array_[char_idx].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic | UctSymbol::DashLike, char_idx, char_idx, L'-');
    }
    for (uint16_t char_idx = 0x2014; char_idx <= 0x2015; char_idx++) {
        char_array_[char_idx].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic | UctSymbol::DashLike, char_idx, char_idx, L'-');
    }
    for (uint16_t char_idx = 0x2016; char_idx <= 0x2027; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic,
            char_idx, char_idx, char_idx);
    }
    char_array_[0x2018].SetSymbol(UctSymbol::Basic | UctSymbol::Quotes, L'\'');
    char_array_[0x2019].SetSymbol(UctSymbol::Basic | UctSymbol::Quotes, L'\'');
    char_array_[0x201C].SetSymbol(UctSymbol::Basic | UctSymbol::Quotes, L'\"');
    char_array_[0x201D].SetSymbol(UctSymbol::Basic | UctSymbol::Quotes, L'\"');
    char_array_[0x2022].SetSymbol(UctSymbol::Basic, 0x00B7); //middle dot as the norm
    for (uint16_t char_idx = 0x2030; char_idx <= 0x2049; char_idx++)
    {
        char_array_[char_idx].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic,
            char_idx, char_idx, char_idx);
    }
    char_array_[0x2043].SetSymbol(UctSymbol::Basic | UctSymbol::DashLike, L'-');
    char_array_[0x20AC].SetSymbol(UctSymbol::Basic, 0x20AC); // '€'

    char_array_[0x2103].SetSymbol(UctSymbol::Basic | UctSymbol::LetterLike, 0x2103); //degree Celsius
    char_array_[0x2109].SetSymbol(UctSymbol::Basic | UctSymbol::LetterLike, 0x2109); //degree Fahrenheit
    char_array_[0x212A].SetSymbol(UctSymbol::Basic | UctSymbol::LetterLike, 0x212A); //Kelvin sign

    //Number forms
    for (uint16_t char_idx = 0x2150; char_idx <= 0x218F; char_idx++)
    {
        UnicodeCharInfo &uci = char_array_[char_idx];
        uci.Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic, char_idx, char_idx, char_idx);
    }

    //arrows
    for (uint16_t char_idx = 0x2190; char_idx <= 0x21FF; char_idx++)
    {
        UnicodeCharInfo &uci = char_array_[char_idx];
        uci.Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic, char_idx, char_idx, char_idx);
    }

    //Mathematical operators
    for (uint16_t char_idx = 0x2200; char_idx <= 0x22FF; char_idx++)
    {
        UnicodeCharInfo &uci = char_array_[char_idx];
        uci.Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic, char_idx, char_idx, char_idx);
    }
    char_array_[0x2219].Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic, 0x2219, 0x2219, 0x00B7); //bullet operator

    for (UInt16 char_idx = 0x2400; char_idx <= 0x27FF; char_idx++)
    {
        UnicodeCharInfo &uci = char_array_[char_idx];
        uci.Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::Basic, char_idx, char_idx, char_idx);
    }

    //CJK symbols and punctuation
    for (uint16_t char_idx = 0x3001; char_idx <= 0x303F; char_idx++)
    {
        UnicodeCharInfo &uci = char_array_[char_idx];
        uci.Set(UcRange::GeneralPunct, UctPrime::Symbol, UctSymbol::CJK, char_idx, char_idx, char_idx);
    }
    //char_array_[0x3001].norm = L',';
    char_array_[0x3003].SetSymbol(UctSymbol::CJK | UctSymbol::Quotes, L'\"');
    char_array_[0x3008].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'<');
    char_array_[0x3009].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L'>');
    char_array_[0x300A].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, 0x300A);
    char_array_[0x300B].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, 0x300B);
    char_array_[0x300C].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'[');
    char_array_[0x300D].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L']');
    char_array_[0x300E].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'[');
    char_array_[0x300F].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L']');
    char_array_[0x3010].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'[');
    char_array_[0x3011].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L']');
    char_array_[0x3014].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'(');
    char_array_[0x3015].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L')');
    char_array_[0x3016].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'[');
    char_array_[0x3017].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L']');
    char_array_[0x3018].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'[');
    char_array_[0x3019].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L']');
    char_array_[0x301A].SetSymbol(UctSymbol::CJK | UctSymbol::OpenBracket, L'[');
    char_array_[0x301B].SetSymbol(UctSymbol::CJK | UctSymbol::CloseBracket, L']');
    char_array_[0x301D].SetSymbol(UctSymbol::CJK | UctSymbol::Quotes, L'\"');
    char_array_[0x301E].SetSymbol(UctSymbol::CJK | UctSymbol::Quotes, L'\"');

    //Arabic
    char_array_[0x060C].SetSymbol(UctSymbol::Arabic, L',');    //comma
    char_array_[0x061F].SetSymbol(UctSymbol::Arabic, L'?');    //question mark
    char_array_[0x066A].SetSymbol(UctSymbol::Arabic, L'%');    //percent sign
    char_array_[0x066B].SetSymbol(UctSymbol::Arabic, 0x066B);  //decimal Separator
    char_array_[0x066C].SetSymbol(UctSymbol::Arabic, 0x066C);  //thousands Separator
    char_array_[0x066D].SetSymbol(UctSymbol::Arabic, 0x066D);  //five pointed star
    char_array_[0x06D4].SetSymbol(UctSymbol::Arabic, 0x06D4);  //full stop
}

void UnicodeTable::SetupOtherChars()
{
}

} //end of namespace
