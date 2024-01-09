#include "std_vocabulary.h"

INFER_FLOW_BEGIN

int CharToDigit(char ch)
{
    if (ch >= '0' && ch <= '9') {
        return (int)(ch - '0');
    }
    if (ch >= 'a' && ch <= 'f') {
        return (int)(ch - 'a') + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return (int)(ch - 'A') + 10;
    }

    return -1;
}

string StdVocabulary::IdToStr(int id, bool enable_decoding) const
{
    if (id < 0 || id >= (int)token_array.size()) {
        return "";
    }

    const string &token_str = token_array[id].str;
    int token_str_len = (int)token_str.size();

    //example: <0x0A>
    if (enable_decoding && token_str_len >= 5 && token_str_len <= 6)
    {
        //char ch3 = token_str[3], ch4 = token_str[4];
        bool has_right_context = token_str[0] == '<' && token_str[1] == '0'
            && token_str[2] == 'x' && token_str[token_str_len - 1] == '>';
        int n1 = CharToDigit(token_str[3]);
        int n2 = token_str_len == 6 ? CharToDigit(token_str[4]) : -1;

        string decoded_str;
        if (has_right_context && token_str_len == 5 && n1 >= 0)
        {
            decoded_str = (char)n1;
            return decoded_str;
        }
        else if (has_right_context && token_str_len == 6 && n1 >= 0 && n2 >= 0)
        {
            decoded_str = (char)((n1 << 4) | n2);
            return decoded_str;
        }
    }

    return token_str;
}

INFER_FLOW_END

