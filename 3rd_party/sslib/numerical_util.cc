#include "numerical_util.h"
#include "macro.h"

namespace sslib
{

bool NumericalUtil::BeValidInteger(const char *str)
{
    if (str != nullptr && (*str == '+' || *str == '-')) {
        str++;
    }
    Macro_RetFalseIf(str == nullptr || *str == '\0');

    for (; *str != '\0'; str++)
    {
        char ch = *str;
        if (!(ch >= '0' && ch <= '9')) {
            return false;
        }
    }
    return true;
}

bool NumericalUtil::BeValidInteger(const std::string &str)
{
    return BeValidInteger(str.c_str());
}

} //end of namespace
