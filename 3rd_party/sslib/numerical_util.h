#pragma once

#include <string>

namespace sslib
{

class NumericalUtil
{
public:
    NumericalUtil() {};
    virtual ~NumericalUtil() {};

    static bool BeValidInteger(const char *str);
    static bool BeValidInteger(const std::string &str);

    //this function is there because atoi() does not check the validity of the input string
    //static int atoi();
};

} //end of namespace
