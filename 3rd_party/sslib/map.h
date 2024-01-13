#pragma once

#include <map>
#include "prime_types.h"
#include "string.h"

namespace sslib
{

//typedef std::map<std::wstring, UInt32, WStrLessNoCase> MapWstr2UInt32;

template <class ValueType>
class WStrMap : public std::map<std::wstring, ValueType, WStrLessNoCase>
{
};

} //end of namespace
