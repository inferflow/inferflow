#pragma once

#include "prime_types.h"
#include <string>

namespace sslib
{

struct Number
{
public:
    static const  uint8_t MaxUInt8  = 0xFF;
    static const   int8_t MaxInt8   = 0x7F;
    static const   int8_t MinInt8   = -128;
    static const uint16_t MaxUInt16 = 0xFFFF;
    static const  int16_t MaxInt16  = 0x7FFF;
    static const  int16_t MinInt16  = -32768;
    static const uint32_t MaxUInt32 = 0xFFFFFFFF;
    static const  int32_t MaxInt32  = 0x7FFFFFFF;
    static const  int32_t MinInt32  = (int32_t)0x80000000;
    static const uint64_t MaxUInt64 = 0xFFFFFFFFFFFFFFFF;
    static const  int64_t MaxInt64  = 0x7FFFFFFFFFFFFFFF;
    static const  int64_t MinInt64  = (int64_t)0x8000000000000000;

    static std::string ToString(uint32_t n);
    static std::string ToString(int32_t n);
    static std::string ToString(uint64_t n);
    static std::string ToString(int64_t n);

    static std::wstring ToWString(uint32_t n);
    static std::wstring ToWString(int32_t n);
    static std::wstring ToWString(uint64_t n);
    static std::wstring ToWString(int64_t n);

    static bool IsAlmostZero(double d)
    {
        return d > -0.0000001 && d < 0.0000001;
    }

    static bool BeAlmostEqual(double d1, double d2)
    {
        return IsAlmostZero(d1 - d2);
    }
};

} //end of namespace
