#pragma once

#include "prime_types.h"

namespace sslib
{

struct UIntD36
{
    uint64_t high = 0, low = 0;

    UIntD36(uint64_t p_low = 0, uint64_t p_high = 0)
    {
        this->low = p_low;
        this->high = p_high;
    }

    void Clear()
    {
        this->high = 0;
        this->low = 0;
    }

    void ShiftDigitsRight(uint16_t k);
    void ShiftDigitsLeft(uint16_t k);
};

class MathConst
{
public:
    static const uint16_t Max10Pow = 18;
    static uint64_t V10E[Max10Pow+1];
    static int64_t V2E31Neg;

    static const uint32_t CacheSize_LPF = 0xFFFFF;
    static uint32_t CacheSize_Primes;
    static uint32_t *LeastPrimeFactor;
    static uint32_t *Primes;

    static uint32_t Prime2Order(uint32_t num);

    MathConst();
    virtual ~MathConst();
    static void Init();
    static void Free();
};

class MathUtils
{
public:
    MathUtils();

    static uint32_t Digits(int64_t num);
    static uint32_t LastZeroNum(int64_t num);

    /// alternative implementations
    static uint32_t Digits2(int64_t num);
    static uint32_t Digits3(int32_t num);
    static uint32_t Digits3(int64_t num);
    static uint32_t LastZeroNum2(int64_t num);

    static void Product(UIntD36 &res, uint64_t n1, uint64_t n2);

    static float Log2(float v);
    static float Log(float v);
    static float Exp(float v);
    static float Expf(float v);
};

} //end of namespace
