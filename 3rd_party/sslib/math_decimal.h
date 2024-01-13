#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include "macro.h"
#include "math_utils.h"
#include "raw_array.h"
#include "unicode_table.h"
#include "number.h"

namespace sslib
{

//#define USE_MPIR

#pragma pack(push, 1)

struct IntFactor
{
    uint64_t base;
    uint16_t exp;

    explicit IntFactor(uint64_t p_base = 1, uint16_t p_exp = 1)
    {
        base = p_base;
        exp = p_exp;
    }
};

//Motivation: In general, decimals cannot be represented precisely by IEEE-754 floating-point values
//related to the scientific notation
//In order for the Compare function to work, should always be in the normalized format
class MathDecimal
{
public:
    static const uint32_t MAX_SIG_DIGITS = 18;
    static const int16_t Max_Exp = 32500;
    static const int16_t Min_Exp = -32500;
    static const uint16_t Error_Start = 0x8000;
    static const uint16_t Error_Std = 0xFFFF;
    static const uint16_t Error_Overflow = 0xFFFE;

    enum class RoundingWay { Std = 0, Up, Down, TowardsZero, AwayFromZero };

public:
    MathDecimal(int64_t n = 0);
    MathDecimal(int64_t sig, int16_t exp);
    MathDecimal(const std::wstring &str);
    MathDecimal(const wchar_t *str, uint32_t len);
    MathDecimal(const MathDecimal &rhs);

    bool IsValid() const {
        return flag_ < Error_Start;
    }
    void SetErrorCode(uint16_t err) {
        flag_ = err;
    }
    uint16_t GetErrorCode() const {
        return flag_ >= Error_Start ? flag_ : 0;
    }
    uint16_t GetSigDigits() const {
        return flag_ < Error_Start ? flag_ : 0;
    }

    const MathDecimal& operator = (int64_t n);
    const MathDecimal& operator = (const MathDecimal &rhs);

    void Set(int64_t sig = 0, int16_t exp = 0);
    void SetDouble(long double num);
    void Set(const MathDecimal &rhs);

    const MathDecimal& operator += (const MathDecimal &rhs);
    const MathDecimal& operator -= (const MathDecimal &rhs);
    const MathDecimal& operator *= (const MathDecimal &rhs);
    const MathDecimal& operator /= (const MathDecimal &rhs);
    const MathDecimal& operator %= (const MathDecimal &rhs);

    MathDecimal operator + (const MathDecimal &rhs) const;
    MathDecimal operator - (const MathDecimal &rhs) const;
    MathDecimal operator * (const MathDecimal &rhs) const;
    MathDecimal operator / (const MathDecimal &rhs) const;
    MathDecimal operator % (const MathDecimal &rhs) const;

    MathDecimal& Minus();

    MathDecimal& PowerDouble(double exp);
    MathDecimal& Power(uint32_t num, uint32_t den);
    MathDecimal& Power(uint16_t exp);
    static MathDecimal Power(const MathDecimal &base, uint16_t exp);

    MathDecimal& Factorial();

    bool operator == (const MathDecimal &rhs) const;
    bool operator != (const MathDecimal &rhs) const;
    bool operator < (const MathDecimal &rhs) const;
    bool operator <= (const MathDecimal &rhs) const;
    bool operator > (const MathDecimal &rhs) const;
    bool operator >= (const MathDecimal &rhs) const;

    //return values:
    //  -1: <
    //  0:  ==
    //  1:  >
    int Compare(const MathDecimal &rhs) const;
    inline int Sign() const { return sig_ < 0 ? -1 : (sig_ == 0 ? 0 : 1); }

    bool IsInt() const;
    bool IsZero() const { return sig_ == 0; };
    bool IsOne() const { return sig_ == 1 && exp_ == 0; }

    void Normalize();

    //GCD: greatest common divisor
    static MathDecimal GCD(const MathDecimal &v1, const MathDecimal &v2);
    //LCM: least common multiple
    static MathDecimal LCM(const MathDecimal &v1, const MathDecimal &v2);

    inline int64_t GetSig() const { return sig_; }
    inline int16_t GetExp() const { return exp_; }

    int32_t ToInt32() const;
    int64_t ToInt64() const;
    double ToDouble() const;

    MathDecimal& RoundTo(int16_t exp, RoundingWay way = RoundingWay::Std);

    bool Factors(std::vector<uint64_t> &factors) const;
    bool PrimeFactorization(std::vector<IntFactor> &factors) const;
    bool PrimeFactorization(std::map<uint64_t, uint16_t> &factors) const;
    static bool PrimeFactorization(std::vector<IntFactor> &factors, uint64_t n);
    static bool PrimeFactorization(std::map<uint64_t, uint16_t> &factors, uint64_t n);

    static MathDecimal PI() {
        return MathDecimal(3141592653589793, -15);
    }
    static MathDecimal E() {
        return MathDecimal(2718281828459, -12);
    }

public:
    struct DisplayFormat
    {
        bool is_raw = false;
        int sci_thres_int = 20, sci_thres_float = 6, sci_thres_pending_zero_num = 6;
        int max_significant_digits = 20, max_decimal_sig_digits = 6;
        bool add_separator = false;
        bool trim_ending_zeroes = false;
    };

    std::string ToString() const;
    std::string ToString(const DisplayFormat &fmt) const;

    static MathDecimal Parse(const std::wstring &str);
    static MathDecimal Parse(const wchar_t *str, uint32_t len = UINT32_MAX);
    bool Read(const std::wstring &str);
    bool Read(const wchar_t *str, uint32_t len = UINT32_MAX);
    bool Read(IN RawWStrStream &stream);

protected:
    void SetInner(int64_t sig, int16_t exp, bool be_normalize);

    //note: Factors are not cleared before factorization. The caller should do that if necessary
    static bool PrimeFactorization_SmallInt(std::vector<IntFactor> &factors, uint32_t n);
    static bool PrimeFactorization_SmallInt(std::map<uint64_t, uint16_t> &factors, uint32_t n);

    MathDecimal& Power_Inner(uint32_t num, uint32_t den, bool is_high_precision);
    bool Power_HighPrecision(uint16_t exp);

private:
    int64_t sig_; //the significand (also coefficient or mantissa) part
    int16_t exp_; //the exponent part
    uint16_t flag_ = 0; //
};

#pragma pack(pop)

} //end of namespace
