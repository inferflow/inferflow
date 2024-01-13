#pragma once

#include "math_decimal.h"
#include <string>

namespace sslib
{

#pragma pack(push, 1)

class MathFraction
{
public:
    explicit MathFraction(int64_t num = 0, int64_t den = 1, bool is_norm = false);
    explicit MathFraction(const MathDecimal &num, const MathDecimal &den, bool is_norm = false);
    explicit MathFraction(const MathDecimal &num, bool is_norm = false);
    MathFraction(const MathFraction &rhs);
    ~MathFraction(); //not virtual

    const MathDecimal numerator() const {
        return num_;
    }
    
    const MathDecimal denominator() const {
        return den_;
    }

    bool IsValid() const {
        return num_.IsValid() && den_.IsValid();
    }
    uint16_t GetSigDigits() const;

    const MathFraction& operator = (const MathFraction &rhs);

    MathFraction& Add(const MathFraction &rhs, bool is_normalize = true);

    MathFraction& operator += (const MathFraction &rhs);
    MathFraction& operator -= (const MathFraction &rhs);
    MathFraction& operator *= (const MathFraction &rhs);
    MathFraction& operator /= (const MathFraction &rhs);

    MathFraction operator + (const MathFraction &rhs) const;
    MathFraction operator - (const MathFraction &rhs) const;
    MathFraction operator * (const MathFraction &rhs) const;
    MathFraction operator / (const MathFraction &rhs) const;

    MathFraction& Reciprocal();
    MathFraction& Minus();

    const MathFraction& Power(const MathFraction &exp);
    const MathFraction& Power(int16_t exp);
    static MathFraction Power(const MathFraction &base, int16_t exp);

    const MathFraction& Log(const MathFraction &base);
    static MathFraction Log(const MathFraction &value, const MathFraction &base);

    void Clear();
    void Set(const MathFraction &rhs, bool is_norm = false);
    void Set(int64_t num = 0, int64_t den = 1, bool is_norm = false);
    void SetInt(int64_t val);
    int64_t ToInt64() const;

    //return values:
    //  -1: <
    //  0:  ==
    //  1:  >
    int Compare(const MathFraction &rhs) const;
    int Compare(int64_t rhs) const;
    int Sign() const { return den_.IsZero() ? num_.Sign() : num_.Sign() * den_.Sign(); };

    bool IsInt() const;
    bool IsZero() const { return num_.IsZero(); }

    MathFraction& Normalize(bool bForceToIntFraction = false); //or Reduce
    void NormalizeToDecimal();

public:
    struct ReadOptions
    {
        bool accept_latex = false;
        bool accept_flat_frac = true;
        bool accept_flat_mixed_frac = true;
        bool accept_ratio = true;
    };

    struct DisplayFormat
    {
        MathDecimal::DisplayFormat dec_fmt;

        DisplayFormat() {
        }
    };

public:
    std::string ToString() const;
    std::string ToString(const DisplayFormat &fmt) const;

    bool Read(IN RawWStrStream &stream, IN const UnicodeTable &uct);
    bool Read(IN RawWStrStream &stream, IN const UnicodeTable &uct, IN const ReadOptions &opt);

protected:
    MathDecimal num_, den_; //numerator and denominator;
};

#pragma pack(pop)

} //end of namespace
