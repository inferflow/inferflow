#include "math_decimal.h"
#include <algorithm>
#include <cmath>
//#ifdef _WIN32
//#   define USE_MPIR
//#endif //_WIN32
#ifdef USE_MPIR
#pragma warning(push)
#pragma warning(disable:4127)
#pragma warning(disable:4800)
#pragma warning(disable:4512)
#pragma warning(disable:4244)
#include <mpirxx.h>
#pragma warning(pop)
#endif //USE_MPIR

using namespace std;

namespace sslib
{

const uint32_t MathDecimal::MAX_SIG_DIGITS;

MathDecimal::MathDecimal(int64_t n)
{
    Set(n, 0);
    flag_ = 0;
}

MathDecimal::MathDecimal(int64_t sig, int16_t exp)
{
    Set(sig, exp);
    flag_ = 0;
}

MathDecimal::MathDecimal(const wstring &str)
{
    Read(str);
}

MathDecimal::MathDecimal(const wchar_t *str, uint32_t len)
{
    Read(str, len);
}

MathDecimal::MathDecimal(const MathDecimal &rhs)
{
    this->sig_ = rhs.sig_;
    this->exp_ = rhs.exp_;
    this->flag_ = rhs.flag_;
}

const MathDecimal& MathDecimal::operator = (int64_t n)
{
    Set(n, 0);
    flag_ = 0;
    return *this;
}

const MathDecimal& MathDecimal::operator = (const MathDecimal &rhs)
{
    if (this != &rhs)
    {
        sig_ = rhs.sig_;
        exp_ = rhs.exp_;
        flag_ = rhs.flag_;
    }
    return *this;
}

void MathDecimal::Set(int64_t sig, int16_t exp)
{
    flag_ = 0;
    sig_ = sig;
    exp_ = exp;
    Normalize();
}

void MathDecimal::SetDouble(long double fnum)
{
    exp_ = 0;
    bool is_negative = fnum < 0;
    if (is_negative) {
        fnum = -fnum;
    }

    int prec_digits = std::min((int)MAX_SIG_DIGITS, 15);
    while (fnum > MathConst::V10E[MathConst::Max10Pow])
    {
        exp_ += (int16_t)MathConst::Max10Pow;
        fnum /= MathConst::V10E[MathConst::Max10Pow];
    }

    while (fnum > 0 && fnum < 1)
    {
        exp_ -= (int16_t)prec_digits;
        fnum *= MathConst::V10E[prec_digits];
    }

    while (fnum > 0 && fnum < MathConst::V10E[prec_digits])
    {
        exp_--;
        fnum *= 10;
    }

    sig_ = (uint64_t)(fnum + 0.5);
    if (is_negative) {
        sig_ = -sig_;
    }

    Normalize();

    bool bProx = (uint64_t)(fnum + 0.00001) != (uint64_t)(fnum + 0.99999);
    flag_ = bProx ? (uint16_t)MAX_SIG_DIGITS + 1 : (uint16_t)MathUtils::Digits(sig_);
}

void MathDecimal::Set(const MathDecimal &rhs)
{
    flag_ = rhs.flag_;
    sig_ = rhs.sig_; exp_ = rhs.exp_;
    Normalize();
}

void MathDecimal::SetInner(int64_t sig, int16_t exp, bool be_normalize)
{
    flag_ = 0;
    sig_ = sig;
    exp_ = exp;
    if (be_normalize) {
        Normalize();
    }
}

const MathDecimal& MathDecimal::operator += (const MathDecimal &rhs)
{
    *this = *this + rhs;
    return *this;
}

const MathDecimal& MathDecimal::operator -= (const MathDecimal &rhs)
{
    *this = *this - rhs;
    return *this;
}

const MathDecimal& MathDecimal::operator *= (const MathDecimal &rhs)
{
    *this = *this * rhs;
    return *this;
}

const MathDecimal& MathDecimal::operator /= (const MathDecimal &rhs)
{
    *this = *this / rhs;
    return *this;
}

const MathDecimal& MathDecimal::operator %= (const MathDecimal &rhs)
{
    *this = *this % rhs;
    return *this;
}

MathDecimal MathDecimal::operator + (const MathDecimal &rhs) const
{
    Macro_RetIf(*this, !IsValid());
    Macro_RetIf(rhs, !rhs.IsValid());
    MathDecimal val1 = exp_ < rhs.exp_ ? *this : rhs;
    MathDecimal val2 = exp_ < rhs.exp_ ? rhs : *this;
    uint32_t d2 = MathUtils::Digits(val2.sig_);
    uint32_t exp_delta = val2.exp_ - val1.exp_;
    if (d2 > MAX_SIG_DIGITS)
    {
        val1.SetErrorCode(Error_Overflow);
        return val1;
    }

    uint32_t move_up_digits = d2 + exp_delta <= MAX_SIG_DIGITS
        ? exp_delta : (MAX_SIG_DIGITS - d2);
    uint32_t move_down_digits = d2 + exp_delta <= MAX_SIG_DIGITS
        ? 0 : min(MAX_SIG_DIGITS, d2 + exp_delta - MAX_SIG_DIGITS);

    val2.sig_ *= (int64_t)MathConst::V10E[move_up_digits];
    val1.sig_ /= (int64_t)MathConst::V10E[move_down_digits];

    val1.sig_ += val2.sig_;
    int32_t adjust_digits = (int32_t)MathUtils::LastZeroNum(val1.sig_);
    if (adjust_digits == 0 && MathUtils::Digits(val1.sig_) > MAX_SIG_DIGITS) {
        adjust_digits = 1;
    }

    val1.sig_ /= (int64_t)MathConst::V10E[adjust_digits];
    val1.exp_ = (int16_t)(val2.exp_ - move_up_digits + adjust_digits);

    val1.Normalize();
    if (flag_ > MAX_SIG_DIGITS || rhs.flag_ > MAX_SIG_DIGITS) {
        val1.flag_ = max(flag_, rhs.flag_);
    }
    else {
        val1.flag_ = (uint16_t)MathUtils::Digits(val1.sig_);
    }
    return val1;
}

MathDecimal MathDecimal::operator - (const MathDecimal &rhs) const
{
    Macro_RetIf(*this, !IsValid() || !rhs.IsValid());
    MathDecimal val(rhs);
    val.sig_ = -val.sig_;
    return (*this) + val;
}

MathDecimal MathDecimal::operator * (const MathDecimal &rhs) const
{
    Macro_RetIf(*this, !IsValid());
    Macro_RetIf(rhs, !rhs.IsValid());
    Macro_RetIf(MathDecimal(0), IsZero() || rhs.IsZero());

    bool is_negative = (sig_ > 0 && rhs.sig_ < 0) || (sig_ < 0 && rhs.sig_ > 0);
    uint64_t sig1 = sig_ >= 0 ? sig_ : -sig_;
    uint64_t sig2 = rhs.sig_ >= 0 ? rhs.sig_ : -rhs.sig_;
#ifndef USE_MPIR
    UIntD36 product;
    MathUtils::Product(product, sig1, sig2);
    int32_t new_exp = (int32_t)exp_ + (int32_t)rhs.exp_;
    while (product.high != 0)
    {
        product.low += 5;
        product.ShiftDigitsRight(1);
        new_exp++;
    }
#else //ifndef USE_MPIR
#   ifdef _WIN64
    mpz_class product(sig1);
    product *= sig2;
#   else
    mpz_class product(sig1 / 0xFFFFFFFF);
    product *= 0xFFFFFFFF;
    product += (sig1 % 0xFFFFFFFF);
    mpz_class p2(sig2 / 0xFFFFFFFF);
    p2 *= 0xFFFFFFFF;
    p2 += (sig2 % 0xFFFFFFFF);
    product *= p2;
#   endif

    int32_t new_exp = (int32_t)exp_ + (int32_t)rhs.exp_;
    mpz_class max_sig(1000000000);
#   ifdef _WIN64
    max_sig *= 1000000000;
#   endif
    while (product >= max_sig)
    {
        new_exp++;
        product += 5;
        product /= 10;
    }
#endif //ifndef USE_MPIR

    MathDecimal val;
    if (new_exp > Max_Exp || new_exp < Min_Exp) //overflow
    {
        val.SetErrorCode(Error_Overflow);
        return val;
    }

#ifndef USE_MPIR
    val.sig_ = product.low;
#else //ifndef USE_MPIR
    val.sig_ = product.get_si();
#endif //ifndef USE_MPIR

    if (is_negative) {
        val.sig_ = -val.sig_;
    }
    val.exp_ = (int16_t)new_exp;
    val.Normalize();
    if (flag_ > MAX_SIG_DIGITS || rhs.flag_ > MAX_SIG_DIGITS) {
        val.flag_ = max(flag_, rhs.flag_);
    }
    else {
        val.flag_ = (uint16_t)MathUtils::Digits(val.sig_);
    }
    return val;
}

MathDecimal MathDecimal::operator / (const MathDecimal &rhs) const
{
    Macro_RetIf(*this, !IsValid());
    Macro_RetIf(rhs, !rhs.IsValid());
    bool is_neg = (sig_ > 0 && rhs.sig_ < 0) || (sig_ < 0 && rhs.sig_ > 0);
    uint64_t dividend = sig_ >= 0 ? sig_ : -sig_;
    uint64_t divisor = rhs.sig_ >= 0 ? rhs.sig_ : -rhs.sig_;
    //MathDecimal val(dividend / divisor, exp_ - rhs.exp_);
    MathDecimal val;
    val.SetInner(dividend / divisor, exp_ - rhs.exp_, false);
    uint32_t digits = val.sig_ > 0 ? MathUtils::Digits(val.sig_) : 0;

    dividend %= divisor;
    while (dividend > 0 && digits < MAX_SIG_DIGITS)
    {
        dividend *= 10; //this value will not exceed 2^64-1, because dividend is uint64_t type and with at most MAX_SIG_DIGITS digits
        //if(val.exp_ > 0)
        //{
        //    val.sig_ = val.sig_ * MathConst::V10E[val.exp_+1] + dividend / divisor;
        //    val.exp_ = -1;
        //}
        //else
        {
            val.exp_--;
            val.sig_ = val.sig_ * 10 + dividend / divisor;
        }

        if (val.sig_ > 0) {
            digits++;
        }
        dividend %= divisor;
    }

    if (is_neg) {
        val.sig_ = -val.sig_;
    }
    val.Normalize();
    if (flag_ > MAX_SIG_DIGITS || rhs.flag_ > MAX_SIG_DIGITS) {
        val.flag_ = max(flag_, rhs.flag_);
    }
    else {
        val.flag_ = (uint16_t)MathUtils::Digits(val.sig_);
    }
    return val;
}

MathDecimal MathDecimal::operator % (const MathDecimal &rhs) const
{
    MathDecimal val(*this);
    Macro_RetIf(val, !IsValid());
    Macro_RetIf(rhs, !rhs.IsValid());
    if (rhs.Sign() == 0)
    {
        val.SetErrorCode(Error_Overflow);
        return val;
    }

    MathDecimal q = *this / rhs; //quotient
    q.RoundTo(0, MathDecimal::RoundingWay::Down);
    /*int32_t exp_delta = min((int32_t)MAX_SIG_DIGITS, (int32_t)MathUtils::LastZeroNum(q.sig_) + q.exp_);
    if (exp_delta < 0)
    {
        if(-exp_delta > MAX_SIG_DIGITS)
        {
            val.sig_ = 0;
            val.exp_ = 0;
        }
        else
        {
            val.sig_ /= (int64_t)MathConst::V10E[-exp_delta];
            val.exp_ -= (int16_t)exp_delta;
        }
    }*/

    val -= q * rhs;
    if (flag_ > MAX_SIG_DIGITS || rhs.flag_ > MAX_SIG_DIGITS) {
        val.flag_ = max(flag_, rhs.flag_);
    }
    else {
        val.flag_ = (uint16_t)MathUtils::Digits(val.sig_);
    }
    return val;
}

MathDecimal& MathDecimal::Minus()
{
    sig_ = -sig_;
    return *this;
}

MathDecimal& MathDecimal::PowerDouble(double fexp)
{
    if (sig_ < 0)
    {
        SetErrorCode(Error_Std);
        return *this;
    }

    uint64_t sig = sig_ < 0 ? -sig_ : sig_;
    int32_t digits = (int32_t)MathUtils::Digits(sig_);
    int32_t exp = exp_ + digits;
    if (fexp > 300 || exp * fexp > 300 || exp * fexp < -300)
    {
        SetErrorCode(Error_Overflow);
        return *this;
    }

    long double ld_sig = (long double)sig / MathConst::V10E[digits];
    long double ld_res = powl(ld_sig, fexp);
    if (exp != 0) {
        ld_res *= powl(10, exp * fexp);
    }

    SetDouble(ld_res);
    flag_ = MAX_SIG_DIGITS + 1;
    return *this;
}

MathDecimal& MathDecimal::Power(uint32_t num, uint32_t den)
{
    return Power_Inner(num, den, true);
}

MathDecimal& MathDecimal::Power_Inner(uint32_t num, uint32_t den, bool is_high_precision)
{
    Macro_RetIf(*this, !IsValid());
    if (num != 0) {
        Macro_RetIf(*this, sig_ == 0 || (sig_ == 1 && exp_ == 0) || num == den);
    }

    if (den == 0)
    {
        SetErrorCode(Error_Std);
        return *this;
    }

    if (num / den > 0xFFFF)
    {
        SetErrorCode(Error_Overflow);
        return *this;
    }

    if (num % 2 == 1 && den % 2 == 0 && sig_ < 0)
    { // e.g., (-2) ^ (1/2)
        SetErrorCode(Error_Std);
        return *this;
    }

    if (is_high_precision && (den == 1 || num == 0))
    {
        MathDecimal bak = *this;
        bool is_succ = Power_HighPrecision((uint16_t)num);
        if (is_succ) {
            return *this;
        }

        *this = bak;
    }

    bool is_negative = sig_ < 0;
    vector<IntFactor> factors;
    uint64_t sig = sig_ < 0 ? -sig_ : sig_;
    bool is_perfect = (int64_t)exp_ * num % den == 0 && sig <= MathConst::CacheSize_LPF;
    if (is_perfect && is_high_precision)
    {
        PrimeFactorization(factors, sig);
        for (size_t iFactor = 0; is_perfect && iFactor < factors.size(); iFactor++)
        {
            IntFactor &fct = factors[iFactor];
            uint64_t new_exp = (uint64_t)num * fct.exp;
            if (new_exp % den != 0) {
                is_perfect = false;
                break;
            }

            new_exp /= den;
            if (new_exp > 0xFFFF) {
                is_perfect = false;
                break;
            }

            fct.exp = (uint16_t)new_exp;
        }
    }

    if (is_perfect && is_high_precision)
    {
#ifdef USE_MPIR
        mpz_class sig(1), mid;
        for (const IntFactor &fct : factors)
        {
            mpz_class p(fct.m_base);
            mpz_pow_ui(mid.get_mpz_t(), p.get_mpz_t(), fct.exp_);
            //for(uint16_t idx = 0; idx < fct.exp_; idx++) {
            //    sig *= fct.m_base;
            //}
            sig *= mid;
        }

        mpz_class max_sig(1000000000);
#       ifdef _WIN64
        max_sig *= 1000000000;
#       endif
        mpz_class big100, big120, ten(10);
        mpz_pow_ui(big120.get_mpz_t(), ten.get_mpz_t(), 120); //10 ^ 120
        mpz_pow_ui(big100.get_mpz_t(), ten.get_mpz_t(), 100); //10 ^ 100

        int64_t new_exp = (int64_t)exp_ * (int)num / (int)den;
        while (sig >= big120)
        {
            new_exp += 100;
            sig /= big100;
        }
        while (sig >= max_sig)
        {
            new_exp++;
            sig += 5;
            sig /= 10;
        }

        if (new_exp > Max_Exp || new_exp < Min_Exp) //overflow
        {
            SetErrorCode(Error_Overflow);
            return *this;
        }

        exp_ = (int16_t)new_exp;
        sig_ = sig.get_si();
        if (is_negative && num % 2 == 1) {
            sig_ = -sig_;
        }
        return *this;
#endif //ifdef USE_MPIR
    }

    /// not perfect
    long double fexp = num * 1.0 / den;
    int32_t digits = (int32_t)MathUtils::Digits(sig_);
    int32_t exp = exp_ + digits;
    if (fexp > 300 || exp * fexp > 300 || exp * fexp < -300) {
        SetErrorCode(Error_Overflow);
        return *this;
    }

    long double ld_sig = (long double)sig / MathConst::V10E[digits];
    long double ld_res = powl(ld_sig, fexp);
    if (exp != 0) {
        ld_res *= powl(10, exp * fexp);
    }
    if (is_negative && num % 2 == 1) {
        ld_res = -ld_res;
    }

    SetDouble(ld_res);
    return *this;
}

//TODO: improve the implementation
MathDecimal& MathDecimal::Power(uint16_t exp)
{
#ifdef USE_MPIR
    Power_HighPrecision(exp);
    return *this;
#else
    return Power_Inner(exp, 1, false);
#endif
}

//TODO: improve the implementation
bool MathDecimal::Power_HighPrecision(uint16_t exp)
{
    Macro_RetIf(true, !IsValid());
    if (exp == 0)
    {
        if (IsZero()) {
            SetErrorCode(Error_Std);
        }
        else {
            Set(1, 0);
        }
        return true;
    }
    Macro_RetIf(true, sig_ == 0 || (sig_ == 1 && exp_ == 0) || exp == 1);

#ifdef USE_MPIR
    mpz_class max_sig(1000000000);
#   ifdef _WIN64
    max_sig *= 1000000000;
#   endif
    mpz_class big100, big120, ten(10);
    mpz_pow_ui(big120.get_mpz_t(), ten.get_mpz_t(), 120); //10 ^ 120
    mpz_pow_ui(big100.get_mpz_t(), ten.get_mpz_t(), 100); //10 ^ 100

    mpz_class base(sig_ > 0 ? sig_ : -sig_), sig;
    mpz_pow_ui(sig.get_mpz_t(), base.get_mpz_t(), exp);

    int64_t new_exp = (int64_t)exp_ * (int)exp;
    while (sig >= big120)
    {
        new_exp += 100;
        sig /= big100;
    }
    while (sig >= max_sig)
    {
        new_exp++;
        sig += 5;
        sig /= 10;
    }
    int64_t new_sig = sig.get_si();
#else //USE_MPIR
    uint64_t base = sig_ > 0 ? sig_ : -sig_;
    UIntD36 product(base, 0);
    for (uint16_t exp_idx = 1; exp_idx < exp; exp_idx++)
    {
        MathUtils::Product(product, product.low, base);
        if (product.high != 0) {
            return false;
        }
    }

    int64_t new_sig = (int64_t)product.low;
    int64_t new_exp = (int64_t)exp_ * (int)exp;
    int64_t max_sig = (int64_t)MathConst::V10E[MathConst::Max10Pow];
    while (new_sig >= max_sig)
    {
        new_exp++;
        new_sig += 5;
        new_sig /= 10;
    }
#endif //USE_MPIR

    if (new_exp > Max_Exp || new_exp < Min_Exp) //overflow
    {
        SetErrorCode(Error_Overflow);
        return true;
    }

    bool is_negative = sig_ < 0;
    exp_ = (int16_t)new_exp;
    sig_ = new_sig;
    if (is_negative && exp % 2 == 1) {
        sig_ = -sig_;
    }

    Normalize();
    return true;
}

//static
MathDecimal MathDecimal::Power(const MathDecimal &base, uint16_t exp)
{
    MathDecimal d = base;
    d.Power(exp);
    return d;
}

MathDecimal& MathDecimal::Factorial()
{
    if (!IsInt() || Sign() < 0)
    {
        SetErrorCode(Error_Std);
        return *this;
    }

    if (IsZero())
    {
        Set(1);
        return *this;
    }

    if (Compare(10000) > 0)
    {
        SetErrorCode(Error_Overflow);
        return *this;
    }

    int32_t n = ToInt32();
    Set(1);
    MathDecimal p(1);
    for (int32_t m = 2; m <= n; m++)
    {
        p.Set(m);
        *this *= p;
    }

    return *this;
}

int MathDecimal::Compare(const MathDecimal &rhs) const
{
    if (sig_ == 0 || rhs.sig_ == 0 || exp_ == rhs.exp_) {
        return sig_ < rhs.sig_ ? -1 : (sig_ == rhs.sig_ ? 0 : 1);
    }

    MathDecimal val = *this - rhs;
    return val.Sign();
    //return exp_ < rhs.exp_ ? -1 : 1;
}

bool MathDecimal::operator == (const MathDecimal &rhs) const
{
    return Compare(rhs) == 0;
}

bool MathDecimal::operator != (const MathDecimal &rhs) const
{
    return Compare(rhs) != 0;
}

bool MathDecimal::operator < (const MathDecimal &rhs) const
{
    return Compare(rhs) < 0;
}

bool MathDecimal::operator <= (const MathDecimal &rhs) const
{
    return Compare(rhs) <= 0;
}

bool MathDecimal::operator > (const MathDecimal &rhs) const
{
    return Compare(rhs) > 0;
}

bool MathDecimal::operator >= (const MathDecimal &rhs) const
{
    return Compare(rhs) >= 0;
}

bool MathDecimal::IsInt() const
{
    //return sig_ == 0 || exp_ + 1 >= (int32_t)(MathUtils::Digits(sig_) - MathUtils::LastZeroNum(sig_));
    return sig_ == 0 || exp_ + (int32_t)MathUtils::LastZeroNum(sig_) >= 0;
}

void MathDecimal::Normalize()
{
    if (sig_ == 0) {
        exp_ = 1;
        return;
    }

    uint32_t zn = MathUtils::LastZeroNum(sig_);
    if (zn > 0)
    {
        sig_ /= (int64_t)MathConst::V10E[zn];
        exp_ += (int16_t)zn;
    }
}

//GCD: greatest common divisor
//static
MathDecimal MathDecimal::GCD(const MathDecimal &v1, const MathDecimal &v2)
{
    if (!v1.IsValid() || !v2.IsValid())
    {
        MathDecimal v(1);
        v.SetErrorCode(max(v1.flag_, v2.flag_));
        return v;
    }

    Macro_RetIf(1, !v1.IsInt() || !v2.IsInt());
    Macro_RetIf(v2, v1.sig_ == 0);
    Macro_RetIf(v1, v2.sig_ == 0);

    MathDecimal v(v1), r(v2);
    if (v.sig_ < 0) { v.sig_ = -v.sig_; }
    if (r.sig_ < 0) { r.sig_ = -r.sig_; }

    MathDecimal vTmp;
    while (r.sig_ != 0)
    {
        vTmp = r;
        r = v % r;
        v = vTmp;
    }

    return v;
}

//LCM: least common multiple
//static
MathDecimal MathDecimal::LCM(const MathDecimal &v1, const MathDecimal &v2)
{
    if (!v1.IsValid() || !v2.IsValid())
    {
        MathDecimal v(1);
        v.SetErrorCode(max(v1.flag_, v2.flag_));
        return v;
    }

    MathDecimal v = v1 / GCD(v1, v2) * v2;
    if (v.sig_ < 0) {
        v.sig_ = -v.sig_;
    }
    return v;
}

std::string MathDecimal::ToString() const
{
    DisplayFormat fmt;
    return ToString(fmt);
}

int32_t MathDecimal::ToInt32() const
{
    return (int32_t)ToInt64();
}

int64_t MathDecimal::ToInt64() const
{
    int64_t val = sig_;
    if (exp_ > 0) {
        val *= (int64_t)MathConst::V10E[min((uint16_t)exp_, MathConst::Max10Pow)];
    }
    else if (exp_ < 0) {
        val /= (int64_t)MathConst::V10E[min(MathConst::Max10Pow, (uint16_t)-exp_)];
    }

    return val;
}

double MathDecimal::ToDouble() const
{
    return (double)sig_ * pow(10, (double)exp_);
}

MathDecimal& MathDecimal::RoundTo(int16_t exp, RoundingWay way)
{
    if (!IsValid()) {
        return *this;
    }

    int16_t exp_delta = exp - exp_ - 1;
    if (exp_delta > MathConst::Max10Pow)
    {
        sig_ = 0;
    }
    else if (exp_delta >= 0)
    {
        int64_t sig = sig_ >= 0 ? sig_ : -sig_;
        int64_t r = sig % (int64_t)MathConst::V10E[exp_delta + 1];
        sig /= (int64_t)MathConst::V10E[exp_delta];

        switch (way)
        {
        case RoundingWay::Up:
            sig = r != 0 && sig_ > 0 ? sig / 10 + 1 : sig / 10;
            break;
        case RoundingWay::Down:
            sig = r != 0 && sig_ < 0 ? sig / 10 + 1 : sig / 10;
            break;
        case RoundingWay::TowardsZero:
            sig /= 10;
            break;
        case RoundingWay::AwayFromZero:
            sig = r != 0 ? sig / 10 + 1 : sig / 10;
            break;
        case RoundingWay::Std:
        default:
            sig = sig % 10 >= 5 ? sig / 10 + 1 : sig / 10;
            break;
        }

        sig_ = sig_ >= 0 ? sig : -sig;
        exp_ += (exp_delta + 1);
        flag_ = (uint16_t)MathUtils::Digits(sig_);
    }

    return *this;
}

string MathDecimal::ToString(const DisplayFormat &fmt) const
{
    char buf[64];
    if (fmt.is_raw) {
        sprintf(buf, "%jdE%d", sig_, exp_);
        return buf;
    }

    if (sig_ == 0) {
        return "0";
    }

    uint64_t sig = sig_ >= 0 ? (uint64_t)sig_ : (uint64_t)(-sig_);
    int32_t last_zero_num = (int32_t)MathUtils::LastZeroNum(sig);
    int32_t digits = (int32_t)MathUtils::Digits(sig_);
    int32_t exp = exp_ + digits - 1;
    int32_t d = max(exp + 1, digits - last_zero_num);
    if (IsValid() && d < flag_ && !fmt.trim_ending_zeroes) {
        d = flag_;
    }
    int32_t pending_zero_num = exp_ + last_zero_num;

    bool is_sci = exp > min((int)MAX_SIG_DIGITS, fmt.sci_thres_int)
        || pending_zero_num > fmt.sci_thres_pending_zero_num
        || exp < -1 * fmt.sci_thres_float;
    int32_t pre_zero_num = !is_sci && exp < 0 ? (-exp) : 0;
    int32_t point_pos = is_sci ? 1 : exp + 1;
    if (pre_zero_num > 0) {
        point_pos = -1;
    }

    uint32_t offset = 0;
    if (sig_ < 0) {
        buf[offset++] = '-';
    }

    for (int32_t digit_idx = 0; digit_idx < pre_zero_num; digit_idx++)
    {
        buf[offset++] = '0';
        if (digit_idx == 0) {
            buf[offset++] = '.';
        }
    }

    //round
    int pos = max(0, point_pos) + max(0, fmt.max_decimal_sig_digits);
    if (fmt.max_decimal_sig_digits >= 0 && pos < digits)
    {
        int tail_digit = (int)(sig / MathConst::V10E[digits - pos - 1] % 10);
        if (tail_digit >= 5) {
            sig = (1 + sig / MathConst::V10E[digits - pos]) * MathConst::V10E[digits - pos];
        }
    }

    int decimal_digit = pre_zero_num > 0 ? 0 : -1;
    for (int32_t digit_idx = 0; digit_idx < d; digit_idx++)
    {
        if (digit_idx >= point_pos && (digit_idx >= min((int)MAX_SIG_DIGITS, fmt.max_significant_digits)
            || decimal_digit >= fmt.max_decimal_sig_digits))
        {
            continue;
        }

        if (digit_idx == point_pos) {
            buf[offset++] = '.';
            decimal_digit = 0;
        }

        if (decimal_digit >= 0) {
            decimal_digit++;
        }

        uint64_t digit_value = digits > digit_idx ? sig / MathConst::V10E[digits - digit_idx - 1] : 0;
        if (digit_value >= 10)
        { //caused by the round operation above (99.9999 --> 100.000)
            if (pre_zero_num > 0)
            {
                if (offset - 1 > 0 && buf[offset - 1] == '.') {
                    buf[offset - 2] = '0' + 1 + (buf[offset - 2] - '0');
                }
                else {
                    buf[offset - 1] = '0' + 1 + (buf[offset - 1] - '0');
                }
                buf[offset++] = '0';
            }
            else
            {
                buf[offset++] = '1';
                if (fmt.add_separator && offset > 0 && point_pos > digit_idx && (point_pos - digit_idx) % 3 == 0) {
                    buf[offset++] = ',';
                }
                buf[offset++] = '0';
            }
        }
        else
        {
            if (fmt.add_separator && offset > 0 && point_pos > digit_idx && (point_pos - digit_idx) % 3 == 0) {
                buf[offset++] = ',';
            }
            buf[offset++] = '0' + (char)digit_value;
        }

        if (digits > digit_idx) {
            sig %= MathConst::V10E[digits - digit_idx - 1];
        }

        if (is_sci && digit_idx + 1 >= digits) {
            break;
        }
    }

    if (is_sci)
    {
        buf[offset++] = 'E';
        snprintf(buf + offset, 64 - offset, "%d", exp);
    }
    else
    {
        buf[offset] = '\0';
    }

    return buf;
}

struct IntFactorCursor : IntFactor
{
    uint16_t m_idx;

    IntFactorCursor()
    {
        m_idx = 0;
    }
};

bool MathDecimal::Factors(std::vector<uint64_t> &factors) const
{
    factors.clear();
    map<uint64_t, uint16_t> factor_map;
    bool ret = PrimeFactorization(factor_map);
    Macro_RetIf(false, !ret);

    if (factor_map.empty()) {
        factors.push_back(1);
        return ret;
    }

    vector<IntFactorCursor> cursor_list;
    IntFactorCursor the_cursor;
    for (auto iter = factor_map.begin(); iter != factor_map.end(); iter++)
    {
        the_cursor.base = iter->first;
        the_cursor.exp = iter->second;
        the_cursor.m_idx = 0;
        cursor_list.push_back(the_cursor);
    }

    uint32_t prime_num = (uint32_t)cursor_list.size();
    uint64_t factor_value = 1;
    while (prime_num > 0)
    {
        factor_value = 1;
        for (uint32_t prime_idx = 0; prime_idx < prime_num; prime_idx++)
        {
            for (uint16_t iDegree = 0; iDegree < cursor_list[prime_idx].m_idx; iDegree++) {
                factor_value *= cursor_list[prime_idx].base;
            }
        }
        factors.push_back(factor_value);

        //move next
        int focus_idx = (int)prime_num - 1;
        cursor_list[focus_idx].m_idx++;
        while (focus_idx > 0 && cursor_list[focus_idx].m_idx > cursor_list[focus_idx].exp)
        {
            cursor_list[focus_idx].m_idx = 0;
            cursor_list[focus_idx - 1].m_idx++;
            focus_idx--;
        }

        if (cursor_list[focus_idx].m_idx > cursor_list[focus_idx].exp) {
            break;
        }
    }

    return ret;
}

bool MathDecimal::PrimeFactorization(vector<IntFactor> &factors) const
{
    map<uint64_t, uint16_t> factor_map;
    bool ret = PrimeFactorization(factor_map);

    factors.clear();
    for (auto iter = factor_map.begin(); iter != factor_map.end(); iter++)
    {
        factors.push_back(IntFactor(iter->first, iter->second));
    }

    return ret;
}

bool MathDecimal::PrimeFactorization(map<uint64_t, uint16_t> &factors) const
{
    Macro_RetFalseIf(!IsInt());

    uint64_t n = sig_ < 0 ? -sig_ : sig_;
    bool ret = PrimeFactorization(factors, n);

    if (exp_ != 0)
    {
        for (int pindex = 0; pindex < 2; pindex++)
        {
            uint32_t p = pindex == 0 ? 2 : 5; //10 = 2 * 5
            auto iter = factors.find(p);
            if (iter != factors.end()) {
                iter->second += exp_;
            }
            else {
                factors[p] = exp_;
            }
        }
    }

    return ret;
}

//static
bool MathDecimal::PrimeFactorization(vector<IntFactor> &factors, uint64_t n)
{
    factors.clear();
    uint64_t m = MathConst::CacheSize_LPF;
    Macro_RetFalseIf(m * m < n);

    uint32_t p = 2;
    while (n > MathConst::CacheSize_LPF)
    {
        for (uint32_t prime_idx = 0; prime_idx < MathConst::CacheSize_Primes; prime_idx++)
        {
            p = MathConst::Primes[prime_idx];
            if (n % p == 0)
            {
                if (!factors.empty() && factors[factors.size() - 1].base == p) {
                    factors[factors.size() - 1].exp++;
                }
                else {
                    factors.push_back(IntFactor(p, 1));
                }
                n /= p;
                break;
            }
        }
    }

    return PrimeFactorization_SmallInt(factors, (uint32_t)n);
}

//static
bool MathDecimal::PrimeFactorization(map<uint64_t, uint16_t> &factors, uint64_t n)
{
    factors.clear();
    uint64_t m = MathConst::CacheSize_LPF;
    Macro_RetFalseIf(m * m < n);

    uint32_t p = 2;
    bool is_found = true;
    while (n > MathConst::CacheSize_LPF && is_found)
    {
        is_found = false;
        for (uint32_t prime_idx = 0; prime_idx < MathConst::CacheSize_Primes; prime_idx++)
        {
            p = MathConst::Primes[prime_idx];
            if (n % p == 0)
            {
                auto iter = factors.find(p);
                if (iter != factors.end()) {
                    iter->second++;
                }
                else {
                    factors[p] = 1;
                }
                n /= p;
                is_found = true;
                break;
            }
        }
    }

    if (n > MathConst::CacheSize_LPF) {
        factors[n] = 1;
        return true;
    }

    return PrimeFactorization_SmallInt(factors, (uint32_t)n);
}

//note: Factors are not cleared before factorization. The caller should do that if necessary
//static
bool MathDecimal::PrimeFactorization_SmallInt(vector<IntFactor> &factors, uint32_t n)
{
    Macro_RetFalseIf(n > MathConst::CacheSize_LPF);

    while (n > 1)
    {
        uint32_t p = MathConst::LeastPrimeFactor[n];
        if (!factors.empty() && factors[factors.size() - 1].base == p) {
            factors[factors.size() - 1].exp++;
        }
        else {
            factors.push_back(IntFactor(p, 1));
        }

        n /= p;
    }

    return true;
}

//note: Factors are not cleared before factorization. The caller should do that if necessary
//static
bool MathDecimal::PrimeFactorization_SmallInt(map<uint64_t, uint16_t> &factors, uint32_t n)
{
    Macro_RetFalseIf(n > MathConst::CacheSize_LPF);

    while (n > 1)
    {
        uint32_t p = MathConst::LeastPrimeFactor[n];
        auto iter = factors.find(p);
        if (iter != factors.end()) {
            iter->second++;
        }
        else {
            factors[p] = 1;
        }

        n /= p;
    }

    return true;
}

//static
MathDecimal MathDecimal::Parse(const std::wstring &str)
{
    MathDecimal dec(str);
    return dec;
}

//static
MathDecimal MathDecimal::Parse(const wchar_t *str, uint32_t len)
{
    MathDecimal dec(str, len);
    return dec;
}

bool MathDecimal::Read(const wstring &str)
{
    RawWStrStream strm(str.c_str(), (uint32_t)str.size());
    return Read(strm);
}

bool MathDecimal::Read(const wchar_t *str, uint32_t len)
{
    RawWStrStream strm(str, len == Number::MaxUInt32 ? (uint32_t)wcslen(str) : len);
    return Read(strm);
}

bool MathDecimal::Read(IN RawWStrStream &stream)
{
    Set(0);

    uint32_t offset_bak = stream.offset;
    uint32_t dot_num = 0, non_space_char_num = 0, char_num_after_exp_sign = 0;
    uint16_t digits_after_dot = 0, significant_digits_count = 0;
    bool is_neg_value = false, is_neg_exp = false;
    int32_t exp_val = 0;
    bool is_exp = false, has_digit = false;
    for (; stream.offset < stream.size; stream.offset++)
    {
        wchar_t wch = stream.data[stream.offset];
        //if(uci.m_iPrimType == UctPrime::Separator && non_space_char_num == 0) {
        //    continue;
        //}

        if (wch == L',' &&  stream.offset > offset_bak && stream.offset + 3 < stream.size)
        {
            bool is_num_sep = true;
            for (int next_idx = 1; next_idx <= 4; next_idx++)
            {
                wchar_t next_ch = stream.offset + next_idx < stream.size
                    ? stream.data[stream.offset + next_idx] : ' ';
                if ((next_idx <= 3 && !(next_ch >= L'0' && next_ch <= L'9'))
                    || (next_idx == 4 && (next_ch >= L'0' && next_ch <= L'9')))
                {
                    is_num_sep = false;
                }
            }

            if (is_num_sep) {
                continue;
            }
        }

        non_space_char_num++;
        char_num_after_exp_sign++;

        if (wch == L'-' || wch == L'+')
        {
            if (non_space_char_num > 1 && char_num_after_exp_sign > 1) {
                break;
            }

            if (wch == L'-')
            {
                if (is_exp) {
                    is_neg_exp = true;
                }
                else {
                    is_neg_value = true;
                }
            }
        }
        else if (wch == L'.' && !is_exp && dot_num <= 0)
        {
            //Macro_RetFalseIf(dot_num > 0 || is_exp);
            dot_num++;
        }
        else if ((wch == L'e' || wch == L'E') && stream.offset > offset_bak)
        {
            Macro_RetFalseIf(is_exp);
            is_exp = true;
            char_num_after_exp_sign = 0;
        }
        else if (wch >= L'0' && wch <= L'9')
        {
            uint32_t digit_idx = (uint32_t)(wch - L'0');
            has_digit = true;
            if (is_exp)
            {
                exp_val *= 10;
                exp_val += digit_idx;
            }
            else
            {
                if (this->IsZero() && digit_idx > 0) {
                    significant_digits_count = 0;
                }
                if (digit_idx > 0 || !this->IsZero() || dot_num > 0) {
                    significant_digits_count++;
                }

                if (dot_num > 0)
                {
                    digits_after_dot++;
                    (*this) += MathDecimal(digit_idx, -1 * digits_after_dot);
                }
                else
                {
                    (*this) *= 10;
                    (*this) += digit_idx;
                }
            }
        }
        else
        {
            break;
        }
    }

    if (is_neg_value) {
        (*this) *= -1;
    }

    if (exp_val != 0)
    {
        if (is_neg_exp) {
            exp_val = -exp_val;
        }
        (*this) *= MathDecimal(1, (int16_t)exp_val);
    }

    bool ret = has_digit && non_space_char_num > 0;
    if (ret)
    {
        //remove the last '.' in "#."
        if (stream.offset > 0 && stream.data[stream.offset - 1] == L'.') {
            stream.offset--;
        }
    }
    else
    {
        stream.offset = offset_bak;
    }

    //flag_ = significant_digits_count;
    return ret;
}

} //end of namespace
