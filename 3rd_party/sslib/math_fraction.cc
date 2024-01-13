#include "math_fraction.h"
#include <algorithm>
#include <cmath>
#include <limits>
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

MathFraction::MathFraction(int64_t num, int64_t den, bool is_norm)
{
    num_ = num;
    den_ = den;
    if (is_norm) {
        Normalize();
    }
}

MathFraction::MathFraction(const MathDecimal &num, const MathDecimal &den, bool is_norm)
{
    num_ = num;
    den_ = den;
    if (is_norm) {
        Normalize();
    }
}

MathFraction::MathFraction(const MathDecimal &num, bool is_norm)
{
    num_ = num;
    den_ = 1;
    if (is_norm) {
        Normalize();
    }
}

MathFraction::MathFraction(const MathFraction &rhs)
{
    num_ = rhs.num_;
    den_ = rhs.den_;
}

MathFraction::~MathFraction()
{
}

uint16_t MathFraction::GetSigDigits() const
{
    return std::max(num_.GetSigDigits(), den_.GetSigDigits());
}

const MathFraction& MathFraction::operator = (const MathFraction &rhs)
{
    if (this != &rhs)
    {
        num_ = rhs.num_;
        den_ = rhs.den_;
    }
    return *this;
}

MathFraction& MathFraction::Add(const MathFraction &rhs, bool is_normalize)
{
    MathDecimal gcd_value = MathDecimal::GCD(den_, rhs.den_);
    MathDecimal den = den_ / gcd_value * rhs.den_;
    num_ = num_ * (rhs.den_ / gcd_value) + rhs.num_ * (den_ / gcd_value);
    den_ = den;

    if (is_normalize) {
        Normalize();
    }
    return *this;
}

MathFraction& MathFraction::operator += (const MathFraction &rhs)
{
    *this = *this + rhs;
    return *this;
}

MathFraction& MathFraction::operator -= (const MathFraction &rhs)
{
    *this = *this - rhs;
    return *this;
}

MathFraction& MathFraction::operator *= (const MathFraction &rhs)
{
    *this = *this * rhs;
    return *this;
}

MathFraction& MathFraction::operator /= (const MathFraction &rhs)
{
    *this = *this / rhs;
    return *this;
}

MathFraction MathFraction::operator + (const MathFraction &rhs) const
{
    MathDecimal gcd_value = MathDecimal::GCD(den_, rhs.den_);
    MathFraction ret_val;
    ret_val.den_ = den_ / gcd_value * rhs.den_;
    ret_val.num_ = num_ * (rhs.den_ / gcd_value) + rhs.num_ * (den_ / gcd_value);
    ret_val.Normalize();
    return ret_val;
}

MathFraction MathFraction::operator - (const MathFraction &rhs) const
{
    MathDecimal gcd_value = MathDecimal::GCD(den_, rhs.den_);
    MathFraction ret_val;
    ret_val.den_ = den_ / gcd_value * rhs.den_;
    ret_val.num_ = num_ * (rhs.den_ / gcd_value) - rhs.num_ * (den_ / gcd_value);
    ret_val.Normalize();
    return ret_val;
}

MathFraction MathFraction::operator * (const MathFraction &rhs) const
{
    MathFraction ret_val;
    ret_val.num_ = num_ * rhs.num_;
    ret_val.den_ = den_ * rhs.den_;
    ret_val.Normalize();
    return ret_val;
}

MathFraction MathFraction::operator / (const MathFraction &rhs) const
{
    MathFraction ret_val;
    ret_val.num_ = num_ * rhs.den_;
    ret_val.den_ = den_ * rhs.num_;
    ret_val.Normalize();
    return ret_val;
}

MathFraction& MathFraction::Reciprocal()
{
    MathDecimal tmp = num_;
    num_ = den_;
    den_ = tmp;
    if (den_.Sign() < 0)
    {
        num_.Minus();
        den_.Minus();
    }
    return *this;
}

MathFraction& MathFraction::Minus()
{
    num_.Minus();
    return *this;
}

const MathFraction& MathFraction::Power(const MathFraction &frac)
{
    int sign = frac.Sign();
    if (sign == 0 && !num_.IsZero() && !den_.IsZero())
    {
        Set(1, 1);
        return *this;
    }

    MathFraction exp = frac;
    exp.Normalize(true); //force-to-int-fraction
    if (sign < 0)
    {
        exp *= MathFraction(-1);
        MathDecimal bak = num_;
        num_ = den_;
        den_ = bak;
    }

    Macro_RetIf(*this, exp.num_.Compare(exp.den_) == 0);
    if (exp.num_ >= 0xFFFF || exp.den_ >= 0xFFFF)
    {
        exp.NormalizeToDecimal();
        if (exp.num_ < 0xFFFF)
        {
            NormalizeToDecimal();
            num_.PowerDouble(exp.num_.ToDouble());
            return *this;
        }
        else
        {
            num_.SetErrorCode(MathDecimal::Error_Overflow);
            return *this;
        }
    }

    uint32_t exp_num = exp.num_.ToInt32(), exp_den = exp.den_.ToInt32();
    num_.Power(exp_num, exp_den);
    den_.Power(exp_num, exp_den);
    Normalize();
    return *this;
}

const MathFraction& MathFraction::Power(int16_t exp)
{
    if (exp == 0 && !num_.IsZero() && !den_.IsZero()) {
        Set(1, 1);
        return *this;
    }

    if (exp < 0)
    {
        exp = -exp;
        MathDecimal bak = num_;
        num_ = den_;
        den_ = bak;
    }

    Macro_RetIf(*this, exp == 1);
    num_.Power(exp);
    den_.Power(exp);
    Normalize();
    return *this;
}

//static
MathFraction MathFraction::Power(const MathFraction &base, int16_t exp)
{
    MathFraction d = base;
    d.Power(exp);
    return d;
}

const MathFraction& MathFraction::Log(const MathFraction &base)
{
    int base_cmp = base.num_.Compare(base.den_);
    if (this->Sign() <= 0 || base.Sign() <= 0 || base_cmp == 0)
    {
        num_.SetErrorCode(MathDecimal::Error_Std);
        return *this;
    }

    int cmp_ret = num_.Compare(den_);
    if (cmp_ret == 0)
    {
        Set(0, 1);
        return *this;
    }

    MathFraction base_frac = base;
    base_frac.Normalize(true); //force-to-int-fraction
    this->Normalize(true);

    bool is_neg = (base_cmp > 0 && cmp_ret < 0) || (base_cmp < 0 && cmp_ret > 0);
    bool has_ratio = false, has_symbolic_result = true;
    MathFraction ratio(1, 1), temp_ratio;
    map<uint64_t, uint16_t> factor_map1, factor_map2;
    for (int pos_idx = 0; has_symbolic_result && pos_idx < 2; pos_idx++)
    {
        const MathDecimal *base_part = pos_idx == 0 ? &base_frac.num_ : &base_frac.den_;
        base_part->PrimeFactorization(factor_map1);
        const MathDecimal *my_part = ((pos_idx == 0 && is_neg) || (pos_idx > 0 && !is_neg)) ? &den_ : &num_;
        my_part->PrimeFactorization(factor_map2);

        if (factor_map1.size() != factor_map2.size()) {
            has_symbolic_result = false;
            break;
        }

        auto iter1 = factor_map1.begin(), iter2 = factor_map2.begin();
        for (; iter1 != factor_map1.end() && iter2 != factor_map2.end(); iter1++, iter2++)
        {
            if (iter1->first != iter2->first) {
                has_symbolic_result = false;
                break;
            }

            temp_ratio.Set(iter2->second, iter1->second, true);
            if (!has_ratio) {
                ratio = temp_ratio;
            }
            else if (ratio.Compare(temp_ratio) != 0) {
                has_symbolic_result = false;
                break;
            }
        }
    }

    if (has_symbolic_result)
    {
        this->Set(ratio);
        if (is_neg) {
            Minus();
        }
        return *this;
    }

    //numerical results
    NormalizeToDecimal();
    base_frac.NormalizeToDecimal();
    //the follow line is better than "d_value = log(num_.ToDouble())", which may generates inf for large numbers
    double d_value = log((double)num_.GetSig()) + num_.GetExp() * log((double)10);
    d_value /= log(base_frac.num_.ToDouble());
    num_.SetDouble(d_value);
    return *this;
}

//static
MathFraction MathFraction::Log(const MathFraction &value, const MathFraction &base)
{
    MathFraction d = value;
    d.Log(base);
    return d;
}

void MathFraction::Clear()
{
    num_.Set(0, 0);
    den_.Set(1, 0);
}

void MathFraction::Set(const MathFraction &rhs, bool is_norm)
{
    num_ = rhs.num_;
    den_ = rhs.den_;
    if (is_norm) {
        Normalize();
    }
}

void MathFraction::Set(int64_t num, int64_t den, bool is_norm)
{
    num_ = num;
    den_ = den;
    if (is_norm) {
        Normalize();
    }
}

void MathFraction::SetInt(int64_t val)
{
    num_.Set(val);
    den_.Set(1);
}

int64_t MathFraction::ToInt64() const
{
    return den_.IsOne() ? num_.ToInt64() : (num_ / den_).ToInt64();
}

//return values:
//  -1: <
//  0:  ==
//  1:  >
int MathFraction::Compare(const MathFraction &rhs) const
{
    MathFraction left, right;
    left.Set(*this, true);
    right.Set(rhs, true);

    left.num_ *= right.den_;
    right.num_ *= left.den_;
    return left.num_.Compare(right.num_);
}

int MathFraction::Compare(int64_t rhs) const
{
    return Compare(MathFraction(rhs));
}

bool MathFraction::IsInt() const
{
    Macro_RetFalseIf(den_.Sign() == 0);
    MathDecimal value = num_ / den_;
    return value.IsInt();
}

MathFraction& MathFraction::Normalize(bool bForceToIntFraction)
{
    if (den_.Sign() == 0)
    {
        num_ = num_.Sign() == 0 ? 0 : num_.Sign() < 0 ? -1 : 1;
        num_.SetErrorCode(MathDecimal::Error_Overflow);
        //den_ = 1;
        return *this;
    }

    if (den_.Sign() < 0)
    {
        den_ *= -1;
        num_ *= -1;
    }

    if (den_.GetSig() == 1 && !bForceToIntFraction)
    {
        if (den_.GetExp() != 0) {
            num_ *= MathDecimal(1, -den_.GetExp());
            den_.Set(1, 0);
        }
        return *this;
    }

    int16_t exp = num_.GetExp();
    if (exp < 0)
    {
        num_ *= MathDecimal(1, -exp);
        den_ *= MathDecimal(1, -exp);
    }
    exp = den_.GetExp();
    if (exp < 0)
    {
        num_ *= MathDecimal(1, -exp);
        den_ *= MathDecimal(1, -exp);
    }

    MathDecimal gcd_value = MathDecimal::GCD(den_, num_);
    num_ /= gcd_value;
    den_ /= gcd_value;
    return *this;
}

void MathFraction::NormalizeToDecimal()
{
    if (!den_.IsOne())
    {
        num_ /= den_;
        den_.Set(1, 0);
    }
}

std::string MathFraction::ToString() const
{
    DisplayFormat fmt;
    return ToString(fmt);
}

std::string MathFraction::ToString(const DisplayFormat &fmt) const
{
    if (den_.Compare(MathDecimal(1, 0)) == 0) {
        return num_.ToString(fmt.dec_fmt);
    }

    string str = num_.ToString(fmt.dec_fmt);
    str += '/';
    str += den_.ToString(fmt.dec_fmt);
    return str;
}

bool MathFraction::Read(IN RawWStrStream &stream, IN const UnicodeTable &uct)
{
    ReadOptions opt;
    bool ret = Read(stream, uct, opt);
    return ret;
}

bool MathFraction::Read(IN RawWStrStream &stream, IN const UnicodeTable &uct,
    IN const ReadOptions &opt)
{
    bool ret = true;
    Clear();

    //Latex fraction example: 3\frac{1}{8}
    bool is_latex = opt.accept_latex && stream.offset < stream.size && stream.data[stream.offset] == L'\\';
    if (!is_latex)
    {
        ret = num_.Read(stream);
        Macro_RetFalseIf(!ret);

        if (opt.accept_latex && num_.IsInt()) {
            is_latex = stream.offset < stream.size && stream.data[stream.offset] == L'\\';
        }
    }

    uint32_t offset_bak = stream.offset;
    if (is_latex)
    {
        const wchar_t *wcs = stream.data + stream.offset;
        bool is_latex_frac = stream.offset + 5 < stream.size && wcsncmp(wcs, L"\\frac", 5) == 0;
        Macro_RetIf(true, !is_latex_frac);

        MathFraction frac;
        stream.offset += 5;
        bool is_succ = stream.offset < stream.size && stream.data[stream.offset] == L'{';
        if (is_succ)
        {
            stream.offset++;
            is_succ = frac.num_.Read(stream);
            is_succ = is_succ && stream.offset < stream.size && stream.data[stream.offset] == L'}';
            stream.offset++;
        }

        is_succ = stream.offset < stream.size && stream.data[stream.offset] == L'{';
        if (is_succ)
        {
            stream.offset++;
            is_succ = frac.den_.Read(stream);
            is_succ = is_succ && stream.offset < stream.size && stream.data[stream.offset] == L'}';
            stream.offset++;
        }

        if (!is_succ) {
            stream.offset = offset_bak;
            return true;
        }

        int sign = Sign();
        if (sign > 0) {
            (*this) += frac;
        }
        else if(sign < 0) {
            (*this) -= frac;
        }
        else { //sign == 0
            *this = frac;
        }

        return true;
    }

    bool is_flat_mixed_frac = false;
    MathDecimal whole_number;
    if (opt.accept_flat_mixed_frac && stream.offset + 3 < stream.size && num_.IsInt())
    {
        wchar_t wch = stream.data[stream.offset];
        const auto &uci = uct.Get(wch);
        if (uci.norm == L' ')
        {
            whole_number = num_;
            stream.offset++;
            const auto &next_uci = uct.Get(stream.data[stream.offset]);
            ret = (next_uci.norm != L'+' && next_uci.norm != L'-') ? num_.Read(stream) : false;
            if (ret && num_.Sign() >= 0) {
                is_flat_mixed_frac = true;
            }
            else {
                num_ = whole_number;
                stream.offset = offset_bak;
            }
        }
    }

    bool has_den = false;
    if (opt.accept_flat_frac || is_flat_mixed_frac)
    {
        for (; stream.offset < stream.size; stream.offset++)
        {
            wchar_t wch = stream.data[stream.offset];
            const auto &uci = uct.Get(wch);

            if (uci.prime_type == UctPrime::Separator) {
                continue;
            }

            if (wch == L'/' || (!is_flat_mixed_frac && opt.accept_ratio && uci.norm == L':'))
            {
                stream.offset++;
                ret = den_.Read(stream);
                if (ret)
                {
                    has_den = true;
                }
            }

            break;
        }
    }

    if (!has_den)
    {
        stream.offset = offset_bak;
        if (is_flat_mixed_frac) {
            num_ = whole_number;
        }
        den_.Set(1);
    }

    if (is_flat_mixed_frac && has_den)
    {
        bool is_negative_whole = whole_number.Sign() < 0;
        if (is_negative_whole) {
            whole_number.Minus();
        }
        num_ += whole_number * den_;
        if (is_negative_whole) {
            num_.Minus();
        }
    }

    return true;
}

} //end of namespace
