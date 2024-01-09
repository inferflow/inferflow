#include "math_utils.h"
#include <cmath>
#include <limits>
#include "macro.h"
#include "number.h"

using namespace std;
namespace sslib
{

class DigitsAcc
{
public:
    uint8_t *digits, *last_zero_num;

public:
    DigitsAcc()
    {
        Init();
    }

    ~DigitsAcc()
    {
        if(digits != nullptr)
        {
            delete []digits;
            digits = nullptr;
        }
        if(last_zero_num != nullptr)
        {
            delete []last_zero_num;
            last_zero_num = nullptr;
        }
    }

    void Init()
    {
        digits = new uint8_t[100000];
        for(int n = 0; n < 10; n++) {
            digits[n] = 1;
        }
        for(int n = 10; n < 100000; n++) {
            digits[n] = digits[n/10] + 1;
        }

        last_zero_num = new uint8_t[100000];
        //last_zero_num[0] = 1;
        for(int n = 0; n < 10; n++) {
            last_zero_num[n] = 0;
        }
        for(int n = 10; n < 100000; n++) {
            last_zero_num[n] = n % 10 != 0 ? 0 : last_zero_num[n/10] + 1;
        }
    }
};

static DigitsAcc gs_digits;

///////////////////////////////////////////////////////////////////////////////////////
// struct UIntD36
///////////////////////////////////////////////////////////////////////////////////////

void UIntD36::ShiftDigitsRight(uint16_t k)
{
    while (k > 0)
    {
        uint16_t d = k < MathConst::Max10Pow ? k : MathConst::Max10Pow;
        k -= d;
        low /= MathConst::V10E[d];
        low += MathConst::V10E[MathConst::Max10Pow - d] * (high % MathConst::V10E[d]);
        high /= MathConst::V10E[d];
    }
}

void UIntD36::ShiftDigitsLeft(uint16_t k)
{
    (void)k;
}

///////////////////////////////////////////////////////////////////////////////////////
// class MathConst
///////////////////////////////////////////////////////////////////////////////////////

const uint16_t MathConst::Max10Pow;
uint64_t MathConst::V10E[Max10Pow+1];
int64_t MathConst::V2E31Neg = 0;
uint32_t MathConst::CacheSize_Primes = 0;
uint32_t *MathConst::LeastPrimeFactor = nullptr;
uint32_t *MathConst::Primes = nullptr;

MathConst::MathConst()
{
    Init();
}

MathConst::~MathConst()
{
    Free();
}

void MathConst::Init()
{
    V10E[0] = 1;
    for(uint32_t i = 1; i <= Max10Pow; i++) {
        V10E[i] = V10E[i-1] * 10;
    }
    V2E31Neg = 0x8000000000000000;

    /// Get all prime numbers under CacheSize_LPF, by Sieve of Eratosthenes
    LeastPrimeFactor = new uint32_t[CacheSize_LPF+1];
    LeastPrimeFactor[0] = 1;
    for(uint32_t num = 1; num <= CacheSize_LPF; num++) {
        LeastPrimeFactor[num] = num;
    }

    uint32_t product = 0;
    uint32_t p = 2;
    for(; p * p <= CacheSize_LPF; p++) //do NOT change to "p <= CacheSize_LPF"; otherwise the following "p*p" will overflow
    {
        if(LeastPrimeFactor[p] != p) { //not a prime number
            continue;
        }

        product = p * p;
        for(; product <= CacheSize_LPF; product += p)
        {
            if(LeastPrimeFactor[product] > p) {
                LeastPrimeFactor[product] = p;
            }
        }
    }

    CacheSize_Primes = 0;
    for(p = 2; p <= CacheSize_LPF; p++)
    {
        if(LeastPrimeFactor[p] == p) {
            CacheSize_Primes++;
        }
    }

    Primes = new uint32_t[CacheSize_Primes];
    for(uint32_t num = 2, idx = 0; num <= CacheSize_LPF && idx < CacheSize_Primes; num++)
    {
        if(LeastPrimeFactor[num] == num) {
            Primes[idx++] = num;
        }
    }
}

//static
void MathConst::Free()
{
    if(LeastPrimeFactor != nullptr) {
        delete []LeastPrimeFactor;
        LeastPrimeFactor = nullptr;
    }

    if(Primes != nullptr) {
        delete []Primes;
        Primes = nullptr;
    }
}

//static
uint32_t MathConst::Prime2Order(uint32_t num)
{
    if(num < CacheSize_LPF && LeastPrimeFactor[num] == num)
    {
        int low = 0, upp = num > CacheSize_Primes-1 ? (int)CacheSize_Primes-1 : (int)num;
        int mid = 0;
        while(low <= upp)
        {
            mid = low + (upp - low) / 2;
            if(num > Primes[mid]) {
                low = mid + 1;
            }
            else if(num < Primes[mid]) {
                upp = mid - 1;
            }
            else {
                return (uint32_t)mid;
            }
        }
    }

    return Number::MaxUInt32;
}

static MathConst gs_mathConst;

///////////////////////////////////////////////////////////////////////////////////////
// class MathUtils
///////////////////////////////////////////////////////////////////////////////////////

MathUtils::MathUtils()
{
}

uint32_t MathUtils::Digits(int64_t num)
{
    if(num == 0) {
        return 1;
    }
    else if(num == MathConst::V2E31Neg) {
        return 19;
    }

    if(num < 0) {
        num = -num;
    }

    int64_t q = num / MathConst::V10E[10];
    if(q == 0)
    {
        q = num / MathConst::V10E[5];
        return q == 0 ?  (uint32_t)gs_digits.digits[num % 100000] : 5 + (uint32_t)gs_digits.digits[q];
    }
    else
    {
        num = q / MathConst::V10E[5];
        return num == 0 ? 10 + (uint32_t)gs_digits.digits[q % 100000] : 15 + (uint32_t)gs_digits.digits[num];
    }
}

uint32_t MathUtils::LastZeroNum(int64_t num)
{
    if(num == 0 || num == MathConst::V2E31Neg || num % 10 != 0) {
        return 0;
    }

    if(num < 0) {
        num = -num;
    }

    int64_t r = num % MathConst::V10E[10];
    if(r == 0)
    {
        num /= MathConst::V10E[10];
        r = num % MathConst::V10E[5];
        return r == 0 ? 15 + (uint32_t)gs_digits.last_zero_num[num/MathConst::V10E[5]] : 10 + (uint32_t)gs_digits.last_zero_num[r];
    }
    else
    {
        num = r % MathConst::V10E[5];
        return num == 0 ? 5 + (uint32_t)gs_digits.last_zero_num[r/MathConst::V10E[5]] : (uint32_t)gs_digits.last_zero_num[num];
    }
}

uint32_t MathUtils::Digits2(int64_t num)
{
    if(num == 0) {
        return 1;
    }
    else if(num == MathConst::V2E31Neg) {
        return 19;
    }

    if(num < 0) {
        num = -num;
    }

    uint32_t n = 0;
    while(num > 0)
    {
        ++n;
        num /= 10;
    }
    return n;
}

uint32_t MathUtils::LastZeroNum2(int64_t num)
{
    if(num == 0 || num == MathConst::V2E31Neg) {
        return 0;
    }

    if(num < 0) {
        num = -num;
    }

    uint32_t n = 0;
    while((num % 10) == 0)
    {
        ++n;
        num /= 10;
    }
    return n;
}

uint32_t MathUtils::Digits3(int32_t num)
{
    if(num == 0) {
        return 1;
    }
    else if((uint32_t)num == 0x80000000) {
        return 10;
    }

    double lg = log10(static_cast<double>(abs(num)));
    double eps = numeric_limits<double>::epsilon();
    return static_cast<uint32_t>(1 + lg + eps);
}

uint32_t MathUtils::Digits3(int64_t num)
{
    if(num == 0) {
        return 1;
    }
    else if(num == MathConst::V2E31Neg) {
        return 19;
    }

    if(num < 0) {
        num = -num;
    }

    //double lg = log10(static_cast<double>(num));
    //double eps = numeric_limits<double>::epsilon();
    //return static_cast<int>(1 + lg + eps);
    return static_cast<uint32_t>(1 + log10(static_cast<double>(num)) + numeric_limits<double>::epsilon());
}

//void MathUtils::Product(IntD36 &res, const int64_t &n1, const int64_t &n2)
//{
//    res.Clear();
//    uint64_t u1 = n1 >= 0 ? n1 : -n1;
//    uint64_t u2 = n2 >= 0 ? n2 : -n2;
//    Product(res, u1, u2);
//    if (n1 > 0 && n2 < 0 || n1 < 0 && n2 > 0) {
//        res.m_high = -res.m_high;
//    }
//}

void MathUtils::Product(UIntD36 &res, uint64_t n1, uint64_t n2)
{
    res.Clear();
    uint64_t a1 = n1 / MathConst::V10E[9], b1 = n1 % MathConst::V10E[9];
    uint64_t a2 = n2 / MathConst::V10E[9], b2 = n2 % MathConst::V10E[9];
    uint64_t m = a1 * b2 + a2 * b1;
    res.low = b1 * b2 + (m % MathConst::V10E[9]) * MathConst::V10E[9];
    res.high = a1 * a2 + m / MathConst::V10E[9];
    if (res.low >= MathConst::V10E[18])
    {
        res.high += res.low / MathConst::V10E[18];
        res.low = res.low % MathConst::V10E[18];
    }
}

//static
float MathUtils::Log2(float v)
{
    //Functions like log2f and expf are optimized in glibc_2.27.
    //When glibc_2.27 or newer is installed,
    //binaries built will rely on glibc_2.27 if log2(float) or log2f is called.
    //To create portable Linux binaries, we choose not to call log2f:
    return (float)log2((double)v);
    //return log(v) / log((float)2); //logf also has the glibc_2.27 issue
    //return log2f(v);
}

//static
float MathUtils::Log(float v)
{
    return (float)log((double)v);
}

//static
float MathUtils::Exp(float v)
{
    return (float)exp((double)v);
}

//static
float MathUtils::Expf(float v)
{
    return (float)exp((double)v);
}

} //end of namespace
