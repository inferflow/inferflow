#include "number.h"

namespace sslib
{

using namespace std;

//static
const uint32_t Number::MaxUInt32;

//static
string Number::ToString(uint32_t n)
{
    char buf[16];
    sprintf(buf, "%u", n);
    return buf;
}

//static
string Number::ToString(int32_t n)
{
    char buf[16];
    sprintf(buf, "%d", n);
    return buf;
}

//static
string Number::ToString(uint64_t n)
{
    char buf[32];
    sprintf(buf, "%ju", n);
    return buf;
}

//static
string Number::ToString(int64_t n)
{
    char buf[32];
    sprintf(buf, "%jd", n);
    return buf;
}

//static
wstring Number::ToWString(uint32_t n)
{
    wchar_t buf[16 + 1];
    swprintf(buf, 16, L"%u", n);
    return buf;
}

//static
wstring Number::ToWString(int32_t n)
{
    wchar_t buf[16 + 1];
    swprintf(buf, 16, L"%d", n);
    return buf;
}

//static
wstring Number::ToWString(uint64_t n)
{
    wchar_t buf[32 + 1];
    swprintf(buf, 32, L"%ju", n);
    return buf;
}

//static
wstring Number::ToWString(int64_t n)
{
    wchar_t buf[32 + 1];
    swprintf(buf, 32, L"%jd", n);
    return buf;
}

}
