#pragma once

#include <string>
#include "prime_types.h"
#include "math_decimal.h"

namespace sslib
{

enum class TimeLevel : uint16_t
{
    Invalid = 0, Century, Year, Season, Month, Week, Day, WeekDay,
    Hour, Minute, Second, Unknown = 0xFFFE, Maximum = 0xFFFF
};

class TimeSpan
{
public:
    //A single tick represents 100 nanoseconds or one ten-millionth of a second. There are 10,000 ticks in a millisecond. 
    TimeSpan(int64_t ticks = 0);
    TimeSpan(const MathDecimal &sec);
    TimeSpan(int64_t hours, int64_t minutes, int64_t seconds);
    TimeSpan(int64_t days, int64_t hours, int64_t minutes, int64_t seconds,
        int64_t milliseconds = 0);

    void SetBySeconds(const MathDecimal &sec);
    void SetByTicks(int64_t ticks);
    void Set(int64_t hours, int64_t minutes, int64_t seconds);
    void Set(int64_t days, int64_t hours, int64_t minutes, int64_t seconds,
        int64_t milliseconds = 0);

    void Set(int64_t years, int64_t months, int64_t days, int64_t hours,
        int64_t minutes, int64_t seconds, int64_t milliseconds = 0);
    void SetByYearMonthDays(int64_t years, int64_t months, int64_t days);

    int64_t GetYears_Int64() const;
    int64_t GetMonths_Int64() const;
    MathDecimal GetYears() const;
    MathDecimal GetMonths() const;
    MathDecimal GetDays() const;
    MathDecimal GetHours() const;
    MathDecimal GetMinutes() const;
    MathDecimal GetSeconds() const;
    MathDecimal GetMilliseconds() const;
    MathDecimal GetTicks() const;

    int Compare(const TimeSpan &rhs, bool is_strict) const;

protected:
    int64_t months_;
    MathDecimal seconds_;
};

} //end of namespace
