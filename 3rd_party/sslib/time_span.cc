#include "time_span.h"

namespace sslib
{

TimeSpan::TimeSpan(int64_t ticks)
{
    SetByTicks(ticks);
}

TimeSpan::TimeSpan(const MathDecimal &sec)
{
    SetBySeconds(sec);
}

TimeSpan::TimeSpan(int64_t hours, int64_t minutes, int64_t seconds)
{
    Set(hours, minutes, seconds);
}

TimeSpan::TimeSpan(int64_t days, int64_t hours, int64_t minutes,
    int64_t seconds, int64_t milliseconds)
{
    Set(days, hours, minutes, seconds, milliseconds);
}

void TimeSpan::SetBySeconds(const MathDecimal &sec)
{
    seconds_ = sec;
    months_ = 0;
}

void TimeSpan::SetByTicks(int64_t ticks)
{
    months_ = 0;
    seconds_.Set(ticks, -7); // 1 tick = 1E-7 seconds
}

void TimeSpan::Set(int64_t hours, int64_t minutes, int64_t seconds)
{
    Set(0, hours, minutes, seconds, 0);
}

void TimeSpan::Set(int64_t days, int64_t hours, int64_t minutes,
    int64_t seconds, int64_t milliseconds)
{
    months_ = 0;
    seconds_.Set(days);
    seconds_ *= 24;
    seconds_ += hours;
    seconds_ *= 60;
    seconds_ += minutes;
    seconds_ *= 60;
    seconds_ += seconds;
    seconds_ += MathDecimal(milliseconds, -3);
}

void TimeSpan::Set(int64_t years, int64_t months, int64_t days, int64_t hours,
    int64_t minutes, int64_t seconds, int64_t milliseconds)
{
    Set(days, hours, minutes, seconds, milliseconds);
    months_ = 12 * years + months;
}

void TimeSpan::SetByYearMonthDays(int64_t years, int64_t months, int64_t days)
{
    seconds_.Set(days);
    seconds_ *= 86400; //1 day = 86400 seconds
    months_ = 12 * years + months;
}

int64_t TimeSpan::GetYears_Int64() const
{
    return GetMonths_Int64() / 12;
}

int64_t TimeSpan::GetMonths_Int64() const
{
    Macro_RetIf(months_, seconds_.IsZero());

    //1 month = 30.436875 days = 2629746 seconds
    return months_ + (seconds_ / 2629746).ToInt64();
}

MathDecimal TimeSpan::GetYears() const
{
    return GetMonths() / 12;
}

MathDecimal TimeSpan::GetMonths() const
{
    MathDecimal ret(months_);
    Macro_RetIf(ret, seconds_.IsZero());

    //1 month = 30.436875 days = 2629746 seconds
    ret += seconds_ / 2629746;
    return ret;
}

MathDecimal TimeSpan::GetDays() const
{
    return GetSeconds() / 86400; //1 day = 86400 seconds
}

MathDecimal TimeSpan::GetHours() const
{
    return GetSeconds() / 3600;
}

MathDecimal TimeSpan::GetMinutes() const
{
    return GetSeconds() / 60;
}

MathDecimal TimeSpan::GetSeconds() const
{
    Macro_RetIf(seconds_, months_ == 0);

    MathDecimal sec(months_);
    sec *= MathDecimal(2629746); //1 month = 30.436875 days = 2629746 seconds
    sec += seconds_;
    return sec;
}

MathDecimal TimeSpan::GetMilliseconds() const
{
    return GetSeconds() * 1000;
}

MathDecimal TimeSpan::GetTicks() const
{
    return GetSeconds() * 10000000; //1 second = 10^7 ticks
}

int TimeSpan::Compare(const TimeSpan &rhs, bool is_strict) const
{
    if(is_strict)
    {
        if(months_ != rhs.months_) {
            return months_ < rhs.months_ ? -1 : 1;
        }
        return seconds_.Compare(rhs.seconds_);
    }

    return GetSeconds().Compare(rhs.GetSeconds());
}

} //end of namespace
