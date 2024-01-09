#pragma once

#include "time_span.h"
#include "macro.h"

namespace sslib
{

struct DateTimeConstData
{
public:
    const static int32_t MaxYear = 9999;
    const static int32_t MinYear = -9999;

    static const int32_t ZeroYearIndex = 0 - DateTimeConstData::MinYear;
    int32_t days_of_acc_years[MaxYear - MinYear + 2]; //acc: accumulated
    //int32_t m_arrDaysOfAccYears[9999 - (-9999) + 2]; //acc: accumulated

public:
    DateTimeConstData();
    virtual ~DateTimeConstData();

    uint16_t GetDaysOfMonth(uint16_t month, bool is_leap_year) const {
        return is_leap_year && month == 1 ? 29 : days_of_month_[month];
    }

    uint16_t GetAccumulatedDaysByMonth(uint16_t month, bool is_leap_year) const
    {
        return is_leap_year && month > 0 ? days_of_acc_months_[month] + 1 : days_of_acc_months_[month];
    }

protected:
    uint16_t days_of_month_[12]; //for common year
    uint16_t days_of_acc_months_[12]; //for common year (accumulation from Jan)
};

extern const DateTimeConstData g_time_const_data HiddenAttribute;

enum class DateTimeKind : uint16_t
{
    Utc = 0, Local
};

enum class DayOfWeek : uint16_t
{
    Sunday = 0, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Unknown
};

struct RawDateTime
{
    int32_t year = 0;
    uint16_t month = 0;
    uint16_t day = 0;
    uint16_t week = 0;
    uint16_t hour = Number::MaxUInt16;
    uint16_t minute = Number::MaxUInt16;
    uint16_t second = Number::MaxUInt16;
    DayOfWeek day_of_week = DayOfWeek::Unknown;

    void SetYMD(int32_t p_year, uint16_t p_month, uint16_t p_day)
    {
        year = p_year;
        month = p_month;
        day = p_day;
    }
};

//Proleptic Gregorian calendar is assumed
class DateTime
{
public:
    DateTime();
    //ticks: The number of 100-nanosecond intervals that have elapsed since 0001-01-01.00:00:00 in the Gregorian calendar
    DateTime(int64_t ticks);
    //month: The month (1 through 12); day: The day (1 through the number of days in month)
    DateTime(int32_t year, uint16_t month, uint16_t day);
    //hour: The hours (0 through 23); minute: The minutes (0 through 59); second: The seconds (0 through 59);
    //millisecond: The milliseconds (0 through 999)
    DateTime(int32_t year, uint16_t month, uint16_t day, uint16_t hour, uint16_t minute,
        uint16_t second, uint16_t millisecond = 0);
    DateTime(const RawDateTime &rdt);

    void Set(int64_t ticks);
    void Set(int32_t year, uint16_t month, uint16_t day);
    void Set(int32_t year, uint16_t month, uint16_t day, uint16_t hour, uint16_t minute,
        uint16_t second, uint16_t millisecond = 0);
    void Set(const RawDateTime &rdt);

    //Gets the date component of this instance
    DateTime GetDate() const;
    RawDateTime ToRawDateTime(TimeLevel lv = TimeLevel::Maximum) const;

    int32_t GetYear() const {
        return year_ <= 0 ? year_ - 1 : year_;
    }

    uint16_t GetMonth() const {
        return month_;
    }

    //Gets the day of the month
    uint16_t GetDay() const {
        return day_;
    }

    DayOfWeek GetDayOfWeek() const {
        return day_of_week_;
    }

    //The day of the year, expressed as a value between 1 and 366
    uint16_t GetDayOfYear() const {
        return day_of_year_;
    }

    uint16_t GetHour() const {
        return hour_;
    }

    uint16_t GetMinute() const {
        return minute_;
    }

    uint16_t GetSecond() const {
        return second_;
    }

    uint16_t GetMillisecond() const {
        return millisecond_;
    }

    int64_t GetTicks() const {
        return ticks_;
    }

    int CompareTo(const DateTime &value) const;

    //A positive or negative time interval
    DateTime& Add(const TimeSpan &value);
    //A number of years. The value parameter can be negative or positive.
    DateTime& AddYears(int32_t value);
    DateTime& AddMonths(int32_t value);
    DateTime& AddDays(int32_t value);
    DateTime& AddHours(int32_t value);
    DateTime& AddMinutes(int32_t value);
    DateTime& AddSeconds(int32_t value);
    DateTime& AddMilliseconds(int32_t value);
    DateTime& AddTicks(int64_t value);
    DateTime& AddUnits(int32_t value, TimeLevel unit_level);

    TimeSpan Subtract(const DateTime &dt, TimeLevel lev) const;

    //level:
    //  3: year-month-day
    //  6: year-month-day.hour:min:sec
    //  7: year-month-day.hour:min:sec.millisec
    std::string ToString(int level = 0) const;

    static DateTime MinValue();
    static DateTime MaxValue();
    static DateTime Today(DateTimeKind kind = DateTimeKind::Local);
    static DateTime Now(DateTimeKind kind = DateTimeKind::Local);
    static DateTime UtcNow() {
        return Now(DateTimeKind::Utc);
    }

protected:
    int32_t year_;
    uint16_t month_, day_, hour_, minute_, second_, millisecond_, remaining_ticks_;
    int64_t ticks_;
    DayOfWeek day_of_week_;
    uint16_t day_of_year_;

protected:
    static bool IsLeapYear_Inner(int32_t inner_year);
    void UpdateSecondaryData();

    friend DateTimeConstData;
};

} //end of namespace
