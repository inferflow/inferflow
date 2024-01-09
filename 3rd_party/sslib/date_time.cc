#include "date_time.h"
#include <algorithm>
#include <ctime>
#include "log.h"
#ifdef _WIN32
#   include <windows.h>
#endif //_WIN32

using namespace std;

namespace sslib
{

const int32_t DateTimeConstData::MaxYear;
const int32_t DateTimeConstData::MinYear;

const DateTimeConstData g_timeConstData;

DateTimeConstData::DateTimeConstData()
{
    //days_of_acc_years = new int32_t[MaxYear-MinYear+2];

    days_of_month_[0] = 31; //Jan
    days_of_month_[1] = 28; //Feb
    days_of_month_[2] = 31; //Mar
    days_of_month_[3] = 30;
    days_of_month_[4] = 31; //May
    days_of_month_[5] = 30;
    days_of_month_[6] = 31; //Jul
    days_of_month_[7] = 31; //Aug
    days_of_month_[8] = 30;
    days_of_month_[9] = 31; //Oct
    days_of_month_[10] = 30;
    days_of_month_[11] = 31; //Dec

    days_of_acc_months_[0] = days_of_month_[0];
    for (uint32_t month = 1; month < 12; month++) {
        days_of_acc_months_[month] = days_of_acc_months_[month - 1] + days_of_month_[month];
    }

    int32_t acc_days = 0;
    for (int32_t year = 1; year <= MaxYear; year++)
    {
        int32_t days = DateTime::IsLeapYear_Inner(year) ? 366 : 365;
        days_of_acc_years[ZeroYearIndex + year] = acc_days + days;
        acc_days += days;
    }

    acc_days = 0;
    days_of_acc_years[ZeroYearIndex] = 0;
    for (int32_t year = 0; year > MinYear; year--)
    {
        int32_t days = DateTime::IsLeapYear_Inner(year) ? 366 : 365;
        days_of_acc_years[ZeroYearIndex + year - 1] = acc_days - days;
        acc_days -= days;
    }
}

DateTimeConstData::~DateTimeConstData()
{
    //if (days_of_acc_years != nullptr)
    //{
    //    delete[] days_of_acc_years;
    //    days_of_acc_years = nullptr;
    //}
}

DateTime::DateTime()
{
    Set(0);
}

DateTime::DateTime(int64_t ticks)
{
    Set(ticks);
}

DateTime::DateTime(int32_t year, uint16_t month, uint16_t day)
{
    Set(year, month, day, 0, 0, 0);
}

DateTime::DateTime(int32_t year, uint16_t month, uint16_t day,
    uint16_t hour, uint16_t minute, uint16_t second, uint16_t millisecond)
{
    Set(year, month, day, hour, minute, second, millisecond);
}

DateTime::DateTime(const RawDateTime &rdt)
{
    Set(rdt);
}

void DateTime::Set(int64_t ticks)
{
    DateTime max_dt = MaxValue();
    if (ticks > max_dt.ticks_) {
        *this = max_dt;
        return;
    }

    DateTime min_dt = MinValue();
    if (ticks < min_dt.ticks_) {
        *this = min_dt;
        return;
    }

    ticks_ = ticks;
    uint64_t delta_ticks = ticks - min_dt.ticks_;
    uint64_t seconds = delta_ticks / 10000 / 1000;
    remaining_ticks_ = delta_ticks % 10000;
    millisecond_ = delta_ticks / 10000 % 1000;
    second_ = seconds % 60;
    minute_ = seconds / 60 % 60;
    hour_ = seconds / 60 / 60 % 24;
    int32_t delta_days = (int32_t)(seconds / 60 / 60 / 24);

    int32_t left = 0, right = DateTimeConstData::MaxYear - DateTimeConstData::MinYear + 1;
    int32_t mid = (left + right) / 2;
    while (right > left)
    {
        if (g_timeConstData.days_of_acc_years[mid] - g_timeConstData.days_of_acc_years[0] > delta_days) {
            right = mid;
        }
        else if (g_timeConstData.days_of_acc_years[mid + 1] - g_timeConstData.days_of_acc_years[0] <= delta_days) {
            left = mid + 1;
        }
        else {
            break;
        }

        mid = (left + right) / 2;
    }

    year_ = mid - DateTimeConstData::ZeroYearIndex + 1;
    day_of_year_ = 1 + (uint16_t)(delta_days - (g_timeConstData.days_of_acc_years[mid] - g_timeConstData.days_of_acc_years[0]));
    Macro_RetxVoidIf(day_of_year_ > 366, LogError("Invalid day-of-year"));

    day_of_week_ = (DayOfWeek)((1 + 7 + (delta_days + g_timeConstData.days_of_acc_years[0]) % 7) % 7);

    bool is_leap_year = IsLeapYear_Inner(year_);
    month_ = 1;
    day_ = day_of_year_;
    for (uint16_t month = 2; month <= 12; month++)
    {
        if (day_of_year_ <= g_timeConstData.GetAccumulatedDaysByMonth((month - 1) - 1, is_leap_year)) {
            break;
        }

        if (day_of_year_ <= g_timeConstData.GetAccumulatedDaysByMonth(month - 1, is_leap_year))
        {
            month_ = month;
            day_ = day_of_year_ - g_timeConstData.GetAccumulatedDaysByMonth((month - 1) - 1, is_leap_year);
            break;
        }
    }
}

void DateTime::Set(int32_t year, uint16_t month, uint16_t day)
{
    Set(year, month, day, 0, 0, 0);
}

void DateTime::Set(int32_t year, uint16_t month, uint16_t day,
    uint16_t hour, uint16_t minute, uint16_t second, uint16_t millisecond)
{
    if (year == 0) {
        Set(0);
    }

    int32_t inner_year = year > 0 ? year : year + 1;
    year_ = max(DateTimeConstData::MinYear, min(inner_year, DateTimeConstData::MaxYear));

    month_ = min(max(1, (int)month), 12);
    hour_ = min(max(0, (int)hour), 23);
    minute_ = min(max(0, (int)minute), 59);
    second_ = min(max(0, (int)second), 59);
    millisecond_ = min(max(0, (int)millisecond), 999);
    remaining_ticks_ = 0;

    bool is_leap_year = IsLeapYear_Inner(year_);
    uint16_t day_of_cur_month = g_timeConstData.GetDaysOfMonth(month_ - 1, is_leap_year);
    day_ = min(max(1, (int)day), (int)day_of_cur_month);

    UpdateSecondaryData();
}

void DateTime::Set(const RawDateTime &rdt)
{
    Set(rdt.year, rdt.month, rdt.day, rdt.hour, rdt.minute, rdt.second);
}

DateTime DateTime::GetDate() const
{
    DateTime dt(year_, month_, day_);
    return dt;
}

RawDateTime DateTime::ToRawDateTime(TimeLevel lv) const
{
    RawDateTime rdt;
    if (lv >= TimeLevel::Year) {
        rdt.year = GetYear();
    }
    if (lv >= TimeLevel::Month) {
        rdt.month = GetMonth();
    }
    if (lv >= TimeLevel::Day) {
        rdt.day = GetDay();
    }
    if (lv >= TimeLevel::Hour) {
        rdt.hour = GetHour();
    }
    if (lv >= TimeLevel::Minute) {
        rdt.minute = GetMinute();
    }
    if (lv >= TimeLevel::Second) {
        rdt.second = GetSecond();
    }

    rdt.day_of_week = GetDayOfWeek();
    return rdt;
}

int DateTime::CompareTo(const DateTime &value) const
{
    return ticks_ < value.ticks_ ? -1 : (ticks_ == value.ticks_ ? 0 : 1);
}

DateTime& DateTime::Add(const TimeSpan &value)
{
    Set(ticks_ + value.GetTicks().ToInt64());
    return *this;
}

DateTime& DateTime::AddYears(int32_t value)
{
    int32_t year = year_ + value;
    year_ = max(DateTimeConstData::MinYear, min(year, DateTimeConstData::MaxYear));

    bool is_leap_year = IsLeapYear_Inner(year_);
    uint16_t day_of_cur_month = g_timeConstData.GetDaysOfMonth(month_ - 1, is_leap_year);
    day_ = min(max(1, (int)day_), (int)day_of_cur_month);

    UpdateSecondaryData();
    return *this;
}

// The description of the corresponding C# function in MSDN:
//    The AddMonths method calculates the resulting month and year, taking into 
// account leap years and the number of days in a month, then adjusts the day
// part of the resulting DateTime object. If the resulting day is not a valid
// day in the resulting month, the last valid day of the resulting month is used.
// For example, March 31st + 1 month = April 30th. The time-of-day part of the 
// resulting DateTime object remains the same as this instance.
DateTime& DateTime::AddMonths(int32_t value)
{
    int32_t year = year_ + ((int32_t)month_ - 1 + value) / 12;
    int32_t month_idx = 1 + ((int32_t)month_ - 1 + value) % 12;
    if (month_idx <= 0)
    {
        month_idx += 12;
        year--;
    }

    year_ = max(DateTimeConstData::MinYear, min(year, DateTimeConstData::MaxYear));
    month_ = (uint16_t)month_idx;

    bool is_leap_year = IsLeapYear_Inner(year_);
    uint16_t day_of_cur_month = g_timeConstData.GetDaysOfMonth(month_ - 1, is_leap_year);
    day_ = min(max(1, (int)day_), (int)day_of_cur_month);

    UpdateSecondaryData();
    return *this;
}

DateTime& DateTime::AddDays(int32_t value)
{
    int64_t delta_tick = value;
    delta_tick = delta_tick * 24 * 60 * 60 * 1000 * 10000;
    Set(ticks_ + delta_tick);
    return *this;
}

DateTime& DateTime::AddHours(int32_t value)
{
    int64_t delta_tick = value;
    delta_tick = delta_tick * 60 * 60 * 1000 * 10000;
    Set(ticks_ + delta_tick);
    return *this;
}

DateTime& DateTime::AddMinutes(int32_t value)
{
    int64_t delta_tick = value;
    delta_tick = delta_tick * 60 * 1000 * 10000;
    Set(ticks_ + delta_tick);
    return *this;
}

DateTime& DateTime::AddSeconds(int32_t value)
{
    int64_t delta_tick = value;
    delta_tick = delta_tick * 1000 * 10000;
    Set(ticks_ + delta_tick);
    return *this;
}

DateTime& DateTime::AddMilliseconds(int32_t value)
{
    int64_t delta_tick = value;
    delta_tick = delta_tick * 10000;
    Set(ticks_ + delta_tick);
    return *this;
}

DateTime& DateTime::AddTicks(int64_t value)
{
    Set(ticks_ + value);
    return *this;
}

DateTime& DateTime::AddUnits(int32_t value, TimeLevel unit_level)
{
    switch (unit_level)
    {
    case TimeLevel::Year:
        return AddYears(value);
        break;
    case TimeLevel::Month:
        return AddMonths(value);
        break;
    case TimeLevel::Day:
        return AddDays(value);
        break;
    case TimeLevel::Hour:
        return AddHours(value);
        break;
    case TimeLevel::Minute:
        return AddMinutes(value);
        break;
    case TimeLevel::Second:
        return AddSeconds(value);
        break;
    default:
        break;
    }

    return *this;
}

TimeSpan DateTime::Subtract(const DateTime &dt, TimeLevel lev) const
{
    TimeSpan ts;
    switch (lev)
    {
    case TimeLevel::Year:
        ts.SetByYearMonthDays(year_ - dt.year_, 0, 0);
        break;
    case TimeLevel::Month:
        ts.SetByYearMonthDays(year_ - dt.year_, (int64_t)(month_ - dt.month_), 0);
        break;
    case TimeLevel::Day:
        ts.Set(ticks_ / 10000000 / 86400 - dt.ticks_ / 10000000 / 86400, 0, 0, 0);
        break;
    default:
        ts.SetByTicks(ticks_ - dt.ticks_);
        break;
    }

    return ts;
}

string DateTime::ToString(int level) const
{
    char buf[128];
    switch (level)
    {
    case 1:
        sprintf(buf, "%04d", GetYear());
        break;
    case 2:
        sprintf(buf, "%04d-%02u", GetYear(), month_);
        break;
    case 3:
        sprintf(buf, "%04d-%02u-%02u", GetYear(), month_, day_);
        break;
    case 4:
        sprintf(buf, "%04d-%02u-%02u.%02u", GetYear(), month_, day_,
            hour_);
        break;
    case 5:
        sprintf(buf, "%04d-%02u-%02u.%02u:%02u", GetYear(), month_, day_,
            hour_, minute_);
        break;
    case 6:
        sprintf(buf, "%04d-%02u-%02u.%02u:%02u:%02u", GetYear(), month_, day_,
            hour_, minute_, second_);
        break;
    case 7:
        sprintf(buf, "%04d-%02u-%02u.%02u:%02u:%02u.%03u", GetYear(), month_, day_,
            hour_, minute_, second_, (int32_t)millisecond_);
        break;
    default:
        sprintf(buf, "%04d-%02u-%02u.%02u:%02u:%02u.%07u", GetYear(), month_, day_,
            hour_, minute_, second_, (int32_t)millisecond_ * 10000 + remaining_ticks_);
        break;
    }

    return buf;
}

//static
DateTime DateTime::MinValue()
{
    DateTime dt(DateTimeConstData::MinYear, 1, 1);
    return dt;
}

//static
DateTime DateTime::MaxValue()
{
    DateTime dt(DateTimeConstData::MaxYear, 12, 31, 23, 59, 59, 999);
    dt.remaining_ticks_ = 9999;
    dt.ticks_ += dt.remaining_ticks_;
    return dt;
}

//static
DateTime DateTime::Today(DateTimeKind kind)
{
    return Now(kind).GetDate();
}

//static
DateTime DateTime::Now(DateTimeKind kind)
{
    time_t rawtime;
    time(&rawtime);
    struct tm *ptm = nullptr;
    if (kind == DateTimeKind::Utc) {
        ptm = gmtime(&rawtime);
    }
    else {
        ptm = localtime(&rawtime);
    }

    DateTime dt(1900 + (int32_t)ptm->tm_year, 1 + (uint16_t)ptm->tm_mon, (uint16_t)ptm->tm_mday,
        (uint16_t)ptm->tm_hour, (uint16_t)ptm->tm_min, (uint16_t)ptm->tm_sec,
        0);
    return dt;

    //SYSTEMTIME tm;
    //if(kind == DateTimeKind::Utc) {
    //    GetSystemTime(&tm);
    //}
    //else {
    //    GetLocalTime(&tm);
    //}

    //DateTime dt((int32_t)tm.wYear, (uint16_t)tm.wMonth, (uint16_t)tm.wDay, (uint16_t)tm.wHour,
    //    (uint16_t)tm.wMinute, (uint16_t)tm.wSecond, (uint16_t)tm.wMilliseconds);
    //return dt;
}

//static
bool DateTime::IsLeapYear_Inner(int32_t inner_year)
{
    return inner_year % 400 == 0 ? true : (inner_year % 100 == 0 ? false : (inner_year % 4 == 0));
}

void DateTime::UpdateSecondaryData()
{
    bool is_leap_year = IsLeapYear_Inner(year_);
    day_of_year_ = day_;
    if (month_ > 1)
    {
        day_of_year_ += (uint16_t)g_timeConstData.GetAccumulatedDaysByMonth((month_ - 1) - 1, is_leap_year);
    }

    int32_t days = g_timeConstData.days_of_acc_years[DateTimeConstData::ZeroYearIndex + year_ - 1];
    days += (int32_t)(day_of_year_ - 1);

    day_of_week_ = (DayOfWeek)((1 + 7 + (days % 7)) % 7); //note that days can be negative
    ticks_ = (((((int64_t)days * 24 + hour_) * 60 + minute_) * 60 + second_) * 1000 + millisecond_) * 10000 + remaining_ticks_;
}

} //end of namespace
