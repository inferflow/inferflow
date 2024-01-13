#include "task_monitor.h"
#include <iostream>
#include "log.h"
#include "string_util.h"

using namespace std;

namespace sslib
{

TaskMonitor::TaskMonitor(uint32_t tick)
{
    Start(tick);
}

void TaskMonitor::Start(uint32_t tick)
{
    start_time_ = chrono::steady_clock::now();
    tick_ = tick > 0 ? tick : 1;
    count_ = 0;
    is_end_ = false;
}

void TaskMonitor::Progress(int count)
{
    Progress((uint64_t)count);
}

void TaskMonitor::Progress(uint32_t count)
{
    Progress((uint64_t)count);
}

void TaskMonitor::Progress(uint64_t count)
{
    count_ = count;
    if (count % (5 * tick_) == 0)
    {
        cout << "+";
        cout.flush();
        dot_count_in_this_line_++;
    }
    else if (count % tick_ == 0)
    {
        cout << ".";
        cout.flush();
        dot_count_in_this_line_++;
    }

    if (count % (10 * tick_) == 0) {
        cout << " ";
    }

    if (count % (50 * tick_) == 0)
    {
        std::chrono::steady_clock::time_point current_time = chrono::steady_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(current_time - start_time_).count();

        char buf[64];
        ElapsedTime2String(buf, elapsed_time);
        if (count % 1000000 == 0) {
            LogKeyInfo("%uM; %s", (uint32_t)(count / 1000000), buf);
        }
        else if (count % 100000 == 0) {
            LogKeyInfo("%.1fM; %s", count / 1000000.0f, buf);
        }
        else if (count % 1000 == 0) {
            LogKeyInfo("%uK; %s", (uint32_t)(count / 1000), buf);
        }
        else {
            LogKeyInfo("%u; %s", (uint32_t)count, buf);
        }

        dot_count_in_this_line_ = 0;
    }
}

void TaskMonitor::ShowElapsedTime(const std::wstring &caption_str)
{
    if (!is_end_) {
        end_time_ = chrono::steady_clock::now();
    }
    auto elapsed_time = chrono::duration_cast<chrono::microseconds>(end_time_ - start_time_).count();
    if (dot_count_in_this_line_ > 0) {
        dot_count_in_this_line_ = 0;
        cout << endl;
    }

    if (elapsed_time < 100 * 1000)
    {
        if (caption_str.empty()) {
            LogKeyInfo(L"Time cost: %.3f ms", elapsed_time / 1000.0f);
        }
        else {
            LogKeyInfo(L"%ls: %.3f ms", caption_str.c_str(), elapsed_time / 1000.0f);
        }
        return;
    }

    char buf[64];
    ElapsedTime2String(buf, elapsed_time / 1000);
    if (caption_str.empty()) {
        LogKeyInfo(L"Time cost: %ls", StringUtil::Utf8ToWideStr(buf).c_str());
    }
    else {
        LogKeyInfo(L"%ls: %ls", caption_str.c_str(), StringUtil::Utf8ToWideStr(buf).c_str());
    }
}

uint64_t TaskMonitor::GetElapsedTime(bool is_millisec) const
{
    std::chrono::steady_clock::time_point theTime = is_end_ ? end_time_ : chrono::steady_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::microseconds>(theTime - start_time_).count();
    return is_millisec ? uint64_t(elapsed_time / 1000) : uint64_t(elapsed_time);
}

void TaskMonitor::ShowProgress()
{
    if (!is_end_) {
        end_time_ = chrono::steady_clock::now();
    }
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time_ - start_time_).count();
    if (dot_count_in_this_line_ > 0) {
        dot_count_in_this_line_ = 0;
        cout << endl;
    }

    char buf[64];
    ElapsedTime2String(buf, elapsed_time);
    LogKeyInfo("Count: %ju; Time cost: %s", count_, buf);
}

void TaskMonitor::End()
{
    ShowProgress();
    is_end_ = true;
}

void TaskMonitor::Progress(uint32_t count, uint32_t tick)
{
    if (count % (5 * tick) == 0)
    {
        cout << "+";
        cout.flush();
    }
    else if (count % tick == 0)
    {
        cout << ".";
        cout.flush();
    }

    if (count % (10 * tick) == 0) {
        cout << " ";
    }

    if (count % (50 * tick) == 0) {
        LogKeyInfo("%u", count);
    }
}

void TaskMonitor::ElapsedTime2String(char buf[], uint64_t elapsed_time)
{
    if (elapsed_time < 1000)
    { //less than one second
        sprintf(buf, "%d ms", (int)elapsed_time);
    }
    else if (elapsed_time < 60 * 1000)
    { //less than 60 seconds (or one minute)
        sprintf(buf, "%.2fs", elapsed_time / 1000.0f);
    }
    else if (elapsed_time < 60 * 60 * 1000)
    { //less than 3600 seconds (or one hour)
        int sec = (int)(elapsed_time / 1000);
        sprintf(buf, "%02u:%02u", sec / 60, sec % 60);
        //sprintf(buf, "%ds (%.02fmin)", sec, sec / 60.0f);
    }
    else
    {
        int sec = (int)(elapsed_time / 1000);
        sprintf(buf, "%02u:%02u:%02u", sec / 3600, sec / 60 % 60, sec % 60);
        //sprintf(buf, "%ds (%.02fh)", sec, sec / 3600.0f);
    }
}

} //end of namespace
