#pragma once

#include <chrono>
#include <string>

namespace sslib
{

class TaskMonitor
{
public:
    TaskMonitor(uint32_t tick = 1);

    void Start(uint32_t tick = 1);

    void Progress(int count);
    void Progress(uint32_t count);
    void Progress(uint64_t count);

    void ShowElapsedTime(const std::wstring &caption_str = L"");

    //is_millisec:
    //  true:   in millisecond
    //  false:  in microsecond
    uint64_t GetElapsedTime(bool is_millisec = true) const;

    void ShowProgress();

    void End();

    static void Progress(uint32_t count, uint32_t tick);

protected:
    std::chrono::steady_clock::time_point start_time_;
    uint32_t tick_;
    uint64_t count_ = 0;

    std::chrono::steady_clock::time_point end_time_;
    bool is_end_ = false;

private:
    uint32_t dot_count_in_this_line_ = 0;

private:
    void ElapsedTime2String(char buf[], uint64_t elapsed_time);
};

} //end of namespace
