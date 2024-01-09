#include "log.h"
#include "macro.h"
#include <iostream>
#include <stdarg.h>
#include "path.h"
#include <set>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <mutex>
#include <thread>
#include "string_util.h"
#ifdef _WIN32
#   include <Windows.h>
#endif

using namespace std;

namespace sslib
{

const char* Logger::MsgLevelNames[] =
{
    "advice",
    "weak_warning",
    "warning",
    "error",
    "severe"
};

const char* Logger::MsgLevelInfoNames[] =
{
    "info_fussy",
    "info_debug",
    "info_ordinary",
    "info_status",
    "info_key",
    "info_milestone"
};

Logger Logger::log;

Logger::Logger()
{
	output_ptr_ = new StdLogOutput;
}

Logger::~Logger(void)
{
    if (output_ptr_ != nullptr)
    {
        delete output_ptr_;
        output_ptr_ = nullptr;
    }
}

const LogOutput* Logger::SetOutput(LogOutput *output)
{
	const LogOutput *old_ptr = output_ptr_;
    output_ptr_ = output;
	return old_ptr;
}

void Logger::Notify(LogMsg &msg)
{
    if (output_ptr_ != nullptr) {
        output_ptr_->OnNotify(msg);
    }
}

void Logger::Notify(int msg_level, const char *format, ...)
{
    if (output_ptr_ != nullptr && format != nullptr)
    {
        va_list args;
        va_start(args, format);
        Notify(nullptr, nullptr, -1, msg_level, format, args);
        va_end(args);
    }
}

void Logger::Notify(const char *file_path, const char *fn_name, int line,
    int msg_level, const char *format, ...)
{
    if (output_ptr_ != nullptr && format != nullptr)
    {
        va_list args;
        va_start(args, format);
        Notify(file_path, fn_name, line, msg_level, format, args);
        va_end(args);
    }
}

void Logger::Notify(const char *file_path, const char *fn_name, int line,
    int msg_level, const char *format, va_list &args)
{
    if (output_ptr_ != nullptr && format != nullptr)
    {
        char buf[MaxMessageSize + 1];
        int count = vsnprintf(buf, MaxMessageSize, format, args);
        if (count > 0 && count <= MaxMessageSize)
        {
            buf[count] = '\0';
            LogMsg msg(msg_level);

            msg.time_stamp = std::time(nullptr);

            msg.thread_id = 0; //(uint64_t)std::this_thread::get_id();
            msg.file_path = file_path;
            msg.function_name = fn_name;
            msg.line_in_source = line;
            msg.mbs_content.Set(buf, count);
            output_ptr_->OnNotify(msg);
        }
    }
}

void Logger::Notify(int msg_level, const wchar_t *format_str, ...)
{
    if (output_ptr_ != nullptr && format_str != nullptr)
    {
        va_list args;
        va_start(args, format_str);
        Notify(nullptr, nullptr, -1, msg_level, format_str, args);
        va_end(args);
    }
}

void Logger::Notify(const char *file_path, const char *fn_name, int line,
    int msg_level, const wchar_t *format_str, ...)
{
    if (output_ptr_ != nullptr && format_str != nullptr)
    {
        va_list args;
        va_start(args, format_str);
        Notify(file_path, fn_name, line, msg_level, format_str, args);
        va_end(args);
    }
}

void Logger::Notify(const char *file_path, const char *fn_name, int line,
    int msg_level, const wchar_t *format_str, va_list &args)
{
    if (output_ptr_ != nullptr && format_str != nullptr)
    {
        wchar_t buf[MaxMessageSize + 1];
        int count = vswprintf(buf, MaxMessageSize, format_str, args);
        if (count > 0 && count <= MaxMessageSize)
        {
            buf[count] = '\0';
            LogMsg msg(msg_level);

            msg.time_stamp = std::time(nullptr);

            msg.thread_id = 0;//(uint64_t)std::this_thread::get_id();
            msg.file_path = file_path;
            msg.function_name = fn_name;
            msg.line_in_source = line;
            msg.wide_content.Set(buf, count);
            msg.mbs_content.Set(nullptr, 0);
            output_ptr_->OnNotify(msg);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// SSLogHelper

SSLogHelper::SSLogHelper(Logger *logger, const char *file,
    const char *function, int line, int lev)
{
    logger_ptr_ = logger;
    file_name_ = file;
    function_ = function;
    line_ = line;
    level_ = lev;
}

void SSLogHelper::operator()(const char *format_str, ...)
{
    if (logger_ptr_ != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        logger_ptr_->Notify(file_name_, function_, line_, level_, format_str, ptr);
        va_end(ptr);
    }
}

void SSLogHelper::operator()(int msg_level, const char *format_str, ...)
{
    if (logger_ptr_ != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        logger_ptr_->Notify(file_name_, function_, line_, msg_level, format_str, ptr);
        va_end(ptr);
    }
}

void SSLogHelper::operator()(int msg_level, int msg_id, const char *msg_type,
    const char *format_str, ...)
{
    if (logger_ptr_ != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        logger_ptr_->Notify(file_name_, function_, line_, msg_id,
            msg_type, msg_level, format_str, ptr);
        va_end(ptr);
    }
}

void SSLogHelper::operator()(Logger *monitor, int msg_id, const char *msg_type,
    int msg_level, const char *format_str, ...)
{
    if (monitor == nullptr) {
        monitor = logger_ptr_;
    }

    if (monitor != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        monitor->Notify(file_name_, function_, line_, msg_id,
            msg_type, msg_level, format_str, ptr);
        va_end(ptr);
    }
}

void SSLogHelper::operator()(const wchar_t *format_str, ...)
{
    if (logger_ptr_ != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        logger_ptr_->Notify(file_name_, function_, line_, level_, format_str, ptr);
        va_end(ptr);
    }
}

void SSLogHelper::operator()(int msg_level, const wchar_t *format_str, ...)
{
    if (logger_ptr_ != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        logger_ptr_->Notify(file_name_, function_, line_, msg_level, format_str, ptr);
        va_end(ptr);
    }
}

void SSLogHelper::operator()(int msg_level, int msg_id, const char *msg_type,
    const wchar_t *format_str, ...)
{
    if (logger_ptr_ != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        logger_ptr_->Notify(file_name_, function_, line_, msg_id,
            msg_type, msg_level, format_str, ptr);
        va_end(ptr);
    }
}

void SSLogHelper::operator()(Logger *monitor, int msg_id, const char *msg_type,
    int msg_level, const wchar_t *format_str, ...)
{
    if (monitor == nullptr) {
        monitor = logger_ptr_;
    }

    if (monitor != nullptr)
    {
        va_list ptr;
        va_start(ptr, format_str);
        monitor->Notify(file_name_, function_, line_, msg_id,
            msg_type, msg_level, format_str, ptr);
        va_end(ptr);
    }
}

/////////////////////////////////////////////////////////////////////////////

StdLogOutput::StdLogOutput(int msg_output)
{
    max_file_idx_ = 1;
    msg_idx_ = 0;

    settings_.SetDefault();

    msg_output_ = msg_output;

    percent_ = 0;
    message_str_ = "";

    set<string, StrLessNoCase> color_terminal_set;
    color_terminal_set.insert("xterm");
    color_terminal_set.insert("rxvt");
    color_terminal_set.insert("vt100");
    color_terminal_set.insert("linux");
    color_terminal_set.insert("screen");

    const char *terminal_name = getenv("TERM");
    if (terminal_name != nullptr)
    {
        auto iter_find = color_terminal_set.find(terminal_name);
        is_color_terminal_ = iter_find != color_terminal_set.end();
        //cout << "Terminal: " << terminal_name << endl;
    }

    lock_ = new mutex;
}

StdLogOutput::~StdLogOutput()
{
    if (lock_ != nullptr) {
        delete (mutex*)lock_;
    }
    lock_ = nullptr;
}

bool StdLogOutput::Init(const string &dir, const string &name_prefix,
    const OpenOptions &options)
{
    dir_ = dir;
    name_prefix_ = name_prefix;
    max_file_idx_ = 1;
    existing_files_.resize(0);

    settings_ = options;
    if (settings_.max_size_per_file < 0) {
        settings_.max_size_per_file = 100 * 1024 * 1024;
    }

    if (!dir.empty()) {
        File::Mkdirs(dir_.c_str());
    }

    string idx_str;
    vector<string> file_list;
    string filter_str = name_prefix_ + "_*" + ".log";
    File::GetFileNames(file_list, dir_.c_str(), filter_str.c_str(), false);
    for (size_t file_idx = 0; file_idx < file_list.size(); file_idx++)
    {
        string &file_path = file_list[file_idx];
        if (file_path.size() > name_prefix_.size() + 5)
        {
            idx_str = file_path.substr(name_prefix_.size() + 1,
                file_path.size() - name_prefix_.size() - 5);
            int idx = atoi(idx_str.c_str());
            if (idx > 0)
            {
                existing_files_.push_back(idx);
                if (idx > max_file_idx_) {
                    max_file_idx_ = idx;
                }
            }
        }
    }

    std::sort(existing_files_.begin(), existing_files_.end());

    int del_to = settings_.max_file_count > 0
        ? ((int)existing_files_.size() - settings_.max_file_count)
        : 0;
    if (settings_.clear_existing) {
        del_to = (int)existing_files_.size();
    }

    string file_path;
    for (int file_idx = 0; file_idx < del_to; file_idx++)
    {
        ConstructFilePath(file_path, existing_files_[file_idx]);
        if (!file_path.empty()) {
            std::remove(file_path.c_str());
        }
    }

    bool ret = CheckCurStreamCapacity();
    if (ret) {
        current_stream_ << endl << "; --------------------------------------------" << endl;
    }

    return ret;
}

bool StdLogOutput::Close()
{
	current_stream_.close();
	return true;
}

void StdLogOutput::SetPercent(int percent)
{
	((mutex*)lock_)->lock();
	percent_ = percent;
    ((mutex*)lock_)->unlock();
}

int StdLogOutput::GetOutputMask()
{
	((mutex*)lock_)->lock();
	int mask = msg_output_;
	((mutex*)lock_)->unlock();

	return mask;
}

void StdLogOutput::SetOutputMask(int mask)
{
	((mutex*)lock_)->lock();
	msg_output_ = mask;
	((mutex*)lock_)->unlock();
}

int StdLogOutput::GetPercent()
{
	((mutex*)lock_)->lock();
	int percent = percent_;
	((mutex*)lock_)->unlock();

	return percent;
}

string StdLogOutput::RetrieveMessage(bool be_clear)
{
    ((mutex*)lock_)->lock();
    string msg_str = message_str_;
    if (be_clear) {
        message_str_.clear();
    }
    ((mutex*)lock_)->unlock();

    return msg_str;
}

//////////////////////////////////////////////////////////////////////////
// Virtual methods
//////////////////////////////////////////////////////////////////////////

void StdLogOutput::OpenOptions::SetDefault()
{
    lev_threshold = LogLevel::Advice;
    info_lev_threshold = LogLevel::Info_Ordinary;
    lev_threshold_for_console = LogLevel::WeakWarning;
    info_lev_threshold_for_console = LogLevel::Info_Key;
    clear_existing = false;
    max_size_per_file = 100 * 1024 * 1024; //100MB
    max_file_count = 50;
#ifdef _WIN32
    enable_color_console = true;
#else
    enable_color_console = false;
#endif
}

bool StdLogOutput::Flush()
{
    current_stream_.flush();
    return current_stream_.good();
}

void StdLogOutput::OnNotify(LogMsg &msg)
{
    bool has_wide_content = msg.wide_content.size > 0 && msg.wide_content.data != nullptr;
    bool is_info = (msg.level & LogLevel::Info_Fussy) != 0;
    bool is_satisfied = (is_info && msg.level >= settings_.info_lev_threshold)
        || (!is_info && msg.level >= settings_.lev_threshold);
    bool is_console_satisfied = (is_info && msg.level >= settings_.info_lev_threshold_for_console)
        || (!is_info && msg.level >= settings_.lev_threshold_for_console);
    if (!is_satisfied && !is_console_satisfied) {
        return;
    }

    ((mutex*)lock_)->lock();

    char msg_prime_meta[256], msg_file[8192];
    stringstream msg_level_stream;

    if (is_satisfied || is_console_satisfied)
    {
        msg_idx_++;

        //primary meta: idx, thread-id, id, time, and type
        std::tm *ptm = std::localtime(&msg.time_stamp);
        //snprintf(msg_prime_meta, 255, "#%d; thread-%ju; %d-%d-%d %d:%d:%d; ",
        //    msg_idx_, msg.m_threadId,
        //    ptm->tm_year, ptm->tm_mon, ptm->tm_mday,
        //    ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
        snprintf(msg_prime_meta, 255, "#%d; %d-%d-%d %d:%d:%d; ",
            msg_idx_,
            1900 + ptm->tm_year, 1 + ptm->tm_mon, ptm->tm_mday,
            ptm->tm_hour, ptm->tm_min, ptm->tm_sec);

        //msg level
        {
            msg_level_stream << hex << "0x" << msg.level << dec << "(";

            uint32_t lev = ((msg.level & LogLevel::Mask) >> 8);
            if (is_info)
            {
                uint32_t size = (uint32_t)(sizeof(Logger::MsgLevelInfoNames) / sizeof(const char*));
                if (lev >= size) {
                    lev = size - 1;
                }
                msg_level_stream << Logger::MsgLevelInfoNames[lev];
            }
            else
            {
                uint32_t size = (uint32_t)(sizeof(Logger::MsgLevelNames) / sizeof(const char*));
                if (lev >= size) {
                    lev = size - 1;
                }
                msg_level_stream << Logger::MsgLevelNames[lev];
            }

            uint32_t sub_lev = (msg.level & LogLevel::SubMask);
            if (sub_lev != 0) {
                msg_level_stream << "+" << sub_lev;
            }

            msg_level_stream << "); ";
        }

        //file, function, and line
        if (msg.file_path != nullptr || msg.function_name != nullptr || msg.line_in_source >= 0)
        {
            if (msg.file_path != nullptr)
            {
                File path(msg.file_path);
                snprintf(msg_file, 8191, "%s#%d@%s",
                    (msg.function_name != nullptr ? msg.function_name : ""),
                    msg.line_in_source, path.GetNameAndExt().c_str());
            }
            else
            {
                snprintf(msg_file, 8191, "%s#%d",
                    (msg.function_name != nullptr ? msg.function_name : ""),
                    msg.line_in_source);
            }
        }
        else
        {
            msg_file[0] = '\0';
        }

        if (is_satisfied)
        {
            bool is_cur_stream_valid = CheckCurStreamCapacity();
            if (is_cur_stream_valid && current_stream_.is_open())
            {
                current_stream_ << msg_prime_meta << msg_level_stream.str().c_str() << msg_file << endl;
                current_stream_ << "\t";
                if (has_wide_content)
                {
                    string content_str;
                    StringUtil::ToUtf8(content_str, msg.wide_content.data, msg.wide_content.size);
                    current_stream_.write(content_str.c_str(), content_str.size());
                }
                else
                {
                    current_stream_.write(msg.mbs_content.data, msg.mbs_content.size);
                }
                current_stream_ << endl;
            }
        }

        if (is_console_satisfied)
        {
            bool is_color_console = settings_.enable_color_console &&
                ((!is_info && msg.level >= LogLevel::WeakWarning)
                    || (is_info && msg.level >= LogLevel::Info_Key));
            //cout << msg_prime_meta << msg_level_stream.str().c_str() << msg_file << endl;
            //cout << "\t";
#ifdef _WIN32
            CONSOLE_SCREEN_BUFFER_INFO default_csbi;
            HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
            GetConsoleScreenBufferInfo(hStdOut, &default_csbi);
#endif

            if (is_color_console)
            {
#ifdef _WIN32
                //set color
                if (!is_info)
                {
                    if (msg.level >= LogLevel::Error) {
                        SetConsoleTextAttribute(hStdOut, FOREGROUND_RED | FOREGROUND_INTENSITY);
                    }
                    else if (msg.level >= LogLevel::Warning) {
                        SetConsoleTextAttribute(hStdOut, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
                    }
                    else if (msg.level >= LogLevel::WeakWarning) {
                        SetConsoleTextAttribute(hStdOut, FOREGROUND_RED | FOREGROUND_GREEN);
                    }
                }
                else
                {
                    if (msg.level >= LogLevel::Info_Milestone) {
                        SetConsoleTextAttribute(hStdOut, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
                    }
                }
#else
                if (is_color_terminal_)
                {
                    if (!is_info)
                    {
                        if (msg.level >= LogLevel::Error) {
                            cout << "\x1B[1;31m";
                        }
                        else if (msg.level >= LogLevel::Warning) {
                            cout << "\x1B[1;33m";
                        }
                        else if (msg.level >= LogLevel::WeakWarning) {
                            cout << "\x1B[0;33m";
                        }
                    }
                    else
                    {
                        if (msg.level >= LogLevel::Info_Milestone) {
                            cout << "\x1B[1;32m";
                        }
                    }
                }
#endif //_WIN32
            }

            //write message
            if (has_wide_content)
            {
                string content_str = StringUtil::ToConsoleEncoding(msg.wide_content.data, msg.wide_content.size);
                cout.write(content_str.c_str(), content_str.size());
            }
            else
            {
                cout.write(msg.mbs_content.data, msg.mbs_content.size);
            }
            cout << endl;

            if (is_color_console)
            {
#ifdef _WIN32
                //set text attributes back
                SetConsoleTextAttribute(hStdOut, default_csbi.wAttributes);
#else
                if (is_color_terminal_) {
                    cout << "\x1B[0m";
                }
#endif //_WIN32
            }
        }
    }

    if ((is_info && msg.level >= LogLevel::Info_Key) || (!is_info && msg.level >= LogLevel::Warning)) {
        Flush();
    }

    ((mutex*)lock_)->unlock();
}

bool StdLogOutput::CheckCurStreamCapacity()
{
    string file_path;
    if (!current_stream_.is_open())
    {
        ConstructFilePath(file_path, max_file_idx_);
        if (!file_path.empty())
        {
            current_stream_.open(file_path.c_str(), ios::out | ios::app);
            if (!current_stream_) {
                return false;
            }
        }
    }

    if (current_stream_.tellp() >= settings_.max_size_per_file)
    {
        current_stream_.close();

        if (settings_.max_file_count > 0
            && (int)existing_files_.size() >= settings_.max_file_count)
        {
            ConstructFilePath(file_path, existing_files_[0]);
            std::remove(file_path.c_str());
            existing_files_.erase(existing_files_.begin());
        }

        max_file_idx_++;
        ConstructFilePath(file_path, max_file_idx_);
        if (!file_path.empty()) {
            current_stream_.open(file_path.c_str(), ios::out | ios::app);
        }
        existing_files_.push_back(max_file_idx_);
    }

    return current_stream_.good();
}

void StdLogOutput::ConstructFilePath(std::string &file_path, int idx)
{
    file_path.clear();
    if (!name_prefix_.empty())
    {
        char buf[1025];
        int nLen = snprintf(buf, 1024, "%s/%s_%08d.log",
            dir_.c_str(), name_prefix_.c_str(), idx);
        if (nLen > 0) {
            file_path.assign(buf);
        }
    }
}

} //end of namespace
