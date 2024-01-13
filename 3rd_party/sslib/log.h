#pragma once

#include "prime_types.h"
#include <string>
#include <vector>
#include <fstream>
#include <stdarg.h>
#include <ctime>
#include "raw_array.h"
#include "macro.h"

namespace sslib
{

//please guarantee that the contents of this enum and MsgLevelNames/MsgLevelInfoNames are consistent
struct LogLevel
{
    static const int Advice = 0x0000;
    static const int WeakWarning = 0x0100;
    static const int Warning = 0x0200;
    static const int Error = 0x0300;
    static const int Severe = 0x0400;

    static const int Info_Fussy = 0x8000;
    static const int Info_Debug = 0x8100;
    static const int Info_Ordinary = 0x8200;
    static const int Info_Status = 0x8300;
    static const int Info_Key = 0x8400;
    static const int Info_Milestone = 0x8500;

    static const int Mask = 0x7fff;
    static const int SubMask = 0x0ff;
};

struct LogMsg
{
    int level = LogLevel::Error;
    std::time_t time_stamp;
    UInt64 thread_id = 0;
    const char *file_path = nullptr, *function_name = nullptr;
    int line_in_source = -1;
    RawArray<char> mbs_content;
    RawArray<wchar_t> wide_content;
    void *data_ptr = nullptr;

    LogMsg(int lev = LogLevel::Error)
    {
        level = lev;
        file_path = nullptr;
        function_name = nullptr;
        line_in_source = -1;
        data_ptr = nullptr;
    }
};

class LogOutput
{
public:
    virtual ~LogOutput() {};
    virtual void OnNotify(LogMsg &msg) = 0;
};

class Logger
{
public:
    const static int MaxMessageSize = 8192;

    static const char *MsgLevelNames[];
    static const char *MsgLevelInfoNames[];

    static Logger log HiddenAttribute;

public:
    Logger();
    virtual ~Logger(void);

    const LogOutput* SetOutput(LogOutput *output_ptr = nullptr);
    void Notify(LogMsg &msg);

    void Notify(int msg_level, const char *format_str, ...);
    void Notify(const char *file, const char *function, int line,
        int msg_level, const char *format_str, ...);
    void Notify(const char *file, const char *function, int line,
        int msg_level, const char *format_str, va_list &args);

    void Notify(int msg_level, const wchar_t *format_str, ...);
    void Notify(const char *file, const char *function, int line,
        int msg_level, const wchar_t *format_str, ...);
    void Notify(const char *file, const char *function, int line,
        int msg_level, const wchar_t *format_str, va_list &args);

    virtual bool Flush() {
        return true;
    }

protected:
    LogOutput *output_ptr_ = nullptr;
};

class SSLogHelper
{
public:
    SSLogHelper(Logger *logger = nullptr, const char *file = nullptr,
        const char *function = nullptr, int line = -1,
        int lev = LogLevel::Info_Status);

    void operator()(const char *format_str, ...);
    void operator()(int msg_level, const char *format_str, ...);
    void operator()(int msg_level, int msg_id, const char *msg_type, const char *format_str, ...);
    void operator()(Logger *pMonitor, int msg_id, const char *msg_type, int msg_level, const char *format_str, ...);
    void operator()(const wchar_t *wcsFmt, ...);
    void operator()(int msg_level, const wchar_t *wcsFmt, ...);
    void operator()(int msg_level, int msg_id, const char *msg_type, const wchar_t *wcsFmt, ...);
    void operator()(Logger *pMonitor, int msg_id, const char *msg_type, int msg_level, const wchar_t *wcsFmt, ...);

protected:
    Logger *logger_ptr_ = nullptr;
    const char *file_name_ = nullptr;
    const char *function_ = nullptr;
    int line_ = 0;
    int level_ = 0;
};

class StdLogOutput : public LogOutput
{
public:
    enum MsgOutputMask
    {
        OUT_CONSOLE = 0x0001,   //Print messages to console
        OUT_DATA = 0x0002,	    //Save messages into inner data structures (e.g. message_str_, percent_, etc)
        OUT_CALLBACK = 0x1000   //Run a callback member function to process messages
    };

    struct OpenOptions
    {
        int lev_threshold = LogLevel::Advice;
        int info_lev_threshold = LogLevel::Info_Ordinary;
        int lev_threshold_for_console = LogLevel::WeakWarning;
        int info_lev_threshold_for_console = LogLevel::Info_Key;
        bool clear_existing = false;
        int max_size_per_file = 100 * 1024 * 1024; //100MB
        int max_file_count = 50;
        bool enable_color_console;

        OpenOptions()
        {
#       ifdef _WIN32
            enable_color_console = true;
#       else
            enable_color_console = false;
#       endif
        }

        void SetDefault();
    };

public:
    StdLogOutput(int msg_output = OUT_CONSOLE);
    virtual ~StdLogOutput();

    bool Init(const std::string &dir, const std::string &name_prefix, const OpenOptions &options);
    bool Close();

    virtual bool Flush();

    int GetOutputMask();
    void SetOutputMask(int mask);

    int GetPercent();
    void SetPercent(int percent);

    std::string RetrieveMessage(bool clear);

protected:
    std::ofstream current_stream_;
    OpenOptions settings_;
    bool is_color_terminal_ = true;

    std::string dir_;
    std::string name_prefix_;
    int max_file_idx_;
    std::vector<int> existing_files_;

    int msg_idx_;
    void *lock_;

    uint32_t msg_output_;

    int percent_;
    std::string message_str_;

protected:
    virtual void OnNotify(LogMsg &msg);

    bool CheckCurStreamCapacity();
    void ConstructFilePath(std::string &file_path, int idx);
};

// Usage:
//      SSLog(uint32_t level, const char *format_string, ...)
//      SSLogD(uint32_t level, const char *format_string, ...)

#ifndef DISABLE_LOG
#   ifndef SSLog
#       define SSLog sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__)
#   endif
#   define LogSevere sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Severe)
#   define LogError sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Error)
#   define LogWarning sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Warning)
#   define LogWeakWarning sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::WeakWarning)
#   define LogAdvice sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Advice)
#   define LogMilestone sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Info_Milestone)
#   define LogCrucialInfo sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Info_Milestone)
#   define LogKeyInfo sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Info_Key)
#   define LogStatusInfo sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Info_Status)
#   define LogOrdinaryInfo sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Info_Ordinary)
#   define LogDebugInfo sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Info_Debug)
#   define LogFussyInfo sslib::SSLogHelper(&sslib::Logger::log, __FILE__, __FUNCTION__, __LINE__, sslib::LogLevel::Info_Fussy)
#else   //ifndef DISABLE_LOG
#   define SSLog(...)
#   define LogSevere(...)
#   define LogError(...)
#   define LogWarning(...)
#   define LogWeakWarning(...)
#   define LogAdvice(...)
#   define LogMilestone(...)
#   define LogCrucialInfo(...)
#   define LogKeyInfo(...)
#   define LogStatusInfo(...)
#   define LogOrdinaryInfo(...)
#   define LogDebugInfo(...)
#   define LogFussyInfo(...)
#endif  //ifndef DISABLE_LOG

#ifdef _DEBUG
#   ifndef DISABLE_LOG
#       define LOG_DEBUG_VERSION
#   endif //ifndef DISABLE_LOG
#endif //ifdef _DEBUG

#ifdef LOG_DEBUG_VERSION
#   ifndef SSLogD
#       define SSLogD SSLog
#   endif //ifndef LogD
#   define LogSevereD LogSevere
#   define LogErrorD LogError
#   define LogWarningD LogWarning
#   define LogWeakWarningD LogWeakWarning
#   define LogAdviceD LogAdvice
#   define LogMilestoneD LogMilestone
#   define LogCrucialInfoD LogCrucialInfo
#   define LogKeyInfoD LogKeyInfo
#   define LogStatusInfoD LogStatusInfo
#   define LogOrdinaryInfoD LogOrdinaryInfo
#   define LogDebugInfoD LogDebugInfo
#   define LogFussyInfoD LogFussyInfo
#else //ifdef LOG_DEBUG_VERSION
#   ifndef SSLogD
#       define SSLogD(...)
#   endif //ifndef LogD
#   define LogSevereD(...)
#   define LogErrorD(...)
#   define LogWarningD(...)
#   define LogWeakWarningD(...)
#   define LogAdviceD(...)
#   define LogMilestoneD(...)
#   define LogCrucialInfoD(...)
#   define LogKeyInfoD(...)
#   define LogStatusInfoD(...)
#   define LogOrdinaryInfoD(...)
#   define LogDebugInfoD(...)
#   define LogFussyInfoD(...)
#   endif  //ifdef LOG_DEBUG_VERSION

} //end of namespace
