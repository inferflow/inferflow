#pragma once

#include <string>
#include "log.h"
#include "path.h"
#include "measurement_unit.h"
#include "macro.h"

namespace sslib
{

using std::string;

struct AppInfo
{
    std::string launched_dir;
    std::string name;
    std::string version;
    std::string config_dir;
    std::string data_root_dir;
    std::string run_name;

    AppInfo() {};
    void Clear()
    {
        launched_dir.clear();
        name.clear();
        version.clear();
        config_dir.clear();
        data_root_dir.clear();
        run_name.clear();
    }
};

struct ProcessMemoryStat
{
    float vsz = 0; //virtual memory size, in MB
    float vsz_peak = 0; //peak virtual memory size, in MB
    float rss = 0; //resident set size, in MB
    float rss_peak = 0; //peak resident set size, in MB

    void Clear()
    {
        vsz = 0;
        vsz_peak = 0;
        rss = 0;
        rss_peak = 0;
    }
};

class AppEnv
{
public:
    AppEnv();
    virtual ~AppEnv();

    bool Init(const std::string &env_file = "", const std::string &app_name = "",
        const std::string &app_version = "", bool is_silent = false);
    bool Init(const std::string &env_file, const AppInfo &app_info, bool is_silent = false);
    void Clear();

    const AppInfo& GetAppInfo() const;
    const AppEnv* Attach(AppEnv *ptr = nullptr);
    bool require_enter_key_to_exit() const {
        return require_enter_key_to_exit_;
    }

    static const std::string& AppDir();
    static const std::string& AppName();
    static const std::string& AppVersion();
    static const std::string& DataRootDir();
    static const std::string& ConfigDir();
    static const std::string& RunName();
    static bool IsDaemon();

    const std::string& status_file_path() const {
        return status_file_;
    }

    static std::string GetConfigFilePath(const std::string &file_name,
        const std::string &app_dir = "");
    static bool GetConfigFilePath(File &path, const std::string &file_name,
        const std::string &app_dir = "");

    //return the memory usage of the current process, in megabyte (MB)
    static ProcessMemoryStat MemoryUsage();
    static void MemoryUsage(ProcessMemoryStat &mstat);
    static void LogProcessMemoryUsage(InformationUnit unit = InformationUnit::GB);

protected:
    AppInfo app_info_;
    bool is_daemon_ = false;
    bool require_enter_key_to_exit_ = false;

    bool enable_logging_ = false;
    std::string log_dir_;
    std::string log_name_;
    StdLogOutput::OpenOptions log_options_;

    bool enable_monitoring_ = false, enable_status_manager_ = false;
    std::string status_file_;
    uint32_t status_manager_port_ = 0;

    bool enable_status_reporting_ = false;

    ///
    LogOutput *log_output_ = nullptr;

    ///
    AppEnv *attached_ptr_ = nullptr;

protected:
    bool LoadEnvInfo(const std::string &env_path);
};

typedef AppEnv AppEnvironment;

extern AppEnv g_app_env HiddenAttribute;
extern bool InitAppEnv(const string &env_file = "", const string &app_name = "",
    const string &app_version = "", bool is_silent = false);
extern bool InitAppEnv(const string &env_file, const AppInfo &app_info,
    bool is_silent = false);
extern void FinalizeAppEnv();

} //end of namespace
