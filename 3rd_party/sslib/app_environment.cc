#include "app_environment.h"
#include <iostream>
#include <cmath>
#include "path.h"
#include "config_data.h"
#include "log.h"
#ifdef _WIN32
#   include <Windows.h>
#   include <Psapi.h>
#else
#   include <unistd.h>
#   include <sys/time.h>
#   include <sys/resource.h>
#endif

using namespace std;

namespace sslib
{

AppEnv::AppEnv()
{
    enable_status_manager_ = true;
}

AppEnv::~AppEnv()
{
    Clear();
}

bool AppEnv::Init(const std::string &env_file, const std::string &app_name,
    const std::string &app_version, bool is_silent)
{
    AppInfo app_info;
    app_info.name = app_name;
    app_info.version = app_version;
    bool ret = Init(env_file, app_info, is_silent);
    return ret;
}

bool AppEnv::Init(const std::string &env_file, const AppInfo &app_info, bool is_silent)
{
    Clear();

    Path app_path;
    app_path.GetModulePath();
    app_info_.launched_dir = app_path.GetDir();
    app_info_.name = app_info.name.empty() ? app_path.GetName() : app_info.name;
    app_info_.version = app_info.version;

    string env_path = env_file.empty() ? (app_info_.launched_dir + Path::SeparatorChar + app_info_.name + ".env") : env_file;
    char firstChar = env_path.empty() ? ' ' : env_path[0];
    if(env_path.find(":") == string::npos && firstChar != '/') {
        env_path = app_info_.launched_dir + Path::SeparatorChar + env_file;
    }

    if(!Path::FileExists(env_path.c_str()))
    {
        string::size_type offset = env_file.find_last_of("\\/");
        string file_name = offset == string::npos ? env_file : env_file.substr(offset + 1);
        env_path = app_info_.launched_dir + "../" + file_name;
    }

    bool ret = LoadEnvInfo(env_path);

    if(ret && enable_logging_)
    {
        StdLogOutput *output_ptr = new StdLogOutput;
        ret = output_ptr->Init(log_dir_, log_name_, log_options_);
        if(ret) {
            log_output_ = output_ptr;
            Logger::log.SetOutput(log_output_);
        }
        else {
            delete output_ptr;
            cout << "Fail to initialize the logging system" << endl;
        }
    }

    /*if(enable_status_manager_)
    {
        ret = g_appStatusManager.Init(status_file_, enable_monitoring_, status_manager_port_);
        if(ret) {
            g_appStatusManager.SetProgramInfo("Default", app_info_.name);
        }

        if (ret)
        {
#ifdef _WIN32
            string key_free_disk_space = "FreeDiskSpace";
            ULARGE_INTEGER free_disk_space;
            string dir_name = app_info_.data_root_dir;
            bool bSuccTmp = GetDiskFreeSpaceExA(dir_name.c_str(), nullptr, nullptr, &free_disk_space) != FALSE;
            if (bSuccTmp)
            {
                float free_disk_space_mb = ((uint64_t)free_disk_space.QuadPart) / 1024.0f / 1024.0f;
                uint32_t free_disk_space_mb_int = (uint32_t)(free_disk_space_mb + 0.49);
                g_appStatusManager.SetProperty(key_free_disk_space, free_disk_space_mb_int, false, "MB");
            }
            else {
                g_appStatusManager.SetProperty(key_free_disk_space, "N/A");
            }
#endif //def _WIN32
        }
    }*/

    if(ret && !is_silent)
    {
        LogKeyInfo("Application environment is set successfullly.");
        LogKeyInfo("app_dir: %s", app_info_.launched_dir.c_str());
        LogKeyInfo("app_name: %s; Version: %s", app_info_.name.c_str(),
            app_info_.version.c_str());
        if (!app_info_.run_name.empty()) {
            LogKeyInfo("run_name: %s", app_info_.run_name.c_str());
        }
        LogKeyInfo("config_dir: %s", app_info_.config_dir.c_str());
        LogKeyInfo("data_root_dir: %s", app_info_.data_root_dir.c_str());
        LogKeyInfo("is_daemon: %s", is_daemon_ ? "true" : "false");

        string platform_name = sizeof(const char*) == 4 ? "win32" : "x64";
#       ifdef _DEBUG
        LogWarning("Configuration = debug; Platform = %s", platform_name.c_str());
#       else
        LogKeyInfo("Configuration = release; Platform = %s", platform_name.c_str());
#       endif

        LogKeyInfo("========== ========== ========== ========== ========== ==========");
    }

    return ret;
}

void AppEnv::Clear()
{
    if(log_output_ != nullptr)
    {
        Logger::log.SetOutput(nullptr);
        delete log_output_;
        log_output_ = nullptr;
    }

    ///
    app_info_.Clear();

    enable_logging_ = false;
    log_dir_.clear();
    log_name_.clear();
    log_options_.SetDefault();

    enable_status_manager_ = true;
    enable_status_reporting_ = false;

    attached_ptr_ = nullptr;
}

const AppInfo& AppEnv::GetAppInfo() const
{
    return attached_ptr_ != nullptr ? attached_ptr_->GetAppInfo() : app_info_;
}

bool AppEnv::LoadEnvInfo(const std::string &env_path)
{
    ConfigData cfg;
    bool ret = cfg.Load(env_path);
    if(!ret) {
        cout << "Fail to open application environment file " << env_path << endl;
        return false;
    }

    String item_value;
    bool is_new_format = cfg.HasSection("app_env.base");
    cfg.AddMacro("app_dir", app_info_.launched_dir);
    cfg.AddMacro("AppDir", app_info_.launched_dir);
    cfg.AddMacro("app_name", app_info_.name);
    cfg.AddMacro("AppName", app_info_.name);

    Path path;
    path.Set(env_path.c_str(), false);
    app_info_.config_dir = path.GetDir();
    cfg.AddMacro("config_dir", app_info_.config_dir);

    ///
    string section = is_new_format ? "app_env.base" : "AppEnv_Base";
    string item_name = is_new_format ? "run_name" : "RunName";
    cfg.GetItem(section, item_name, app_info_.run_name);
    cfg.AddMacro("RunName", app_info_.run_name);
    cfg.AddMacro("run_name", app_info_.run_name);

    item_name = is_new_format ? "data_root_dir" : "DataHomeDir";
    if(!cfg.GetItem(section, item_name, item_value))
    {
        //do NOT use Log, because it has not been setup yet
        cout << "Fail to get the value for item data_root_dir" << endl;
        ret = false;
    }
    Path data_root_dir; //we rely on Path to perform dir normalization
    data_root_dir.SetDir(item_value, true, true);
    app_info_.data_root_dir = data_root_dir.GetDir();
    cfg.AddMacro("DataHomeDir", app_info_.data_root_dir);
    cfg.AddMacro("data_root_dir", app_info_.data_root_dir);

    item_name = is_new_format ? "require_enter_key_to_exit" : "RequireEnterKeyToExit";
    cfg.GetItem(section, item_name, require_enter_key_to_exit_, false);
    item_name = is_new_format ? "is_daemon" : "IsDaemon";
    cfg.GetItem(section, item_name, is_daemon_, false);

    ///logging
    section = is_new_format ? "app_env.logging" : "AppEnv_Logging";
    item_name = is_new_format ? "enable_logging" : "EnableLogging";
    if(!cfg.GetItem(section, item_name, enable_logging_))
    {
        cout << "Fail to get the value for item LOGGING|EnableLogging" << endl;
        ret = false;
    }

    item_name = is_new_format ? "log_dir" : "LogDir";
    if(!cfg.GetItem(section, item_name, item_value))
    {
        cout << "Fail to get the value for item LOGGING|LogDir" << endl;
        ret = false;
    }
    log_dir_ = item_value;

    item_name = is_new_format ? "log_name" : "LogName";
    if(!cfg.GetItem(section, item_name, item_value))
    {
        cout << "Fail to get the value for item LOGGING|LogName" << endl;
        ret = false;
    }
    log_name_ = item_value;

    cfg.GetItem(section, "max_size_per_file", log_options_.max_size_per_file, false);
    cfg.GetItem(section, "max_file_count", log_options_.max_file_count, false);

    bool is_color_console = true;
    item_name = is_new_format ? "color_console" : "ColorConsole";
    if (cfg.GetItem(section, item_name, is_color_console)) {
        log_options_.enable_color_console = is_color_console;
    }

    ///
    section = is_new_format ? "app_env.status_manager" : "AppEnv_ProgramStatusManager";
    item_name = is_new_format ? "enable_monitoring" : "EnableMonitoring";
    cfg.GetItem(section, item_name, enable_monitoring_);
    item_name = is_new_format ? "status_file" : "StatusFile";
    if(cfg.GetItem(section, item_name, item_value))
    {
        status_file_ = item_value;
        Path thePath;
        thePath.Set(status_file_.c_str());
        Path::Mkdirs(thePath.GetDir().c_str());
    }
    item_name = is_new_format ? "listening_port" : "ListeningPort";
    cfg.GetItem(section, item_name, status_manager_port_);

    ///
    section = is_new_format ? "app_env.status_reporting" : "AppEnv_StatusReporting";

    return ret;
}

const AppEnv* AppEnv::Attach(AppEnv *ptr)
{
    const AppEnv *old_ptr = attached_ptr_;
    attached_ptr_ = ptr != nullptr
        ? (ptr->attached_ptr_ != nullptr ? ptr->attached_ptr_ : ptr)
        : nullptr;
    return old_ptr;
}

//static
const std::string& AppEnv::AppDir()
{
    return g_app_env.GetAppInfo().launched_dir;
}

//static
const std::string& AppEnv::AppName()
{
    return g_app_env.GetAppInfo().name;
}

//static
const std::string& AppEnv::AppVersion()
{
    return g_app_env.GetAppInfo().version;
}

//static
const std::string& AppEnv::DataRootDir()
{
    return g_app_env.GetAppInfo().data_root_dir;
}

//static
const std::string& AppEnv::ConfigDir()
{
    return g_app_env.GetAppInfo().config_dir;
}

//static
const std::string& AppEnv::RunName()
{
    return g_app_env.GetAppInfo().run_name;
}

//static
bool AppEnv::IsDaemon()
{
    return g_app_env.is_daemon_;
}

//static
std::string AppEnv::GetConfigFilePath(const std::string &file_name, const string &app_dir)
{
    Path path;
    bool ret = GetConfigFilePath(path, file_name, app_dir);
    return ret ? path.GetPath() : "";
}

//static
bool AppEnv::GetConfigFilePath(Path &path, const std::string &file_name, const string &app_dir)
{
    string base_dir = !app_dir.empty() ? app_dir : AppDir();

    string config_path = base_dir + "/" + file_name;
    bool bExist = Path::FileExists(config_path.c_str());
    if(!bExist) {
        config_path = base_dir + "/../" + file_name;
        bExist = Path::FileExists(config_path.c_str());
    }
    if(!bExist) {
        config_path = base_dir + "/../ini/" + file_name;
        bExist = Path::FileExists(config_path.c_str());
    }

    path.Set(config_path.c_str());
    return bExist;
}

//static
ProcessMemoryStat AppEnv::MemoryUsage()
{
    ProcessMemoryStat mstat;
    MemoryUsage(mstat);
    return mstat;
}

//static
void AppEnv::MemoryUsage(ProcessMemoryStat &mstat)
{
    mstat.Clear();

#ifdef _WIN32
    const float ratio = 1024.0f * 1024.0f;
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    mstat.rss = pmc.WorkingSetSize / ratio;
    mstat.rss_peak = pmc.PeakWorkingSetSize / ratio;
    //to do: get actuall vsz (now simply set vsz = rss, peak_vsz = peak_rss)
    mstat.vsz = mstat.rss;
    mstat.vsz_peak = mstat.rss_peak;
#else //_WIN32
    const float ratio = 1024.0f;
    map<string, int> attr_map = { {"VmSize", 1}, {"VmPeak", 2}, {"VmRSS", 3}, {"VmHWM", 4}};
    ifstream reader("/proc/self/status", ios_base::in);
    string line_str;
    while (getline(reader, line_str))
    {
        if (line_str.size() > 2 && line_str[0] == L'V' && line_str[1] == L'm')
        {
            auto pos = line_str.find(":");
            if (pos == string::npos) {
                continue;
            }

            string attr_name = line_str.substr(0, pos);
            auto attr_iter = attr_map.find(attr_name);
            if (attr_iter == attr_map.end()) {
                continue;
            }

            int start_value = pos + 1;
            while (start_value < (int)line_str.size())
            {
                if (line_str[start_value] != ' ' && line_str[start_value] != '\t') {
                    break;
                }
                start_value++;
            }
            auto value_end = line_str.find(' ', start_value);
            if (value_end == string::npos) {
                value_end = line_str.size();
            }
            string value_str = line_str.substr(start_value, value_end - start_value);

            switch(attr_iter->second)
            {
            case 1: mstat.vsz = String::ToInt32(value_str) / ratio; break;
            case 2: mstat.vsz_peak = String::ToInt32(value_str) / ratio; break;
            case 3: mstat.rss = String::ToInt32(value_str) / ratio; break;
            case 4: mstat.rss_peak = String::ToInt32(value_str) / ratio; break;
            default: break;
            }
        }
    }
    //rusage u;
    //int ret = getrusage(RUSAGE_SELF, &u);
    //return ret == 0 ? (UInt32)u.ru_maxrss : 0;
#endif //_WIN32
}

//static
void AppEnv::LogProcessMemoryUsage(InformationUnit unit)
{
    string unit_name = "MB";
    float ratio = 1.0f;
    switch (unit)
    {
    case InformationUnit::Byte:
        unit_name = "bytes";
        ratio = 1 / pow(1024.0f, 2);
        break;
    case InformationUnit::KB:
        unit_name = "KB";
        ratio = 1 / 1024.0f;
        break;
    case InformationUnit::MB:
        unit_name = "MB";
        ratio = 1.0f;
        break;
    case InformationUnit::GB:
        unit_name = "GB";
        ratio = 1024.0f;
        break;
    case InformationUnit::TB:
        unit_name = "TB";
        ratio = pow(1024.0f, 2);
        break;
    default:
        break;
    }

    ProcessMemoryStat mstat = MemoryUsage();
    LogKeyInfo("Memory usage (%s): %.2f, %.2f (Peak)",
        unit_name.c_str(),
        mstat.rss / ratio,
        mstat.rss_peak / ratio);
/*
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    {
        LogKeyInfo("Memory counters (%s): %.2f, %.2f (Peak), %.2f, %.2f (Peak)",
            unit_name.c_str(),
            pmc.WorkingSetSize / ratio,
            pmc.PeakWorkingSetSize / ratio,
            pmc.PagefileUsage / ratio,
            pmc.PeakPagefileUsage / ratio);
    }
#else
    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat", ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string num_threads, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
        >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
        >> utime >> stime >> cutime >> cstime >> priority >> nice
        >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    float resident_set = rss * page_size_kb;

    //vsz: virtual memory size; rss: resident set size
    LogKeyInfo("Memory counters (%s): %.2f (vsz), %.2f (rss)",
        unit_name.c_str(), vsize / ratio, resident_set * 1024.0f / ratio);
#endif*/
}

//////////////////////////////////////////////////////////////////////////////////////
//

AppEnv g_app_env;

bool InitAppEnv(const std::string &env_file, const std::string &app_name,
    const std::string &app_version, bool is_silent)
{
    if (!is_silent) {
        AppEnv::LogProcessMemoryUsage(InformationUnit::MB);
    }
    return g_app_env.Init(env_file, app_name, app_version, is_silent);
}

bool InitAppEnv(const std::string &env_file, const AppInfo &app_info, bool is_silent)
{
    return g_app_env.Init(env_file, app_info, is_silent);
}

void FinalizeAppEnv()
{
    AppEnv::LogProcessMemoryUsage(InformationUnit::MB);
    if(g_app_env.require_enter_key_to_exit())
    {
        cout << "Press the enter key to quit..."<< endl;
        string strLine;
	    getline(cin, strLine);
    }

    g_app_env.Clear();
}

} //end of namespace
