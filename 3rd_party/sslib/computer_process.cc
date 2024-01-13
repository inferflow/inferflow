#include "computer_process.h"
#include <fstream>
#include "string_util.h"
#include "path.h"
#include "numerical_util.h"
#ifdef _WIN32
#   include <Windows.h>
#   include <shellapi.h>
#   include <tlhelp32.h>
#else
#   include <signal.h>
#endif

namespace sslib
{

using namespace std;

//static
bool ComputerProcess::StartProcess(const wstring &cmdline)
{
    //std::system(cmdline.c_str());
    //return true;
#ifdef _WIN32
    HINSTANCE ret_code = ShellExecute(nullptr, nullptr, cmdline.c_str(),
        nullptr, nullptr, SW_SHOWNORMAL);
    return ret_code > (HINSTANCE)32;
#else
    int ret_code = std::system(StringUtil::ToUtf8(cmdline).c_str());
    (void)ret_code;
    //use popen() or system();
    return true;
#endif
}

//static
bool ComputerProcess::KillProcess(const string &dir, const string &proc_name)
{
    wstring process_dir = StringUtil::Utf8ToWideStr(dir);
    wstring process_name = StringUtil::Utf8ToWideStr(proc_name);
    return KillProcess(process_dir, process_name);
}

//static
bool ComputerProcess::KillProcess(const wstring &dir, const wstring &proc_name)
{
    ProcessId pid = GetPidByName(dir, proc_name);
    if (pid == ProcessId_Invalid) {
        return false;
    }

    //LogKeyInfo("Will kill process %u: %s", pid, StringUtil::ToUtf8(process_name).c_str());
    return KillProcess(pid);
}

//static
bool ComputerProcess::KillProcess(ProcessId pid)
{
#ifdef _WIN32
    bool ret = KillProcess_Win(pid);
#else
    bool ret = KillProcess_Linux(pid);
#endif
    return ret;
}

//static
bool ComputerProcess::ListProcesses(vector<ProcessData> &process_list)
{
#ifdef _WIN32
    return ListProcesses_Win(process_list);
#else
    return ListProcesses_Linux(process_list);
#endif
}

//static
ProcessId ComputerProcess::GetPidByName(const wstring &dir, const wstring &name,
    bool is_case_sensitive, bool is_exact_match)
{
    (void)dir;  (void)is_case_sensitive; (void)is_exact_match;
    vector<ProcessData> process_list;
    ListProcesses(process_list);

    wstring process_name;
    for (const ProcessData &proc_data : process_list)
    {
        auto offset = proc_data.cmdline.find(L'\0');
        wstring bin_path = offset == wstring::npos ? proc_data.cmdline : proc_data.cmdline.substr(0, offset);

        offset = bin_path.find_last_of(L"\\/");
        process_name = offset != wstring::npos ? bin_path.substr(offset + 1) : bin_path;

        //LogKeyInfo("cmdline: %s, bin path: %s, process name: %s",
        //    StringUtil::ToUtf8(proc_data.cmdline).c_str(),
        //    StringUtil::ToUtf8(bin_path).c_str(),
        //    StringUtil::ToUtf8(process_name).c_str());

        if (wcscasecmp(process_name.c_str(), name.c_str()) == 0)
        {
            return proc_data.id;
        }
    }

    return ProcessId_Invalid;
}

#ifdef _WIN32
//static
bool ComputerProcess::ListProcesses_Win(vector<ProcessData> &process_list)
{
    // Take a snapshot of all processes in the system.
    HANDLE process_snap_handle = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (process_snap_handle == INVALID_HANDLE_VALUE) {
        return false;
    }

    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);
    if (!Process32First(process_snap_handle, &pe32))
    {
        CloseHandle(process_snap_handle);
        return false;
    }

    do
    {
        ProcessData proc_data;
        proc_data.cmdline = pe32.szExeFile;
        proc_data.id = (ProcessId)pe32.th32ProcessID;
        proc_data.thread_count = (UInt32)pe32.cntThreads;
        proc_data.parent_id = (ProcessId)pe32.th32ParentProcessID;
        process_list.push_back(proc_data);
    } while (Process32Next(process_snap_handle, &pe32));

    CloseHandle(process_snap_handle);
    return true;
}

//static
bool ComputerProcess::KillProcess_Win(ProcessId pid)
{
    DWORD desired_access = PROCESS_TERMINATE;
    BOOL is_inherit_handle = FALSE;
    HANDLE process_handle = OpenProcess(desired_access, is_inherit_handle, (DWORD)pid);
    if (process_handle == nullptr) {
        return false;
    }

    UINT exit_code = 9999;
    BOOL result = TerminateProcess(process_handle, exit_code);
    CloseHandle(process_handle);

    return result != FALSE;
}
#endif //def _WIN32

#ifndef _WIN32
//static
bool ComputerProcess::ListProcesses_Linux(vector<ProcessData> &process_list)
{
    File::ListDirOptions list_dir_opt;
    list_dir_opt.including_dirs = true;
    list_dir_opt.is_recursive = false;
    list_dir_opt.res_with_full_path = false;
    list_dir_opt.res_with_prefix_dir = false;

    const string proc_dir = "/proc/";
    vector<string> process_dir_list;
    File::ListDir(process_dir_list, proc_dir, &list_dir_opt);

    string cmdline_file, cmdline_str;
    for (const string &pid_str : process_dir_list)
    {
        if (!NumericalUtil::BeValidInteger(pid_str)) {
            continue;
        }

        cmdline_file = proc_dir + pid_str + "/cmdline";
        ifstream reader(cmdline_file);
        if (reader.good() && getline(reader, cmdline_str))
        {
            ProcessData proc_data;
            proc_data.id = (ProcessId)String::ToInt32(pid_str);
            proc_data.cmdline = StringUtil::Utf8ToWideStr(cmdline_str);
            process_list.push_back(proc_data);
        }
        reader.close();
    }

    return true;
}

//static
bool ComputerProcess::KillProcess_Linux(ProcessId pid)
{
    int sig = 9; //SIGKILL
    int ret = kill(pid, sig);
    return ret == 0;
}
#endif

} //end of namespace
