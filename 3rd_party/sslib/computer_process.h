#pragma once

#include "prime_types.h"
#include <string>
#include <vector>

namespace sslib
{

using std::string;
using std::wstring;
using std::vector;

typedef UInt32 ProcessId;
#define ProcessId_Invalid (UInt32)(-1)

struct ProcessData
{
    ProcessId id = ProcessId_Invalid;
    ProcessId parent_id = ProcessId_Invalid;
    wstring cmdline;
    UInt32 thread_count = 0;
};

class ComputerProcess
{
public:
    //start a process
    static bool StartProcess(const wstring &cmdline);
    //kill a process
    static bool KillProcess(const string &dir, const string &proc_name);
    static bool KillProcess(const wstring &dir, const wstring &proc_name);
    static bool KillProcess(ProcessId pid);

    static ProcessId GetPidByName(const wstring &dir, const wstring &name,
        bool is_case_sensitive = false, bool is_exact_match = true);
    static bool ListProcesses(vector<ProcessData> &process_list);

protected:
#ifdef _WIN32
    static bool ListProcesses_Win(vector<ProcessData> &process_list);
    static bool KillProcess_Win(ProcessId pid);
#else
    static bool ListProcesses_Linux(vector<ProcessData> &process_list);
    static bool KillProcess_Linux(ProcessId pid);
#endif
};

} //end of namespace
