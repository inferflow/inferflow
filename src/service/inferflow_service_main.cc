//#include "sslib/memory_check.h" //place this before any other includes
#include "inferflow_service.h"
#include "sslib/path.h"
#include "sslib/app_environment.h"

using namespace std;
using namespace sslib;
use_inferflow::transformer;

void Run(const string &config_path)
{
    InferFlowService server;
    LogKeyInfo("Initializing the Inferflow service...");
    TaskMonitor tm;
    bool ret = server.Init(config_path);
    Macro_RetxVoidIf(!ret, LogError("Failed to initialize the Inferflow service"));

    tm.ShowElapsedTime();

    LogKeyInfo("===== Inferflow Service =====");
    string version = server.Version();
    LogKeyInfo("Version: %s", version.c_str());
    server.Start();
}

int main(int argc, const char *argv[])
{
    //EnableMemoryLeakReport();

    /// application environment
    string app_name = "inferflow_service";
    string app_dir = Path::GetModuleDir();
    string config_path = app_dir + app_name + ".ini";
    if (argc > 1) {
        config_path = app_dir + argv[1];
    }
    else if (!Path::FileExists(config_path.c_str())) {
        config_path = app_dir + "../" + app_name + ".ini";
    }

    string env_file = argc > 2 ? argv[2] : config_path;
    bool ret = InitAppEnv(env_file, app_name, "0.1.0");
    if (!ret) {
        LogError("Fail to initialize the application environment");
        return 9999;
    }

    //string data_dir = AppEnv::DataRootDir();

    ///
    Run(config_path);

    FinalizeAppEnv();
    return ret ? 0 : 1;
}
