#include "sslib/memory_check.h"
#include "sslib/http_client.h"
#include "sslib/config_data.h"
#include "sslib/path.h"
#include "sslib/app_environment.h"
#include <sstream>
#include "transformer/service_data.h"

using namespace std;
using namespace sslib;
use_inferflow::transformer;

struct InferflowClientConfig
{
    string service_url;
    int timeout = 5000; //ms
    int connection_timeout = 100; //ms

    string request_str;
    InferFlowRequest request;
};

bool LoadConfig(InferflowClientConfig &config, const string &file_path)
{
    ConfigData cfg_data;
    bool ret = cfg_data.Load(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to load the configuration data"));

    string section = "main";
    ret = cfg_data.GetItem(section, "service", config.service_url, true);
    Macro_RetFalseIf(!ret);

    ret = cfg_data.GetItem(section, "request", config.request_str, true);
    Macro_RetFalseIf(!ret);

    JsonParser jparser;
    jparser.Init();

    JsonDoc jdoc;
    ret = jparser.ParseUtf8(jdoc, config.request_str);
    Macro_RetxFalseIf(!ret, LogError("The request should be in JSON format"));

    JsonObject jobj = jdoc.GetJObject();
    ret = config.request.FromJson(jobj, jdoc);
    Macro_RetxFalseIf(!ret, LogError("Invalid request format"));

    return ret;
}

bool Run(const string &config_path)
{
    InferflowClientConfig config;
    bool ret = LoadConfig(config, config_path);
    if (!ret) {
        LogError("Failed to load the configuration from %s", config_path.c_str());
        return ret;
    }

    const InferFlowRequest &request = config.request;
    LogKeyInfo(L"decoding_strategy: %ls", request.decoding_alg.c_str());
    LogKeyInfo("query_random_seed: %d", request.random_seed);
    LogKeyInfo("temperature: %.2f", request.temperature);

    HttpRequest http_request;
    http_request.type = HttpRequest::Method::Post;
    http_request.url = config.service_url;
    http_request.AddConnectionHeader(true);

    HttpClient client;
    HttpResponseReader response_reader;

    wstringstream ss;
    request.ToJson(ss);
    http_request.body = StringUtil::ToUtf8(ss.str());
    //LogKeyInfo("query: %s", http_request.body.c_str());

    auto tm1 = chrono::steady_clock::now();
    HttpRetCode ret_code = client.Execute(response_reader, http_request,
        config.timeout, config.connection_timeout);
    auto tm2 = chrono::steady_clock::now();
    float time_cost = (int)chrono::duration_cast<chrono::microseconds>(tm2 - tm1).count() / 1000.0f;

    if (ret_code != HttpRetCode::Success)
    {
        LogError("Failed to process the request (error-code: %d)", (int)ret_code);
        return false;
    }

    LogKeyInfo("Time cost: %.3f ms", time_cost);
    LogKeyInfo("## Status code: %d", response_reader.status_code());
    cout << "## Header lines:" << endl;
    for (const auto &header_line : response_reader.header_lines())
    {
        cout << header_line << endl;
    }

    JsonParser jparser;
    jparser.Init();

    string utf8_chunk_str;
    InferFlowResponseChunk chunk;
    cout << "Response text:" << endl;
    ret_code = HttpRetCode::Success;
    while (ret_code == HttpRetCode::Success)
    {
        ret_code = response_reader.NextChunk(utf8_chunk_str);
        if (ret_code != HttpRetCode::Success) {
            LogError("ret_code: %d", (int)ret_code);
            break;
        }

        if (utf8_chunk_str.empty()) {
            LogError("[empty chunk]");
            break;
        }

        wstring chunk_wstr = StringUtil::Utf8ToWideStr(utf8_chunk_str);
        ret = chunk.FromJson(chunk_wstr, jparser);
        if (!ret) {
            LogError("Failed to parse the response chunk");
            return false;
        }

        cout << StringUtil::ToConsoleEncoding(chunk.text);
        cout.flush();
        if (chunk.is_end)
        {
            cout << endl << "[end_of_response_text]" << endl;
            break;
        }
    }

    return ret;
}

int main(int argc, const char *argv[])
{
    EnableMemoryLeakReport();

    /// application environment
    string app_name = "inferflow_client";
    string app_dir = Path::GetModuleDir();
    string config_path = app_dir + app_name + ".ini";
    if (argc > 1) {
        config_path = app_dir + argv[1];
    }
    else if (!Path::FileExists(config_path.c_str())) {
        config_path = app_dir + "../" + app_name + ".ini";
    }

    string env_file = argc > 2 ? argv[2] : config_path;
    bool ret = InitAppEnv(env_file, app_name, "0.1");
    if (!ret) {
        LogError("Fail to initialize the application environment");
        return 9999;
    }

    ///
    Run(config_path);

    FinalizeAppEnv();
    return ret ? 0 : 1;
}
