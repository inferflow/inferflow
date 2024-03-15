#include "sslib/http_server.h"
#include "sslib/thread.h"
#include "transformer/inference_engine.h"
#include "transformer/service_data.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

class InferFlowServiceCore : public sslib::Thread
{
public:
    struct Config
    {
        int http_port = 8080;
        int worker_count = 16;
        bool is_study_mode = false;

        transformer::InferenceConfig engine_config;
    };

    enum class FunctionId
    {
        ProcessQuery = 0,
        GetStat
    };

    struct QueryResult
    {
        string text;
        bool is_end = false;
    };

public:
    string Version() const;

    const Config& config() const {
        return config_;
    }

protected:
    Config config_;

    transformer::InferenceEngine engine_;
    WStrMap<FunctionId> fn_map_;

    JsonParser json_parser_;
    std::mutex json_parsr_lock_;

    map<int, QueryResult> query_to_result_;
    std::mutex result_lock_;

public:
    bool Init(const string &config_path);

    Socket::RetCode HandleRequest(BaseHttpServer::HttpResponseWriter &writer,
        const InferFlowRequest &request, bool is_openai_mode);
    bool HandleRequest(string &response, const InferFlowRequest &request, FunctionId fn, bool is_openai_mode);

    bool ParseRequest(InferFlowRequest &request, const wstring &request_str);

    FunctionId GetFunctionId(const string &url, const InferFlowRequest &request) const;

    void LogHttpHeader(const HttpRequest &request) const;

protected:
    virtual void Run() override;

    bool Infer(int max_output_len);

    bool LoadConfig(const string &config_path);

    bool HandleRequest_ProcessQuery(string &response, const InferFlowRequest &request, bool is_openai_mode);
    bool HandleRequest_GetStat(string &response, const InferFlowRequest &request);

    Socket::RetCode HandleRequest_Inner(BaseHttpServer::HttpResponseWriter *writer,
        InferFlowResponseChunk *chunk_ptr, const InferFlowRequest &request, bool is_openai_mode);

    static int GetUtf8EndPos(const string &text);
};

class InferFlowService : public BaseHttpServer
{
public:
    static const int MAX_REQUEST_LEN = 1000 * 1000;

public:
    InferFlowService() {};
    virtual ~InferFlowService() {};

    bool Init(const string &config_path);
    bool Start();

    string Version() const;

    virtual bool HandleRequest(HttpResponseWriter &writer,
        const HttpTask &task) override;

protected:
    InferFlowServiceCore core_;
};

TRANSFORMER_END
INFER_FLOW_END
