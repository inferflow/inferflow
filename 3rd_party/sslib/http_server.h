#pragma once

#include <map>
#include <mutex>
#include <atomic>
#include "string.h"
#include "socket.h"
#include "thread_group.h"
#include "http_common.h"
#include "base_tcp_server.h"

namespace sslib
{

class BaseHttpServer : public BaseTcpServer
{
public:
    struct HttpTask : TcpTask
    {
        std::chrono::steady_clock::time_point star_tm;
        HttpMessageStream msg_stream;
        HttpMessageStream::HeaderInfo hdr;
        HttpMessageStream::UpdatingState state = HttpMessageStream::UpdatingState::Header;

        HttpRequest request;
    };

    class HttpResponseWriter
    {
    public:
        HttpResponseWriter(HttpTask *task) {
            task_ptr_ = task;
        }

        Socket::RetCode WriteHeader(const vector<string> &header_lines,
            const string &content_type, int status_code, bool keep_alive,
            int body_len = -1);
        Socket::RetCode WriteChunk(const string &chunk_str);
        Socket::RetCode WriteString(const string &str);

    protected:
        HttpTask *task_ptr_ = nullptr;
    };

public:
    BaseHttpServer();
    virtual ~BaseHttpServer();

    bool Init(int port, int worker_count = 2, bool is_study_mode = false);
    bool Init(int port, bool reuse_addr, int worker_count = 2, bool is_study_mode = false);
    bool Start();

    virtual bool HandleRequest(HttpResponse &response,
        const HttpRequest &request);

    virtual bool HandleRequest(HttpResponseWriter &writer,
        const HttpTask &task);

    UInt32 GetErrorCount() const;
    void AddError();

protected:
    std::atomic<int> error_count_;

    std::map<string, HttpHeaderFieldId, StrLessNoCase> header_field_map_;
    std::map<string, HttpRequest::Method, StrLessNoCase> request_type_map_;

protected:
    virtual bool PrepareTask(ConnectionData *connection,
        const char *data, uint32_t data_len) override;
    virtual bool HandleTask(TcpTask *task) override;

    void UpdateRequestFromHeaderLines(HttpRequest &request);

protected:
    BaseHttpServer(const BaseHttpServer &rhs) = delete;
    BaseHttpServer& operator = (const BaseHttpServer &rhs) = delete;
};

} //end of namespace
