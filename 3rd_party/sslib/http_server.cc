#include "http_server.h"
#include <iostream>
#include "prime_types.h"
#include "string.h"
#include "log.h"
#include "macro.h"
#include <chrono>
#include <sstream>
#include "thread.h"

namespace sslib
{

using namespace std;


////////////////////////////////////////////////////////////////////////////////
// class BaseHttpServer::HttpResponseWriter

Socket::RetCode BaseHttpServer::HttpResponseWriter::WriteHeader(
    const vector<string> &header_lines, const string &content_type,
    int status_code, bool keep_alive, int body_len)
{
    const char *connection_line = keep_alive ? "Connection: keep-alive\r\n"
        : "Connection: close\r\n";

    stringstream response_strm;
    string status_code_str = status_code == 200 ? "OK" : "";
    response_strm << "HTTP/1.1 " << status_code << " " << status_code_str << "\r\n";
    response_strm << "Content-Type: " << content_type << "\r\n";

    response_strm << connection_line;
    response_strm << "Access-Control-Allow-Origin: *\r\n";
    response_strm << "Access-Control-Allow-Headers: x-requested-with\r\n";
    response_strm << "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n";
    response_strm << "X-Content-Type-Options: nosniff\r\n";

    if (body_len >= 0) {
        response_strm << "Content-Length: " << body_len << "\r\n";
    }
    else {
        response_strm << "Transfer-Encoding: chunked\r\n";
    }

    for (size_t idx = 0; idx < header_lines.size(); idx++)
    {
        response_strm << header_lines[idx] << "\r\n";
    }

    response_strm << "\r\n";
    string str = response_strm.str();
    Socket::RetCode ret_code = task_ptr_->socket_ptr->WriteAll(str.c_str(), (int)str.size());
    return ret_code;
}

Socket::RetCode BaseHttpServer::HttpResponseWriter::WriteChunk(const string &chunk_str)
{
    int len = (int)chunk_str.size();
    char buf[32];
    sprintf(buf, "%X\r\n", len);
    Socket::RetCode ret_code = WriteString(buf);

    string crlf = "\r\n";
    if (ret_code == Socket::RetCode::Success) {
        ret_code = WriteString(chunk_str);
    }

    if (ret_code == Socket::RetCode::Success) {
        ret_code = WriteString(crlf);
    }

    return ret_code;
}

Socket::RetCode BaseHttpServer::HttpResponseWriter::WriteString(const string &str)
{
    return task_ptr_->socket_ptr->WriteAll(str.c_str(), (int)str.size());
}

////////////////////////////////////////////////////////////////////////////////
// class BaseHttpServer

BaseHttpServer::BaseHttpServer() : BaseTcpServer()
{
    error_count_.store(0);
    HttpUtility::InitHttpFieldMap(header_field_map_);
    request_type_map_["get"] = HttpRequest::Method::Get;
    request_type_map_["post"] = HttpRequest::Method::Post;
    request_type_map_["put"] = HttpRequest::Method::Put;
    request_type_map_["delete"] = HttpRequest::Method::Delete;
}

BaseHttpServer::~BaseHttpServer()
{
}

bool BaseHttpServer::Init(int port, int worker_count, bool is_study_mode)
{
    bool ret = Init(port, false, worker_count, is_study_mode);
    return ret;
}

bool BaseHttpServer::Init(int port, bool reuse_addr, int worker_count,
    bool is_study_mode)
{
    Socket::Spec spec;
    spec.port = port;
    spec.is_tcp = true;
    spec.is_no_delay = true;
    spec.is_blocking = true;
    spec.reuse_addr = reuse_addr;
    spec.bind_retrying_times = 10;
    spec.bind_retrying_delay = 10000;

    bool ret = BaseTcpServer::Init(spec, worker_count, is_study_mode);
    return ret;
}

bool BaseHttpServer::Start()
{
    bool ret = BaseTcpServer::Start();
    return ret;
}

bool BaseHttpServer::HandleRequest(HttpResponse &response,
    const HttpRequest &request)
{
    (void)request; (void)response;
    return false;
}

bool BaseHttpServer::HandleRequest(HttpResponseWriter &writer, const HttpTask &task)
{
    bool is_unknown_type = task.request.type == HttpRequest::Method::Unknown;

    HttpResponse response;
    if (is_unknown_type) {
        response.status_code = 400; //bad request
    }
    else {
        HandleRequest(response, task.request);
    }

    bool be_keep_alive = !is_unknown_type && !task.hdr.is_close;
    string content_type = task.request.type == HttpRequest::Method::Get
        ? "text/html" : "application/json";
    int body_len = (int)response.body.size();

    Socket::RetCode ret_code = writer.WriteHeader(response.header_lines, content_type,
        response.status_code, be_keep_alive, body_len);

    if (ret_code == Socket::RetCode::Success && !response.body.empty())
    {
        ret_code = writer.WriteString(response.body);
    }

    return ret_code == Socket::RetCode::Success;
}

UInt32 BaseHttpServer::GetErrorCount() const
{
    return (UInt32)error_count_.load();
}

void BaseHttpServer::AddError()
{
    error_count_++;
}

//virtual
bool BaseHttpServer::HandleTask(TcpTask *task)
{
    HttpTask *this_task = (HttpTask*)task;
    if (this_task == nullptr) {
        return false;
    }

    //uint64_t socket_id = this_task->socket_ptr->GetId();
    bool is_unknown_type = this_task->request.type == HttpRequest::Method::Unknown;

    HttpResponseWriter writer(this_task);
    bool is_succ = HandleRequest(writer, *this_task);

    bool be_keep_alive = !is_unknown_type && !this_task->hdr.is_close;
    bool be_keep_connection = be_keep_alive && is_succ;
    if (!be_keep_connection) {
        this_task->socket_ptr->Shutdown();
    }
    return be_keep_connection;
}

//virtual
bool BaseHttpServer::PrepareTask(ConnectionData *connection,
    const char *data, uint32_t data_len)
{
    HttpTask *task = (HttpTask*)connection->task;
    if (task == nullptr)
    {
        task = new HttpTask;
        task->socket_ptr = connection->socket_ptr;
        task->star_tm = chrono::steady_clock::now();
        task->request.remote_address = task->socket_ptr->RemoteAddress();
        connection->task = task;
    }

    task->msg_stream.str.append(data, data_len);
    if (!task->msg_stream.str.empty())
    {
        task->msg_stream.UpdateMessage(task->request, task->hdr, task->state, nullptr, header_field_map_);
        if (task->state >= HttpMessageStream::UpdatingState::Body
            && task->request.type == HttpRequest::Method::Unknown)
        {
            UpdateRequestFromHeaderLines(task->request);
        }

        if (task->state == HttpMessageStream::UpdatingState::Body)
        {
            bool to_end = task->request.body_length <= 0
                || task->request.type == HttpRequest::Method::TraceMethod;
            if (to_end) {
                task->state = HttpMessageStream::UpdatingState::End;
            }
        }

        if (task->state == HttpMessageStream::UpdatingState::End)
        { //the task if ready to process
            task_queue_.Push(task);
            connection->task = nullptr;

            //to do: consider the case of multiple http requests
            //if (xxx) {
            //    PrepareTask(connection, data, remain_len);
            //}
        }
    }

    //auto cur_tm = chrono::steady_clock::now();
    //int elapsed_time = (int)chrono::duration_cast<chrono::milliseconds>(cur_tm - task->star_tm).count();
    //if (elapsed_time >= timeout_)
    //{
    //    //to do
    //}

    return true;
}

void BaseHttpServer::UpdateRequestFromHeaderLines(HttpRequest &request)
{
    if (!request.header_lines.empty())
    {
        const auto &firstLine = request.header_lines[0];
        size_t pos = firstLine.find(' ');
        if (pos != string::npos)
        {
            string request_type_str = firstLine.substr(0, pos);
            auto iter = request_type_map_.find(request_type_str);
            request.type = iter != request_type_map_.end() ? iter->second : HttpRequest::Method::Unknown;

            size_t pos2 = firstLine.find_last_of(' ');
            if (pos2 <= pos || pos2 == string::npos) {
                pos2 = firstLine.size();
            }
            request.url = firstLine.substr(pos + 1, pos2 - pos - 1);
        }
    }
}

} //end of namespace
