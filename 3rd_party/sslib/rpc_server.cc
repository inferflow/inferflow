#include "rpc_server.h"
#include "macro.h"
#include "chained_str_stream.h"
#include "net_msg.h"

namespace sslib
{

struct RpcTask : TcpTask
{
    //16: magic_code (4) + id (8) + length (4)
    const static uint32_t BUF_CAPACITY = 16;

    uint32_t buf_len = 0;
    char buf[BUF_CAPACITY]; //for storing message header data

    uint32_t request_length = 0;
    NaiveNetMsg request;
    NaiveNetMsg response;

    virtual ~RpcTask() {};
};

BaseRpcServer::BaseRpcServer()
{
}

BaseRpcServer::~BaseRpcServer()
{
    Clear();
}

void BaseRpcServer::Clear()
{
    BaseTcpServer::Clear();
}

bool BaseRpcServer::Init(int port, int worker_count, bool is_study_mode)
{
    bool ret = BaseTcpServer::Init(port, worker_count, is_study_mode);
    return ret;
}

bool BaseRpcServer::Start()
{
    return BaseTcpServer::Start();
}

void BaseRpcServer::Stop()
{
    return BaseTcpServer::Stop();
}

bool BaseRpcServer::HandleRequest(string &response, const string &request)
{
    response = "You said: " + request;
    return true;
}

//virtual
bool BaseRpcServer::HandleTask(TcpTask *task)
{
    RpcTask *rpc_task = (RpcTask*)task;
    if (rpc_task == nullptr) {
        return false;
    }

    rpc_task->response.id = rpc_task->request.id;
    HandleRequest(rpc_task->response.str, rpc_task->request.str);

    //send response
    Socket::RetCode ret_code = rpc_task->response.Write(*task->socket_ptr, MagicCode);
    if (ret_code != Socket::RetCode::Success) {
        rpc_task->socket_ptr->Shutdown();
    }
    return ret_code == Socket::RetCode::Success;
}

bool BaseRpcServer::PrepareTask(ConnectionData *connection,
    const char *data, uint32_t data_len)
{
    RpcTask *this_task = (RpcTask*)connection->task;
    if (this_task == nullptr)
    {
        this_task = new RpcTask;
        this_task->socket_ptr = connection->socket_ptr;
        connection->task = this_task;
    }

    uint32_t remain_len = data_len;
    if (this_task->buf_len < RpcTask::BUF_CAPACITY)
    {
        uint32_t write_len = min(remain_len, RpcTask::BUF_CAPACITY - this_task->buf_len);
        memcpy(this_task->buf + this_task->buf_len, data, write_len);
        this_task->buf_len += write_len;
        data += write_len;
        remain_len -= write_len;
    }

    if (this_task->buf_len == RpcTask::BUF_CAPACITY && this_task->request_length == 0)
    {
        uint32_t magic_code = 0;
        memcpy(&magic_code, this_task->buf, sizeof(magic_code));
        if (magic_code != MagicCode)
        {
            uint64_t socket_id = connection->socket_ptr->GetId();
            LogError("Incorrect magic code %u for connection %u", magic_code, (uint32_t)socket_id);
            return false;
        }

        memcpy(&this_task->request.id, this_task->buf + 4, sizeof(uint64_t));
        memcpy(&this_task->request_length, this_task->buf + 12, sizeof(uint32_t));
        //don't call this_task->request.str.reserve here (in case of large-length hack)
    }

    if (this_task->buf_len >= RpcTask::BUF_CAPACITY)
    {
        uint32_t len_so_far = (uint32_t)this_task->request.str.size();
        uint32_t write_len = min(remain_len, this_task->request_length - len_so_far);
        if (write_len > 0)
        {
            this_task->request.str.append(data, write_len);
            data += write_len;
            remain_len -= write_len;
            len_so_far += write_len;
        }

        if (len_so_far >= this_task->request_length)
        { //the task if ready to process
            task_queue_.Push(this_task);

            connection->task = nullptr;
            if (remain_len > 0) {
                PrepareTask(connection, data, remain_len);
            }
        }
    }

    return true;
}

} //end of namespace
