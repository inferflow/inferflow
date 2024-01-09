#pragma once

#include "socket.h"
#include "net_msg.h"
#include "number.h"
#include <queue>
#include <mutex>

namespace sslib
{

using std::string;

struct RawNetMsg
{
    uint64_t id = 0;
    uint32_t length = 0;
    char *data = nullptr;

    void Delete()
    {
        if (length > 0 && data != nullptr)
        {
            length = 0;
            delete[] data;
            data = nullptr;
        }
    }

    Socket::RetCode Read(Socket &sock, uint32_t magic_code);
};

enum class RpcRetCode
{
    Success = 0, NotConnected, Timeout, SendError, RecvError, DataWrong, OtherError
};

class NaiveRpcClient
{
public:
    NaiveRpcClient() {};
    virtual ~NaiveRpcClient();

    bool Init(const string &server_addr, int server_port,
        bool is_eager_connect = true, bool is_study_mode = false);

    RpcRetCode Invoke(string &response, const string &request,
        int timeout = UINT32_MAX);

protected:
    string server_addr_;
    int server_port_ = 0;
    bool is_study_mode_ = false;

    Socket socket_;
    bool is_socket_initialized_ = false;
    bool is_connected_ = false;
    uint64_t next_request_id_ = 0;

    static const uint32_t MagicCode = ((31415 << 16) | 27182);

protected:
    RpcRetCode SendRequest(const string &request, int timeout);
    RpcRetCode ReceiveResponse(string &response, int timeout);
};

} //end of namespace
