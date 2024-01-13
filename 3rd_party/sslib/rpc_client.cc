#include "rpc_client.h"
#include <iostream>
#include <chrono>
#include "prime_types.h"
#include "log.h"
#include "macro.h"

namespace sslib
{

using namespace std;

Socket::RetCode RawNetMsg::Read(Socket &sock, uint32_t magic_code)
{
    const uint32_t buf_len = 4 + 8 + 4;
    char buf[buf_len];
    Socket::RetCode ret_code = sock.ReadAll(buf, buf_len);
    if (ret_code != Socket::RetCode::Success)
    {
        this->Delete();
        return ret_code;
    }

    memcpy(&magic_code, buf, 4); //4: sizeof(magic_code)
    memcpy(&this->id, buf + 4, 8); //8: sizeof(id)
    memcpy(&this->length, buf + 12, 4); //4: sizeof(length)

    if (this->length > (uint32_t)INT32_MAX)
    {
        LogError("Too large message");
        return Socket::RetCode::InvalidData;
    }

    this->data = nullptr;
    if (this->length > 0)
    {
        this->data = new char[this->length];
        ret_code = sock.ReadAll(this->data, this->length);
        if (ret_code != Socket::RetCode::Success)
        {
            this->Delete();
            return ret_code;
        }
    }

    return ret_code;
}

///////////////////////////////////////////////////////////////////////////////////////////
// class TcpClient

NaiveRpcClient::~NaiveRpcClient()
{
}

bool NaiveRpcClient::Init(const string &server_addr, int server_port,
    bool is_eager_connect, bool is_study_mode)
{
    bool ret = true;
    server_addr_ = server_addr;
    server_port_ = server_port;
    is_study_mode_ = is_study_mode;

    ret = socket_.Initialize();
    Macro_RetxFalseIf(!ret, LogError("Failed to initialize the client socket"));

    is_socket_initialized_ = true;
    if (is_eager_connect)
    {
        int timeout = 500; //ms
        is_connected_ = socket_.Connect(server_addr, server_port, timeout);
        if (!is_connected_ && is_study_mode_) {
            LogWarning("Connection failed (socket: %u)", (uint32_t)socket_.GetId());
        }
    }

    return ret;
}

RpcRetCode NaiveRpcClient::Invoke(string &response, const string &request, int timeout)
{
    RpcRetCode ret_code = RpcRetCode::Success;
    if (timeout <= 0) {
        timeout = INT32_MAX;
    }

    auto start_tm = chrono::steady_clock::now();

    //Consider the two cases:
    //  1) Server shutdown and back online again;
    //  2) Disconnected due to network problems and then back to normal
    //In either case, the RPC client cannot find the issue until a sending
    //or receiving error is obtained.
    int try_times = is_socket_initialized_ && is_connected_ ? 2 : 1;
    for (int try_idx = 0; try_idx < try_times; try_idx++)
    {
        if (!is_socket_initialized_)
        {
            is_socket_initialized_ = socket_.Initialize();
            if (!is_socket_initialized_)
            {
                LogError("Failed to initialize the client socket");
                return RpcRetCode::NotConnected;
            }
        }

        if (!is_connected_)
        {
            is_connected_ = socket_.Connect(server_addr_, server_port_, timeout);
            if (!is_connected_)
            {
                if (!is_connected_ && is_study_mode_) {
                    LogWarning("Connection failed (socket: %u)", (uint32_t)socket_.GetId());
                }
                return RpcRetCode::NotConnected;
            }
        }

        socket_.SetSendTimeout(timeout);
        ret_code = SendRequest(request, timeout);
        if (ret_code != RpcRetCode::Success)
        {
            if (is_study_mode_)
            {
                LogWarning("Failed to send request (socket: %u, error code: %u)",
                    (uint32_t)socket_.GetId(), ret_code);
            }
        }

        if (ret_code == RpcRetCode::Success)
        {
            auto now_tm = chrono::steady_clock::now();
            int elapsed_time = (int)chrono::duration_cast<chrono::milliseconds>(now_tm - start_tm).count();
            if (timeout > elapsed_time)
            {
                ret_code = ReceiveResponse(response, timeout - elapsed_time);
            }
            else
            {
                ret_code = RpcRetCode::Timeout;
            }
        }

        //it is very important to close the socket upon failure,
        //to avoid negative impact on the next Invoke() call
        if (ret_code != RpcRetCode::Success)
        {
            socket_.Shutdown();
            socket_.Close();
            is_socket_initialized_ = false;
            is_connected_ = false;
        }

        //retry only upon ever-disconnection (detected by a sending or receiving error)
        if (ret_code != RpcRetCode::NotConnected) {
            break;
        }
    }

    return ret_code;
}

RpcRetCode NaiveRpcClient::SendRequest(const string &request, int timeout)
{
    uint64_t request_id = ++next_request_id_;
    Socket::RetCode ret_code = NaiveNetMsg::Write(socket_, MagicCode, request_id, request, timeout);
    switch (ret_code)
    {
    case Socket::RetCode::Success:
        return RpcRetCode::Success;
        break;
    case Socket::RetCode::Disconnected:
        return RpcRetCode::NotConnected;
        break;
    default:
        return RpcRetCode::SendError;
        break;
    }
}

RpcRetCode NaiveRpcClient::ReceiveResponse(string &response, int timeout)
{
    (void)timeout;
    RawNetMsg net_msg;
    Socket::RetCode ret_code = net_msg.Read(socket_, MagicCode);
    if (ret_code == Socket::RetCode::Success)
    {
        response.assign(net_msg.data, net_msg.length);
    }

    net_msg.Delete();
    switch (ret_code)
    {
    case Socket::RetCode::Success:
        return RpcRetCode::Success;
        break;
    case Socket::RetCode::Disconnected:
        return RpcRetCode::NotConnected;
        break;
    default:
        return RpcRetCode::RecvError;
        break;
    }
}

} //end of namespace
