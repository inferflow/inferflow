#pragma once

#include <string>
#include <vector>
#include "prime_types.h"
#ifdef _WIN32
//#   include <winsock2.h>
#else
#   include <sys/types.h>   // for socket
#   include <sys/socket.h>  // for socket
#   include <netdb.h>
#   include <arpa/inet.h>
//#   include <netinet/in.h>  // for sockaddr_in
#endif

namespace sslib
{

using std::string;
using std::vector;

struct SocketOptions
{
public:
    struct KeepAliveInfo
    {
        bool is_on;
        uint32_t time_val, interval;

        KeepAliveInfo(bool p_is_on = true, uint32_t p_time = 0x6DDD00, uint32_t p_interval = 10)
        {
            is_on = p_is_on;
            time_val = p_time;
            interval = p_interval;
        }
    };

public:
    KeepAliveInfo keep_alive_info;

    SocketOptions()
    {
    }
};

class Socket  
{
public:
    struct Spec
    {
        int port = 0;
        bool is_tcp = true;
        bool is_no_delay = true;
        bool is_blocking = true;
        bool reuse_addr = false;
        int bind_retrying_times = 0;
        int bind_retrying_delay = 10 * 1000;
    };

    enum class RetCode
    {
        Success = 0, Disconnected, Timeout, Nonblocking, InvalidData, OtherError
    };

protected:
#ifdef _WIN32
    typedef uint64_t SocketId;
    const static uint64_t SocketId_Invalid = (uint64_t)(~0);
#else
    typedef int SocketId;
    const static int SocketId_Invalid = -1;
#endif

    SocketId id_;
    std::string local_address_, remote_address_;

    //10 means the most detailed description
    int verbosity_;
    bool is_blocking_ = true;

public:
    Socket(int verbosity = 5);
    virtual ~Socket();

    bool Initialize(int port = 0, bool is_tcp = true, bool is_no_delay = true,
        int bind_retrying_times = 0, int bind_retrying_delay = 10 * 1000);
    bool Initialize(const Spec &spec);
    void Close();
    void Shutdown();

    uint64_t GetId() const {
        return (uint64_t)id_;
    }

    bool Listen();
    bool Accept(Socket& accept_socket);
    Socket* Accept();

    bool Connect(const std::string &addr, int port, int timeout_ms = -1);

    int Send(const char *buffer, int length, int flags = 0);
    RetCode SendEx(int &bytes_sent, const char *buf, int len, int flags);
    int Recv(char *buffer, int length, int flags = 0);
    RetCode RecvEx(int &bytes_read, char *buf, int len, int flags = 0);
    int Read(void *buffer, int nCount) {
        return this->Recv((char*)buffer, nCount, 0);
    }
    int Write(const void *buffer, int nCount) {
        return this->Send((const char*)buffer, nCount, 0);
    }

    //return the amount of data pending in the network's input buffer that can be read from socket
    int DataSizeAvailableForRead();

    //These two functions are to guarantee reading/writing enough characters from/into the socket
    //Note read or write may return fewer characters than requested for some reason.
    RetCode ReadAll(char *buf, int len);
    int Readn(char *buffer, int size);
    RetCode WriteAll(const char *buf, int len);
    int Writen(const char *buffer, int size, int timeout = INT32_MAX);
    int WritenForNonblocking(const char *buf, int len, int timeout = INT32_MAX);

    // read/write simple data types
    int ReadInt8(int8_t &val);
    int ReadInt16(int16_t &val);
    int ReadInt32(int32_t &val);
    int ReadInt64(int64_t &val);
    int ReadUInt8(uint8_t &val);
    int ReadUInt16(uint16_t &val);
    int ReadUInt32(uint32_t &val);
    int ReadUInt64(uint64_t &val);
    int WriteInt8(int8_t val);
    int WriteInt16(int16_t val);
    int WriteInt32(int32_t val);
    int WriteInt64(int64_t val);
    int WriteUInt8(uint8_t val);
    int WriteUInt16(uint16_t val);
    int WriteUInt32(uint32_t val);
    int WriteUInt64(uint64_t val);

    //
    int ReadString8(std::string& val);
    int WriteString8(const std::string& val);
    int ReadString16(std::string& val);
    int WriteString16(const std::string& val);
    int ReadString32(std::string& val);
    int WriteString32(const std::string& val);

    int SetVerbosity(int verbosity);
    bool SetOptions(const SocketOptions &opt);
    int SetOption(int level, int optname, const char* optval, int optlen);
    bool SetOptNoDelay(bool value);
    bool SetSendTimeout(int timeout_ms); //in milliseconds
    bool SetRecvTimeout(int timeout_ms); //in milliseconds
    bool SetKeepAliveInfo(bool keep_alive, uint32_t time = 0x6DDD00, uint32_t interval = 10);

    std::string LocalAddress() const {
        return local_address_;
    }
    std::string RemoteAddress() const {
        return remote_address_;
    }

    bool SetBlocking(bool is_blocking);
    bool is_blocking() const {
        return is_blocking_;
    }

    static bool IsIPAddress(const std::string &str);
    //static bool GetAddressInfo(vector<const addrinfo*> &addr_list,
    //    const string &node_name, int port);
    //static string GetIpAddress(const addrinfo &addr_info);

protected:
    static bool IsValidSocketId(SocketId sid);

    bool ConnectEx(const std::string &addr, int port, int timeout = 100);
};

} //end of namespace
