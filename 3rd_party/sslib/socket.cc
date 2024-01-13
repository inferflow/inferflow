#include "socket.h"
#include "log.h"
#include "thread.h"
#include <cstring>
#ifdef _WIN32
#   include <winsock2.h>
#   include <mstcpip.h>
#   include <ws2tcpip.h>
#elif defined __APPLE__ || defined Macintosh
#   include <stdio.h>
#   include <stdlib.h>
#   include <unistd.h>
#   include <sys/types.h>
#   include <sys/socket.h>
#   include <netinet/in.h>
#   include <netinet/tcp.h>
#   include <sys/ioctl.h>
#   include <poll.h>
#   ifndef MSG_NOSIGNAL
#       define MSG_NOSIGNAL 0x2000
#   endif
#else
#   include <arpa/inet.h>
#   include <netdb.h>
#   include <unistd.h>
#   include <sys/time.h>
#   include <sys/ioctl.h>
#   include <netinet/tcp.h>
#   include <poll.h>
#endif

using namespace std;

namespace sslib
{

#ifdef _WIN32
class NetEnvironment
{
public:
	NetEnvironment()
	{
		StaticInit();
	};

    virtual ~NetEnvironment()
    {
        WSACleanup();
    }

protected:
	static bool StaticInit()
	{
		WSADATA wsa_data;
		WORD version = MAKEWORD(2,0);
		int ret = WSAStartup(version, &wsa_data);
		if (ret != 0)
		{
            LogError("Can not initialize Ws2_32.lib");
			return false;
		}
		return true;
	};
};
static NetEnvironment g_net_environment;
#endif //def _WIN32

bool GetAddressInfo(vector<const addrinfo*> &addr_list,
    const string &node_name, int port)
{
    addr_list.clear();

    addrinfo hints;
    memset(&hints, 0, sizeof(addrinfo));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = node_name.empty() ? AI_PASSIVE : 0;
    hints.ai_protocol = 0; //other options: IPPROTO_TCP, SOCK_DGRAM

    addrinfo *addr_res = nullptr;
    string service_name = to_string(port);
    int result = getaddrinfo(
        node_name.empty() ? nullptr : node_name.c_str(),
        port == 0 ? nullptr : service_name.c_str(),
        &hints, &addr_res);
    if (result != 0) {
        //LogError("Failed to call getaddrinfo");
        return false;
    }

    for (const auto *addr_ptr = addr_res; addr_ptr != nullptr; addr_ptr = addr_ptr->ai_next)
    {
        addr_list.push_back(addr_ptr);
    }

    return true;
}

string GetIpAddress(const addrinfo &addr_info)
{
    string ip_address;

    const sockaddr_in *ipv4_addr = nullptr;
    const sockaddr_in6 *ipv6_addr = nullptr;
    char addr_str[INET6_ADDRSTRLEN];
    switch (addr_info.ai_family)
    {
    case AF_INET:
        ipv4_addr = (sockaddr_in*)addr_info.ai_addr;
        inet_ntop(AF_INET, &ipv4_addr->sin_addr, addr_str, INET6_ADDRSTRLEN);
        ip_address = addr_str;
        break;
    case AF_INET6:
        ipv6_addr = (sockaddr_in6*)addr_info.ai_addr;
        inet_ntop(AF_INET6, &ipv6_addr->sin6_addr, addr_str, INET6_ADDRSTRLEN);
        ip_address = addr_str;
        break;
    default:
        break;
    }

    return ip_address;
}

////////////////////////////////////////////////////////////////////////////////
// Socket

Socket::Socket(int verbosity)
{
#ifdef _WIN32
	id_ = INVALID_SOCKET;
#else
    id_ = -1;
#endif
	verbosity_ = verbosity;
}

Socket::~Socket()
{
	Close();
}

bool Socket::Initialize(int port, bool is_tcp, bool is_no_delay,
    int bind_retrying_times, int bind_retrying_delay)
{
    Spec spec;
    spec.port = port;
    spec.is_tcp = is_tcp;
    spec.is_no_delay = is_no_delay;
    spec.is_blocking = true;
    spec.bind_retrying_times = bind_retrying_times;
    spec.bind_retrying_delay = bind_retrying_delay;
    return Initialize(spec);
}

bool Socket::Initialize(const Spec &spec)
{
    /// 1. create socket handle
#ifdef _WIN32
    //The socket that is created will have the overlapped attribute as a default.
    //SO_OPENTYPE (defined in Mswsock.h) can affect this default.
    //See Microsoft-specific documentation for a detailed description of SO_OPENTYPE.
    id_ = socket(AF_INET, spec.is_tcp ? SOCK_STREAM : SOCK_DGRAM, 0);
#else
    id_ = socket(PF_INET, spec.is_tcp ? SOCK_STREAM : SOCK_DGRAM, 0);
#endif

    if (!IsValidSocketId(id_))
    {
        if (verbosity_ > 3) {
            LogError("Failed to create socket");
        }
        return false;
    }

    SetOptNoDelay(spec.is_no_delay); //socket options have to be called before binding

    if (spec.reuse_addr)
    {
        int reuse_addr = 1;
        //LogKeyInfo("Set SO_REUSEADDR as 1");
        if (SetOption(SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse_addr, sizeof(int)) != 0)
        {
            LogWarning("setsockopt(SO_REUSEADDR) failed");
        }
    }

    if (!spec.is_blocking) {
        SetBlocking(false);
    }

    /// 2. bind
    if (!spec.is_tcp || spec.port > 0)
    {
        struct sockaddr_in server;
        memset(&server, 0, sizeof(server));
        server.sin_family = AF_INET;
        server.sin_addr.s_addr = htonl(INADDR_ANY);
        server.sin_port = htons((u_short)spec.port);

        bool is_bind_succ = ::bind(id_, (struct sockaddr*)&server, sizeof(server)) >= 0;
        for (int idx = 0; !is_bind_succ && idx < spec.bind_retrying_times; idx++)
        {
            LogKeyInfo("Binding error in port %d, will retry (%d/%d) in %d ms...",
                spec.port, idx + 1, spec.bind_retrying_times, spec.bind_retrying_delay);
            Thread::SleepMilli(spec.bind_retrying_delay);

            is_bind_succ = ::bind(id_, (struct sockaddr*)&server, sizeof(server)) >= 0;
            if (is_bind_succ) {
                LogKeyInfo("Binded successfully.");
            }
        }

        if (!is_bind_succ)
        {
            if (verbosity_ > 4) {
                LogError("Binding error in port %d", spec.port);
            }
            Close();
            return false;
        }
    }

    return true;
}

bool Socket::Listen()
{
	int ret_code = ::listen(id_, SOMAXCONN);
	return ret_code == 0;
}

bool Socket::Accept(Socket &accept_socket)
{
    struct sockaddr_in client;
    SocketId msg_socket;
#ifdef _WIN32
    int sock_len = sizeof(client);
#else
    socklen_t sock_len = (socklen_t)sizeof(client);
#endif
    msg_socket = ::accept(id_, (struct sockaddr*)&client, &sock_len);

    if (!IsValidSocketId(msg_socket))
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Stopped to accept connections");
        }
        return false;
    }

    if (verbosity_ > 6)
    {
        LogStatusInfo("Accepting stream connection from %s:%d",
            inet_ntoa(client.sin_addr),
            client.sin_port);
    }

    accept_socket.id_ = msg_socket;
    accept_socket.remote_address_ = inet_ntoa(client.sin_addr);
    accept_socket.remote_address_ += ":";
    accept_socket.remote_address_ += std::to_string(client.sin_port);

    return true;
}

Socket* Socket::Accept()
{
    Socket *socket_ptr = new Socket();
    bool ret_code = Accept(*socket_ptr);
    if (!ret_code)
    {
        delete socket_ptr;
        return nullptr;
    }
    else return socket_ptr;
}

//issue with this implementation: Support at most 1021 connections
//(maybe due to the calling of select)
bool Socket::ConnectEx(const std::string &addr, int port, int timeout)
{
    string ip_addr;
    vector<const addrinfo*> addr_list;
    GetAddressInfo(addr_list, addr, port);
    for (const addrinfo *addr_info : addr_list)
    {
        if (addr_info != nullptr && addr_info->ai_family == AF_INET)
        {
            ip_addr = GetIpAddress(*addr_info);
            break;
        }
    }

    if (ip_addr.empty())
    {
        //LogError("Cannot get the IP address of %s", addr.c_str());
        return false;
    }

    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr(ip_addr.c_str());
    server.sin_port = htons((u_short)port);

    //set the socket in non-blocking
    SetBlocking(false);

    /// 2. connect
    int ret_code = ::connect(id_, (struct sockaddr *)&server, sizeof(server));
    //if(ret_code != 0 && ret_code != EINPROGRESS)
    //{
    //    if(verbosity_ > 4) {
    //        LogWarning("Failed to connect to %s:%d", inet_ntoa(server.sin_addr), nPort);
    //    }
    //    return false;
    //}

    // restart the socket mode
    SetBlocking(true);

#ifdef _WIN32
    fd_set rfds, wfds, efds;
    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    FD_ZERO(&efds);
    FD_SET(id_, &rfds);
    FD_SET(id_, &wfds);
    FD_SET(id_, &efds);

    timeval the_timeout;
    the_timeout.tv_sec = timeout / 1000;
    the_timeout.tv_usec = (timeout % 1000) * 1000;

    // check if the socket is ready
    ret_code = select((int)id_ + 1, &rfds, &wfds, &efds, &the_timeout);
    if (ret_code <= 0) {
        return false;
    }

#ifndef _WIN32
    int retW = FD_ISSET(id_, &wfds);
    int retR = FD_ISSET(id_, &rfds);
    if (!(retW == 1 && retR == 0))
    {
        Shutdown();
        return false;
    }
#endif
#else
    struct pollfd pfd;
    pfd.fd = id_;
    pfd.events = POLLOUT;
    //struct pollfd pfds[] = { {.fd = id_, .events = POLLOUT } };
    ret_code = poll(&pfd, 1, timeout);
    if (ret_code <= 0) {
        return false;
    }
#endif

    if (verbosity_ > 6) {
        LogStatusInfo("connected to %s:%d OK", inet_ntoa(server.sin_addr), port);
    }

    return true;
}

bool Socket::Connect(const std::string &addr, int port, int timeout_ms)
{
    if (timeout_ms >= 0) {
        return ConnectEx(addr, port, timeout_ms);
    }

    string ip_addr;
    vector<const addrinfo*> addr_list;
    GetAddressInfo(addr_list, addr, port);
    for (const addrinfo *addr_info : addr_list)
    {
        if (addr_info != nullptr && addr_info->ai_family == AF_INET)
        {
            ip_addr = GetIpAddress(*addr_info);
            break;
        }
    }

    if (ip_addr.empty())
    {
        //LogError("Cannot get the IP address of %s", addr.c_str());
        return false;
    }

    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr(ip_addr.c_str());
    server.sin_port = htons((u_short)port);

    //if (timeout_ms >= 0)
    //{
    //    SetSendTimeout(timeout_ms);
    //    SetRecvTimeout(timeout_ms);
    //}

    /// 2. connect
    int ret_code = ::connect(id_, (struct sockaddr *)&server, sizeof(server));

    //if (timeout_ms >= 0)
    //{
    //    SetSendTimeout(INT32_MAX);
    //    SetRecvTimeout(INT32_MAX);
    //}
    return ret_code == 0;
}

void Socket::Close()
{
    if (IsValidSocketId(id_))
    {
#ifdef _WIN32
        ::closesocket(id_);
#else
        ::close(id_);
#endif
        id_ = SocketId_Invalid;
    }
}

void Socket::Shutdown()
{
    if (IsValidSocketId(id_))
    {
        shutdown(id_, 2); //2: SD_BOTH
    }
}

//return the amount of data pending in the network's input buffer that can be read from socket
int Socket::DataSizeAvailableForRead()
{
    u_long num = 0;
#ifdef _WIN32
    int ret_code = ioctlsocket(id_, FIONREAD, &num);
#else
    int ret_code = ioctl(id_, FIONREAD, &num);
#endif
    return ret_code == 0 ? (int)num : -1;
}

int Socket::Send(const char *buf, int len, int flags)
{
#ifdef _WIN32
    int bytes_sent = ::send(id_, buf, len, flags);
    if (bytes_sent == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("No data have been sent");
        }
    }
    else if (bytes_sent == SOCKET_ERROR)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Socket error in sending data. WSAGetLastError returns %d", WSAGetLastError());
        }
    }
#else
    ssize_t bytes_sent = ::send(id_, buf, len, flags | MSG_NOSIGNAL);
    if (bytes_sent == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("No data have been sent");
        }
    }
    else if (bytes_sent < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK)
        {
            if (verbosity_ > 5)
            {
                LogWeakWarning("Socket error in sending data. Data length: %d, Error code: %d",
                    len, errno);
            }
        }
    }
#endif

    return (int)bytes_sent;
}

Socket::RetCode Socket::SendEx(int &bytes_sent, const char *buf, int len, int flags)
{
#ifdef _WIN32
    bytes_sent = ::send(id_, buf, len, flags);
    if (bytes_sent == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("No data have been sent");
        }
        return RetCode::Disconnected;
    }
    else if (bytes_sent == SOCKET_ERROR)
    {
        int last_error = WSAGetLastError();
        if (verbosity_ > 5) {
            LogWeakWarning("Socket error in sending data. WSAGetLastError returns %d", last_error);
        }

        if (last_error == WSAECONNABORTED || last_error == WSAECONNRESET) {
            return RetCode::Disconnected;
        }
        return RetCode::OtherError;
    }
#else
    bytes_sent = ::send(id_, buf, len, flags | MSG_NOSIGNAL);
    if (bytes_sent == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("No data have been sent");
        }
        return RetCode::Disconnected;
    }
    else if (bytes_sent < 0)
    {
        if (verbosity_ > 5)
        {
            LogWeakWarning("Socket error in sending data. Data length: %d, Error code: %d",
                len, errno);
        }

        if (errno == ECONNRESET) {
            return RetCode::Disconnected;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return RetCode::Nonblocking;
        }
        return RetCode::OtherError;
    }
#endif

    return RetCode::Success;
}

int Socket::Recv(char *buf, int len, int flags)
{
#ifdef _WIN32
    int ret_code = ::recv(id_, buf, len, flags);
    if (ret_code == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Error in receiving data, because the connection has been gracefully closed");
        }
    }
    else if (ret_code == SOCKET_ERROR)
    {
        int last_error = WSAGetLastError();
        if (verbosity_ > 5) {
            LogWeakWarning("Socket error in receiving data. WSAGetLastError returns %d", last_error);
        }
    }
#else
    int ret_code = ::recv(id_, buf, len, flags);
    if (ret_code == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Error in receiving data, because the connection has been gracefully closed");
        }
    }
    else if (ret_code == ssize_t(-1))
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Socket error in receiving data.");
        }
    }
#endif

    return ret_code;
}

Socket::RetCode Socket::RecvEx(int &bytes_read, char *buf, int len, int flags)
{
#ifdef _WIN32
    bytes_read = ::recv(id_, buf, len, flags);
    if (bytes_read == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Error in receiving data, because the connection has been gracefully closed");
        }
        return RetCode::Disconnected;
    }
    else if (bytes_read == SOCKET_ERROR)
    {
        int last_error = WSAGetLastError();
        if (verbosity_ > 5) {
            LogWeakWarning("Socket error in receiving data. WSAGetLastError returns %d", last_error);
        }

        if (last_error == WSAECONNABORTED || last_error == WSAECONNRESET) {
            return RetCode::Disconnected;
        }
        else if (last_error == WSAETIMEDOUT) {
            return RetCode::Timeout;
        }
        return RetCode::OtherError;
    }
#else
    bytes_read = ::recv(id_, buf, len, flags);
    if (bytes_read == 0 && len > 0)
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Error in receiving data, because the connection has been gracefully closed");
        }
        return RetCode::Disconnected;
    }
    else if (bytes_read == ssize_t(-1))
    {
        if (verbosity_ > 5) {
            LogWeakWarning("Socket error in receiving data (errno: %d)", errno);
        }

        if (errno == ECONNRESET) {
            return RetCode::Disconnected;
        }
        else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return RetCode::Timeout;
        }
        return RetCode::OtherError;
    }
#endif

    return RetCode::Success;
}

Socket::RetCode Socket::ReadAll(char *buf, int len)
{
    RetCode ret_code = RetCode::Success;
    int bytes_left = len;
    int bytes_read = 0;
    bytes_left = len;

    while (bytes_left > 0)
    {
        ret_code = RecvEx(bytes_read, buf, bytes_left);
        if (bytes_read <= 0) {
            return ret_code;
        }

        bytes_left -= bytes_read;
        buf += bytes_read;
    }

    return bytes_left == 0 ? RetCode::Success : RetCode::OtherError;
}

int Socket::Readn(char *buf, int len)
{
    int bytes_left = len;
    int bytes_read = 0;
    bytes_left = len;

    while (bytes_left > 0)
    {
        bytes_read = Read(buf, bytes_left);

        if (bytes_read <= 0) {
            return bytes_read;
        }

        bytes_left -= bytes_read;
        buf += bytes_read;
    }

    return len - bytes_left;
}

Socket::RetCode Socket::WriteAll(const char *buf, int len)
{
    RetCode ret_code = RetCode::Success;
    int bytes_left = len, bytes_written = 0;
    while (bytes_left > 0)
    {
        ret_code = SendEx(bytes_written, buf, bytes_left, 0);
        if (bytes_written <= 0) {
            return ret_code;
        }

        bytes_left -= bytes_written;
        buf += bytes_written;
    }

    return bytes_left == 0 ? RetCode::Success : RetCode::OtherError;
}

int Socket::Writen(const char *buf, int len, int timeout)
{
    if (!is_blocking())
    {
        return WritenForNonblocking(buf, len, timeout);
    }

    int bytes_left = len, bytes_written = 0;
    while (bytes_left > 0)
    {
        bytes_written = Write(buf, bytes_left);
        if (bytes_written <= 0) {
            break;
        }

        bytes_left -= bytes_written;
        buf += bytes_written;
    }

    return len - bytes_left;
}

int Socket::WritenForNonblocking(const char *buf, int len, int timeout)
{
    auto start_tm = chrono::steady_clock::now();
    timeout = max(0, timeout);
    int bytes_left = len, bytes_written = 0;
    while (bytes_left > 0)
    {
        bytes_written = Write(buf, bytes_left);
        if (bytes_written < 0) {
            break;
        }

        bytes_left -= bytes_written;
        buf += bytes_written;

        if (bytes_left > 0)
        {
            auto now_tm = chrono::steady_clock::now();
            int elapsed_time = (int)chrono::duration_cast<chrono::milliseconds>(now_tm - start_tm).count();
            if (elapsed_time >= timeout) {
                break;
            }
        }
    }

    return len - bytes_left;
}

int Socket::ReadInt8(int8_t &val)
{
    return Readn((char*)&val, sizeof(val));
}

int Socket::ReadInt16(int16_t &val)
{
    int ret = Readn((char*)&val, sizeof(val));
    val = ntohs(val);
    return ret;
}

int Socket::ReadInt32(int32_t &val)
{
    int ret = Readn((char*)&val, sizeof(val));
    val = ntohl(val);
    return ret;
}

int Socket::ReadInt64(int64_t &val)
{
    uint32_t val_high = 0, val_low = 0;
    int ret = ReadUInt32(val_high);
    if (ret >= 0) ret = ReadUInt32(val_low);

    val = val_high;
    val <<= 32;
    val |= val_low;

    return ret;
}

int Socket::ReadUInt8(uint8_t &val)
{
    return Readn((char*)&val, sizeof(val));
}

int Socket::ReadUInt16(uint16_t &val)
{
    int ret = Readn((char*)&val, sizeof(val));
    val = ntohs(val);
    return ret;
}

int Socket::ReadUInt32(uint32_t &val)
{
    int ret = Readn((char*)&val, sizeof(val));
    val = ntohl(val);
    return ret;
}

int Socket::ReadUInt64(uint64_t &val)
{
    uint32_t val_high = 0, val_low = 0;
    int ret = ReadUInt32(val_high);
    if (ret >= 0) ret = ReadUInt32(val_low);

    val = val_high;
    val <<= 32;
    val |= val_low;

    return ret;
}

int Socket::WriteInt8(int8_t val)
{
    return Writen((char*)&val, sizeof(val));
}

int Socket::WriteInt16(int16_t val)
{
    val = htons(val);
    return Writen((char*)&val, sizeof(val));
}

int Socket::WriteInt32(int32_t val)
{
    val = htonl(val);
    return Writen((char*)&val, sizeof(val));
}

int Socket::WriteInt64(int64_t val)
{
    uint32_t val_high = (uint32_t)((val >> 32) & 0xFFFFFFFF);
    uint32_t val_low = (uint32_t)(val & 0xFFFFFFFF);

    int ret = WriteUInt32(val_high);
    if (ret >= 0) {
        ret = WriteUInt32(val_low);
    }
    return ret;
}

int Socket::WriteUInt8(uint8_t val)
{
    return Writen((char*)&val, sizeof(val));
}

int Socket::WriteUInt16(uint16_t val)
{
    val = htons(val);
    return Writen((char*)&val, sizeof(val));
}

int Socket::WriteUInt32(uint32_t val)
{
    val = htonl(val);
    return Writen((char*)&val, sizeof(val));
}

int Socket::WriteUInt64(uint64_t val)
{
    uint32_t val_high = (uint32_t)((val >> 32) & 0xFFFFFFFF);
    uint32_t val_low = (uint32_t)(val & 0xFFFFFFFF);

    int ret = WriteUInt32(val_high);
    if (ret >= 0) ret = WriteUInt32(val_low);
    return ret;
}

/**
 * Read a short string from this socket.
 * The string is formated as a 8-bit length plus its contents
 * @param val: the string been read
 */
int Socket::ReadString8(string &val)
{
    uint8_t len = 0;
    int ret = ReadUInt8(len);

    //Attention: len=0 is also valid (means an empty string)
    if (ret > 0 && len > 0)
    {
        char buf[256];
        ret = Readn(buf, len);
        val.assign(buf, len);
    }

    return ret;
}

/**
 * Write a short string to this socket.
 * The string is written into the socket as a 8-bit length plus its contents
 * @param val: the string to be written
 */
int Socket::WriteString8(const string &val)
{
    uint8_t len = (uint8_t)val.size();
    int ret = WriteUInt8(len);

    //Attention: len=0 is also valie (means an empty string)
    if (ret > 0 && len > 0)
    {
        ret = Writen(val.c_str(), len);
    }

    return ret;
}

/**
 * Read a medium-size string from this socket.
 * The string is formated as a 16-bit length plus its contents
 * @param val: the string been read
 */
int Socket::ReadString16(string &val)
{
    uint16_t len = 0;
    int ret = ReadUInt16(len);

    //Attention: len=0 is also valid (means an empty string)
    if (ret > 0 && len > 0)
    {
        char *buf = new char[len];
        ret = Readn(buf, len);
        val.assign(buf, len);
        delete[] buf;
    }

    return ret;
}

/**
 * Write a medium-size string to this socket.
 * The string is written into the socket as a 16-bit length plus its contents
 * @param val: the string to be written
 */
int Socket::WriteString16(const string& val)
{
    uint16_t len = (uint16_t)val.size();
    int ret = WriteUInt16(len);

    //Attention: len=0 is also valid (means an empty string)
    if (ret > 0 && len > 0)
    {
        ret = Writen(val.c_str(), len);
    }

    return ret;
}

/**
 * Read a long string from this socket.
 * The string is formated as a 32-bit length plus its contents
 * @param val: the string been read
 */
int Socket::ReadString32(string& val)
{
    uint32_t len = 0;
    int ret = ReadUInt32(len);

    //Attention: len=0 is also valid (means an empty string)
    if (ret > 0 && len > 0)
    {
        char *buf = new char[len];
        ret = Readn(buf, len);
        val.assign(buf, len);
        delete[] buf;
    }

    return ret;
}

/**
 * Write a long string to this socket.
 * The string is written into the socket as a 32-bit length plus its contents
 * @param val: the string to be written
 */
int Socket::WriteString32(const string &val)
{
    uint32_t len = (uint32_t)val.size();
    int ret = WriteUInt32(len);

    //Attention: len=0 is also valid (means an empty string)
    if (ret > 0 && len > 0)
    {
        ret = Writen(val.c_str(), len);
    }

    return ret;
}
/*
//Write a 4-bytes length first, and then write the specified number of bytes
int Socket::WriteData(const char *buf, int len)
{
	u_long net_length = htonl(len);
	int bytes_written = Writen((char*)&net_length, sizeof(net_length));
	if (bytes_written > 0)
	{
		bytes_written = Writen(buf, len);
	}

	return bytes_written;
}

//Read a 4-byte length first, and then read the specified number of bytes
int Socket::ReadData(char *&buf)
{
	u_long net_length = 0;
	int read_ret = Readn((char*)&net_length, sizeof(net_length));
	if (read_ret >= 0)
	{
		int len = ntohl(net_length);
		buf = new char[len+1];
		buf[len] = '\0';
		read_ret = Readn(buf, len);
	}
	return read_ret;
}
*/

int Socket::SetVerbosity(int verbosity)
{
    int old_verbosity = verbosity_;
    verbosity_ = verbosity;
    return old_verbosity;
}

bool Socket::SetOptions(const SocketOptions &opt)
{
    bool ret = true;
    const SocketOptions::KeepAliveInfo &info = opt.keep_alive_info;
    ret = SetKeepAliveInfo(info.is_on, info.time_val, info.interval) && ret;

    return ret;
}

int Socket::SetOption(int level, int optname, const char *optval, int optlen)
{
    return ::setsockopt(id_, level, optname, optval, optlen);
}

bool Socket::SetOptNoDelay(bool bval)
{
#ifdef _WIN32
    int ret_code = ::setsockopt(id_, IPPROTO_TCP, TCP_NODELAY, (const char*)&bval, sizeof(bval));
#elif defined __APPLE__ || defined Macintosh
    int value = bval ? 1 : 0;
    int ret_code = ::setsockopt(id_, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value));
#else
    int value = bval ? 1 : 0;
    int ret_code = ::setsockopt(id_, SOL_TCP, TCP_NODELAY, &value, sizeof(value));
#endif

    return ret_code == 0;
}

bool Socket::SetSendTimeout(int timeout_ms)
{
#ifdef _WIN32
    DWORD value = (DWORD)timeout_ms;
    int ret = ::setsockopt(id_, SOL_SOCKET, SO_SNDTIMEO, (const char*)&value, sizeof(value));
#else
    timeval value;
    value.tv_sec = timeout_ms / 1000;
    value.tv_usec = (timeout_ms % 1000) * 1000;
    int ret = ::setsockopt(id_, SOL_SOCKET, SO_SNDTIMEO, (const char*)&value, sizeof(value));
#endif

    return ret == 0;
}

bool Socket::SetRecvTimeout(int timeout_ms)
{
#ifdef _WIN32
    DWORD value = (DWORD)timeout_ms;
    int ret = ::setsockopt(id_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&value, sizeof(value));
#else
    timeval value;
    value.tv_sec = timeout_ms / 1000;
    value.tv_usec = (timeout_ms % 1000) * 1000;
    int ret = ::setsockopt(id_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&value, sizeof(value));
#endif

    return ret == 0;
}

bool Socket::SetKeepAliveInfo(bool keep_alive, uint32_t time_val, uint32_t interval)
{
#ifdef _WIN32
    tcp_keepalive keep_alive_info;
    keep_alive_info.onoff = keep_alive ? 1 : 0;
    keep_alive_info.keepalivetime = (ULONG)time_val;
    keep_alive_info.keepaliveinterval = (ULONG)interval;
    DWORD ret_data = 0;
    int ret_code = WSAIoctl(id_, SIO_KEEPALIVE_VALS, &keep_alive_info, sizeof(keep_alive_info),
        nullptr, 0, &ret_data, nullptr, nullptr);
    if (ret_code != 0)
    {
        LogWarning("Failed to set the KeepAlive information (KeepAlive = %s; Time = %u; Interval = %u; LastError = %u)",
            (keep_alive ? "true" : "false"), time_val, interval, WSAGetLastError());
    }

    return ret_code == 0;
#else
    (void)time_val; (void)interval;
    int ret1 = 0, ret2 = 0, ret3 = 0, ret4 = 0;
    int optval = keep_alive ? 1 : 0;
    socklen_t optlen = sizeof(optlen);
    ret1 = setsockopt(id_, SOL_SOCKET, SO_KEEPALIVE, &optval, optlen);
    if (ret1 < 0) {
        LogWarning("Error occurred in setting SO_KEEPALIVE: %s", strerror(errno));
    }

    //optval = 5;
    //int ret2 = setsockopt(id_, SOL_TCP, TCP_KEEPIDLE, &optval, optlen);
    //if(ret2 < 0) {
    //    LogWarning("Error occurred in setting TCP_KEEPIDLE: %s", strerror(errno));
    //}

    //optval = 5;
    //int ret3 = setsockopt(mySocket, SOL_TCP, TCP_KEEPCNT, &optval, optlen);
    //if(ret3 < 0) {
    //    LogWarning("Error occurred in setting TCP_KEEPCNT: %s", strerror(errno));
    //}

    //optval = (int)interval;
    //int ret4 = setsockopt(mySocket, SOL_TCP, TCP_KEEPINTVL, &optval, optlen);
    //if(ret4 < 0) {
    //    LogWarning("Error occurred in setting TCP_KEEPINTVL: %s", strerror(errno));
    //}

    return ret1 == 0 && ret2 == 0 && ret3 == 0 && ret4 == 0;
#endif
}

bool Socket::SetBlocking(bool is_blocking)
{
    if (!IsValidSocketId(id_)) {
        return false;
    }

    unsigned long mode = is_blocking ? 0 : 1;
#ifdef _WIN32
    int result = ioctlsocket(id_, FIONBIO, &mode);
    if (result != NO_ERROR) {
        LogWarning("ioctlsocket failed with error: %ld", result);
    }
    else {
        is_blocking_ = is_blocking;
    }
    return result == NO_ERROR;
#else
    //to do: study fcntl
    int result = ioctl(id_, FIONBIO, &mode);
    if (result != 0) {
        LogWarning("ioctlsocket failed with error: %ld", result);
    }
    else {
        is_blocking_ = is_blocking;
    }
    return result == 0;
#endif //ifdef _WIN32
}

//static
bool Socket::IsIPAddress(const string &str)
{
    for (size_t ch_idx = 0; ch_idx < str.size(); ch_idx++)
    {
        char ch = str[ch_idx];
        if (ch != '.' && !(ch >= '0' && ch <= '9')) {
            return false;
        }
    }

    return true;
}

//static
bool Socket::IsValidSocketId(SocketId sid)
{
#ifdef _WIN32
    return sid != INVALID_SOCKET;
#else
    return sid >= 0;
#endif
}

} //end of namespace
