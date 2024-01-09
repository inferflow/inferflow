#pragma once

#include <cstdint>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <memory>
#ifdef _WIN32
//#   include <winsock2.h>
//#   include <windows.h>
#elif defined __APPLE__ || defined Macintosh
#   include <unistd.h>
#   include <sys/types.h>
#   include <sys/event.h>
#   include <sys/time.h>
#else
#   include <unistd.h>
#   include <sys/epoll.h>
#   include <map>
#endif
#include "socket.h"
#include "raw_array.h"

namespace sslib
{

struct TcpTask
{
    std::shared_ptr<Socket> socket_ptr;

    virtual ~TcpTask() {}
};

class TcpTaskQueue
{
public:
    TcpTaskQueue() {};
    virtual ~TcpTaskQueue();
    void Clear();

    void Push(TcpTask *task);
    TcpTask* Pop();

protected:
    std::queue<TcpTask*> core_data_;
    std::mutex lock_;
};

class BaseTcpServer
{
public:
    enum class EventType
    {
        Unknown = 0,
        ReadReady,
        ReadDone,
        WriteReady,
        WriteDone,
        Hup, //hang up
        RdHup
    };

    struct Event
    {
        EventType type = EventType::Unknown;
        std::shared_ptr<Socket> socket_ptr;
        uint64_t fd = 0;
        void *ptr = nullptr;
        uint32_t len = 0;

        bool IsHangUp() const;
    };

    class EventHandler
    {
    public:
        virtual void HandleEvent(const Event &evt) = 0;
    };

public:
    BaseTcpServer() {};
    virtual ~BaseTcpServer();
    void Clear();

    bool Init(int port, int worker_count, bool is_study_mode = false);
    bool Init(const Socket::Spec &spec, int worker_count, bool is_study_mode = false);
    bool Start();
    void Stop();

protected:
    struct ConnectionData
    {
        uint32_t id = 0;
        std::shared_ptr<Socket> socket_ptr;
        void *event_context = nullptr;
        EventHandler *handler = nullptr;

        TcpTask *task = nullptr;
    };

protected:
    bool is_study_mode_ = false;
    int max_event_count_ = 100;
    int iocp_recv_buffer_capacity_ = 4096;
    RawArray<char> global_recv_buffer_;

    std::mutex connection_map_lock_;
    std::map<uint64_t, ConnectionData*> connection_map_;

    bool is_running_ = false;
    Socket listen_socket_;
    vector<std::thread*> net_io_threads_;

    TcpTaskQueue task_queue_;

    int worker_count_ = 2;
    vector<std::thread*> worker_list_;

#ifdef _WIN32
    void* iocp_handle_ = (void*)(uint64_t)(-1);
#elif defined __APPLE__ || defined Macintosh
    int my_fd_ = -1;
    struct kevent *event_list_ = nullptr;
#else
    int my_fd_ = -1;
    epoll_event *event_list_ = nullptr;
#endif

protected:
    virtual bool PrepareTask(ConnectionData *connection,
        const char *data, uint32_t data_len) = 0;

    //return: whether we should keep the connection
    virtual bool HandleTask(TcpTask *task) = 0;

    virtual bool HandleEvent(const Event &evt);
    virtual bool OnDataReceived(uint64_t socket_id, const char *data, uint32_t data_len);

    void AddEvent(uint64_t fd, ConnectionData *connection_data);
    void RemoveEvent(uint64_t fd);
    std::shared_ptr<Socket> GetSocketPtr(uint64_t fd);

    bool WaitForEvents();
    bool WorkerFunction(int thread_idx);

    void ClearConnectionData(ConnectionData &connection_data);

#ifdef _WIN32
    bool CreateIocp();
    bool AddEvent_Iocp(uint64_t fd, ConnectionData *connection_data);
    bool WaitForEvents_Iocp(int timeout);
    bool StartRecvData(uint64_t socket_id, void *context);
#else
    bool CreateEpoll();
    void AddEvent_Epoll(uint64_t fd, uint32_t event_types);
    void RemoveEvent_Epoll(uint64_t fd);
    bool WaitForEvents_Epoll(int timeout);
#endif

    BaseTcpServer(const BaseTcpServer &rhs) = delete;
    BaseTcpServer& operator = (const BaseTcpServer &rhs) = delete;
};

} //end of namespace
