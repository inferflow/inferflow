#include "base_tcp_server.h"
#include <algorithm>
#include "macro.h"
#include "log.h"
#ifdef _WIN32
#   include <winsock2.h>
#   include <windows.h>
#endif

using namespace std;

namespace sslib
{

TcpTaskQueue::~TcpTaskQueue()
{
    Clear();
}

void TcpTaskQueue::Clear()
{
    while (!core_data_.empty())
    {
        TcpTask *task = core_data_.front();
        if (task != nullptr) {
            delete task;
        }
        core_data_.pop();
    }
}

void TcpTaskQueue::Push(TcpTask *task)
{
    lock_.lock();
    core_data_.push(task);
    lock_.unlock();
}

TcpTask* TcpTaskQueue::Pop()
{
    TcpTask *task = nullptr;
    lock_.lock(); //lock
    if (!core_data_.empty())
    {
        task = core_data_.front();
        core_data_.pop();
    }
    lock_.unlock(); //unlock

    return task;
}

#ifdef _WIN32
enum class IocpOperationType { Unknown = 0, Read, Write };

struct IocpOperationContext : public OVERLAPPED
{
    WSABUF wsa_buf;
    uint32_t buf_len = 0;
    char *buffer = nullptr;
    uint32_t bytes_recv = 0;
    uint32_t type = (uint32_t)IocpOperationType::Unknown;

    IocpOperationContext(uint32_t p_buf_len = 4096);
    ~IocpOperationContext();
    void ClearOverlapped();
};

IocpOperationContext::IocpOperationContext(uint32_t p_buf_len)
{
    //Members of OVERLAPPED must be initialized to zero before the structure is
    //used in a function call.
    //Otherwise, the function may fail and return ERROR_INVALID_PARAMETER.
    Internal = InternalHigh = 0;
    Offset = OffsetHigh = 0;
    hEvent = 0;

    buf_len = p_buf_len;
    buffer = new char[p_buf_len];
    wsa_buf.len = buf_len;
    wsa_buf.buf = buffer;
    bytes_recv = 0;
}

IocpOperationContext::~IocpOperationContext()
{
    if (buffer != nullptr) {
        delete []buffer;
    }
    buffer = nullptr;
}

void IocpOperationContext::ClearOverlapped()
{
    Internal = InternalHigh = 0;
    Offset = OffsetHigh = 0;
    hEvent = 0;
}
#endif

bool BaseTcpServer::Event::IsHangUp() const
{
    return type == EventType::Hup || type == EventType::RdHup;
}

BaseTcpServer::~BaseTcpServer()
{
    Clear();
}

void BaseTcpServer::Clear()
{
    for (std::thread *thread_ptr : worker_list_)
    {
        if (thread_ptr->joinable()) {
            thread_ptr->join();
        }

        delete thread_ptr;
    }
    worker_list_.clear();

    connection_map_lock_.lock(); //lock
    auto iter = connection_map_.begin();
    for (; iter != connection_map_.end(); iter++)
    {
        auto *connection = iter->second;
        if (connection != nullptr)
        {
            connection->socket_ptr->Shutdown();
        }
    }
    connection_map_lock_.unlock(); //unlock

    task_queue_.Clear();

#ifdef _WIN32
    if (iocp_handle_ != INVALID_HANDLE_VALUE)
    {
        CloseHandle(iocp_handle_);
        iocp_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (my_fd_ != -1) {
        close(my_fd_);
    }

    if (event_list_ != nullptr) {
        delete[] event_list_;
    }
#endif

    for (std::thread *thread_ptr : net_io_threads_)
    {
        if (thread_ptr->joinable()) {
            thread_ptr->join();
        }

        delete thread_ptr;
    }
    net_io_threads_.clear();

    iter = connection_map_.begin();
    for (; iter != connection_map_.end(); iter++)
    {
        ConnectionData *connection_data = iter->second;
        if (connection_data != nullptr)
        {
            ClearConnectionData(*connection_data);
            delete connection_data;
            iter->second = nullptr;
        }
    }
    connection_map_.clear();

    global_recv_buffer_.Delete();
}

bool BaseTcpServer::Init(int port, int worker_count, bool is_study_mode)
{
    Socket::Spec spec;
    spec.port = port;
    spec.is_tcp = true;
    spec.is_no_delay = true;
    spec.is_blocking = true;
    spec.reuse_addr = false;
    spec.bind_retrying_times = 10;
    spec.bind_retrying_delay = 10000;

    bool ret = Init(spec, worker_count, is_study_mode);
    return ret;
}

bool BaseTcpServer::Init(const Socket::Spec &spec, int worker_count, bool is_study_mode)
{
    bool ret = true;
    worker_count_ = max(1, min(255, worker_count));
    is_study_mode_ = is_study_mode;

    global_recv_buffer_.New(40960);

#ifdef _WIN32
    ret = CreateIocp();
    if (!ret && is_study_mode_) {
        LogError("Failed to initialize the IOCP");
    }
#else
    ret = CreateEpoll();
    if (!ret && is_study_mode_) {
        LogError("Failed to initialize Epoll");
    }
#endif

    ret = listen_socket_.Initialize(spec);
    Macro_RetxFalseIf(!ret, LogError("Failed to initialize the listening socket"));

    ret = listen_socket_.Listen();
    Macro_RetxFalseIf(!ret, LogError("Listen failed"));

    return ret;
}

bool BaseTcpServer::Start()
{
    bool ret = true;
    is_running_ = true;

    uint32_t net_io_worker_num = 1;
#ifndef _WIN32
    net_io_worker_num = 1; //one thread for epoll
#endif
    for (uint32_t thread_idx = 0; thread_idx < net_io_worker_num; thread_idx++)
    {
        auto *thread_ptr = new std::thread(&BaseTcpServer::WaitForEvents, this);
        net_io_threads_.push_back(thread_ptr);
    }

    for (int thread_idx = 0; thread_idx < worker_count_; thread_idx++)
    {
        auto *thread_ptr = new std::thread(&BaseTcpServer::WorkerFunction, this, thread_idx);
        worker_list_.push_back(thread_ptr);
    }

    // Accept connections and assign to the completion port
    uint64_t total_connection_num = 0;
    while (is_running_)
    {
        std::shared_ptr<Socket> socket_ptr(new Socket);
        ret = listen_socket_.Accept(*socket_ptr);
        if (!ret)
        {
            LogError("Accept failed");
            return false;
        }

        total_connection_num++;
        //if (total_connection_num > 100)
        //{
        //    is_running_ = false;
        //    break;
        //}

        ConnectionData *connection_data = new ConnectionData;
        connection_data->socket_ptr = socket_ptr;

        uint64_t sock_id = connection_data->socket_ptr->GetId();
        if (is_study_mode_) {
            LogKeyInfo("Socket %u got connected...", (uint32_t)sock_id);
        }

#ifndef _WIN32
        //connection_data->socket_ptr->SetBlocking(false);
#endif
        AddEvent(sock_id, connection_data);
    }

    return ret;
}

void BaseTcpServer::Stop()
{
    is_running_ = false;
    listen_socket_.Close();
}

bool BaseTcpServer::WaitForEvents()
{
    int timeout = 5000;
#ifdef _WIN32
    return WaitForEvents_Iocp(timeout);
#else
    return WaitForEvents_Epoll(timeout);
#endif
}

bool BaseTcpServer::WorkerFunction(int thread_idx)
{
    //sleep time in microseconds
    int sleep_time_len = thread_idx < 4 ? (thread_idx < 2 ? 100 : 300)
        : (thread_idx < 16 ? 1000 : (thread_idx < 32 ? 5000 : 10000));

    while (is_running_)
    {
        //get task
        TcpTask *task = task_queue_.Pop();
        if (task == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_len));
            continue;
        }

        //handle request
        //uint64_t socket_id = task->socket_ptr->GetId();
        bool be_keep_connection = HandleTask(task);
        if (!be_keep_connection) {
            //RemoveEvent(socket_id);
        }

        delete task;
    }

    return true;
}

void BaseTcpServer::ClearConnectionData(ConnectionData &connection_data)
{
    if (connection_data.event_context != nullptr)
    {
#ifdef _WIN32
        auto *context = (IocpOperationContext*)connection_data.event_context;
        delete context;
        connection_data.event_context = nullptr;
#endif
    }

    if (connection_data.task != nullptr)
    {
        delete connection_data.task;
        connection_data.task = nullptr;
    }

    connection_data.handler = nullptr;
}

bool BaseTcpServer::HandleEvent(const Event &evt)
{
    bool ret = true;
    if (is_study_mode_) {
        LogKeyInfo("Event (%u, %u, %u)", (uint32_t)evt.fd, evt.type, evt.len);
    }

    if (evt.socket_ptr.get() == nullptr)
    {
        LogError("Null socket pointer");
        return false;
    }

    int read_ret = 0;
    switch(evt.type)
    {
    case EventType::Hup:
    case EventType::RdHup:
        RemoveEvent(evt.fd);
        break;
    case EventType::ReadReady:
        read_ret = evt.socket_ptr->Recv(global_recv_buffer_.data, (int)global_recv_buffer_.size);
        if (read_ret < 0)
        {
            if (errno == EWOULDBLOCK || errno == EAGAIN)
            {
                LogKeyInfo("errno: %d", errno);
            }
            else
            {
                LogWarning("Socket closed?");
            }
        }
        else if (ret == 0)
        {
            LogKeyInfo("No data is available");
        }
        else
        {
            ret = OnDataReceived(evt.fd, global_recv_buffer_.data, (uint32_t)read_ret);
        }
        break;
    case EventType::ReadDone:
        ret = OnDataReceived(evt.fd, (const char*)evt.ptr, evt.len);
        break;
    default:
        LogWarning("Event type %u is not handled", (uint32_t)evt.type);
        break;
    }

    if (!ret) {
        RemoveEvent(evt.fd);
    }

    return ret;
}

bool BaseTcpServer::OnDataReceived(uint64_t socket_id,
    const char *data, uint32_t data_len)
{
    bool ret = true;
    if (data == nullptr || data_len <= 0)
    {
        if (is_study_mode_) {
            LogWeakWarning("Null data or zero length");
        }
        return true;
    }

    if (is_study_mode_) {
        LogKeyInfo("Data received (socket: %u; len: %u)", (uint32_t)socket_id, data_len);
    }

    connection_map_lock_.lock(); //lock
    auto iter_find = connection_map_.find(socket_id);
    ConnectionData *connection = iter_find != connection_map_.end()
        ? iter_find->second : nullptr;
    if (connection != nullptr)
    {
        ret = PrepareTask(connection, data, data_len);
    }
    connection_map_lock_.unlock(); //unlock

    return ret;
}

void BaseTcpServer::AddEvent(uint64_t fd, ConnectionData *connection_data)
{
    if (connection_data == nullptr) {
        LogError("Null connection data");
        return;
    }

    connection_map_lock_.lock(); //lock
    uint32_t old_num = (uint32_t)connection_map_.size();
    auto iter_find = connection_map_.find(fd);
    bool has_this_fd = iter_find != connection_map_.end();
    if (has_this_fd)
    {
        LogKeyInfo("Will replace the handler of fd %u", (uint32_t)fd);
        if (iter_find->second != nullptr)
        {
            ClearConnectionData(*iter_find->second);
            delete iter_find->second;
        }
        iter_find->second = connection_data;
    }
    else
    {
        connection_map_[fd] = connection_data;
        if (is_study_mode_)
        {
            LogKeyInfo("Adding connection %u (%u --> %u)",
                (uint32_t)fd, old_num, old_num + 1);
        }
    }
    connection_map_lock_.unlock(); //unlock

    if (!has_this_fd)
    {
#ifdef _WIN32
        AddEvent_Iocp(fd, connection_data);
#elif defined __APPLE__ || defined Macintosh
        uint32_t event_types = EVFILT_READ;
        AddEvent_Epoll(fd, event_types);
#else
        uint32_t event_types = EPOLLIN | EPOLLRDHUP;
        AddEvent_Epoll(fd, event_types);
#endif
    }
}

void BaseTcpServer::RemoveEvent(uint64_t fd)
{
#ifdef _WIN32
    //RemoveEvent_Iocp(fd);
#else
    RemoveEvent_Epoll(fd);
#endif

    connection_map_lock_.lock(); //lock
    auto iter_find = connection_map_.find(fd);
    bool has_this_fd = iter_find != connection_map_.end();
    if (has_this_fd)
    {
        if (iter_find->second != nullptr)
        {
            ClearConnectionData(*iter_find->second);
            delete iter_find->second;
        }
        connection_map_.erase(iter_find);

        if (is_study_mode_)
        {
            uint32_t new_num = (uint32_t)connection_map_.size();
            LogKeyInfo("Removing connection %u (%u --> %u)",
                (uint32_t)fd, new_num + 1, new_num);
        }
    }
    connection_map_lock_.unlock(); //unlock
}

std::shared_ptr<Socket> BaseTcpServer::GetSocketPtr(uint64_t fd)
{
    std::shared_ptr<Socket> socket_ptr;
    connection_map_lock_.lock(); //lock
    auto iter_find = connection_map_.find(fd);
    if (iter_find != connection_map_.end() && iter_find->second != nullptr)
    {
        socket_ptr = iter_find->second->socket_ptr;
    }
    connection_map_lock_.unlock(); //unlock

    return socket_ptr;
}

#ifdef _WIN32
bool BaseTcpServer::CreateIocp()
{
    // Setup an I/O completion port
    iocp_handle_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (iocp_handle_ == INVALID_HANDLE_VALUE)
    {
        LogError("CreateIoCompletionPort failed with error %d", GetLastError());
        return false;
    }
    return true;
}
#endif

#ifdef _WIN32
bool BaseTcpServer::AddEvent_Iocp(uint64_t fd, ConnectionData *connection_data)
{
    // Associate the accepted socket with the original completion port
    HANDLE this_iocp = CreateIoCompletionPort((HANDLE)fd, iocp_handle_, (ULONG_PTR)connection_data, 0);
    if (this_iocp == INVALID_HANDLE_VALUE)
    {
        LogError("CreateIoCompletionPort failed with error %d", GetLastError());
        return false;
    }

    // Create per I/O socket information structure to associate with the WSARecv call below
    auto *read_context = new IocpOperationContext(iocp_recv_buffer_capacity_);
    connection_data->event_context = read_context;
    read_context->type = (uint32_t)IocpOperationType::Read;
    bool is_succ = StartRecvData(fd, read_context);
    if (!is_succ) {
        LogWarning("to do: handle this error");
    }

    return true;
}
#endif

#ifdef _WIN32
bool BaseTcpServer::WaitForEvents_Iocp(int timeout)
{
    timeout;
    HANDLE completion_port = (HANDLE)iocp_handle_;
    DWORD bytes_transferred;
    ConnectionData *connection_data = nullptr;
    IocpOperationContext *iocp_context = nullptr;

    while (is_running_)
    {
        BOOL is_succ = GetQueuedCompletionStatus(completion_port, &bytes_transferred,
            (PULONG_PTR)&connection_data, (LPOVERLAPPED*)&iocp_context, INFINITE);
        if (!is_succ)
        {
            DWORD last_error = GetLastError();
            if (last_error == ERROR_NETNAME_DELETED)
            {
                bytes_transferred = 0;
            }
            else
            {
                LogError("GetQueuedCompletionStatus failed with error %d", last_error);
                return false;
            }
        }

        if (connection_data == nullptr || iocp_context == nullptr) {
            break;
        }

        Event evt;
        evt.fd = connection_data->socket_ptr->GetId();
        evt.socket_ptr = connection_data->socket_ptr;

        //if an error has occurred on the socket
        if (bytes_transferred == 0)
        {
            evt.type = EventType::RdHup;
            HandleEvent(evt);
            continue;
        }

        if (iocp_context->buf_len < bytes_transferred)
        {
            LogError("Transferred bytes exceed buffer capacity");
            return false;
        }

        switch ((IocpOperationType)iocp_context->type)
        {
        case IocpOperationType::Read:
            evt.type = EventType::ReadDone;
            if (connection_data->event_context != nullptr)
            {
                const auto *read_context = (const IocpOperationContext*)connection_data->event_context;
                evt.ptr = read_context->buffer;
                evt.len = bytes_transferred;
            }
            HandleEvent(evt);
            //OnDataReceived(connection_data->sock, iocp_context->buffer, bytes_transferred);
            StartRecvData(evt.fd, iocp_context);
            break;
        case IocpOperationType::Write:
            evt.type = EventType::WriteDone;
            LogError("Should not get IocpOperationType::Write");
            break;
        default:
            LogError("Unknown IOCP operation type");
            break;
        }
    }

    return true;
}
#endif

#ifdef _WIN32
bool BaseTcpServer::StartRecvData(uint64_t socket_id, void *context_ptr)
{
    IocpOperationContext *context = (IocpOperationContext*)context_ptr;
    if (context == nullptr) {
        return false;
    }

    context->ClearOverlapped();
    context->wsa_buf.buf = context->buffer;
    context->wsa_buf.len = context->buf_len;

    DWORD flags = 0;
    int ret_val = WSARecv((SOCKET)socket_id, &context->wsa_buf, 1,
        nullptr, &flags, (OVERLAPPED*)context, nullptr);

    // could get a synchronous completion
    if (ret_val == SOCKET_ERROR)
    {
        int err = WSAGetLastError();
        if (err != WSA_IO_PENDING)
        {
            LogError("WSARecv error %d on socket %u", err, (uint32_t)socket_id);
            return false;
        }

        if (is_study_mode_) {
            LogKeyInfo("Pending data on WSARecv call for socket %u", (uint32_t)socket_id);
        }
    }
    else
    {
        if (is_study_mode_) {
            LogKeyInfo("WSARecv returns %d for socket %u", ret_val, (uint32_t)socket_id);
        }
    }

    return true;
}
#endif

#ifndef _WIN32
bool BaseTcpServer::CreateEpoll()
{
#if defined __APPLE__ || defined Macintosh
    signal(SIGPIPE, SIG_IGN);
    my_fd_ = kqueue();
    if (my_fd_ != -1)
    {
        event_list_ = new struct kevent[max_event_count_];
    }
#else
    my_fd_ = epoll_create1(0);
    if (my_fd_ != -1)
    {
        event_list_ = new epoll_event[max_event_count_];
    }
#endif

    return my_fd_ != -1;
}
#endif

#ifndef _WIN32
void BaseTcpServer::AddEvent_Epoll(uint64_t fd, uint32_t event_types)
{
#if defined __APPLE__ || defined Macintosh
    struct kevent evt;
    EV_SET(&evt, fd, event_types, EV_ADD | EV_ENABLE, 0, 0, (void*)(intptr_t)fd);
    int ret = kevent(my_fd_, &evt, 1, nullptr, 0, nullptr);
    if (ret < 0) {
        LogError("Failed to register event %d", (int)fd);
        return;
    }
#else
    epoll_event evt;
    memset(&evt, 0, sizeof(evt));
    evt.data.fd = fd;
    evt.events = event_types;

    int ret = epoll_ctl(my_fd_, EPOLL_CTL_ADD, (int)fd, &evt);
    if (ret < 0) {
        LogError("Failed to add fd %d to epoll", (int)fd);
        return;
    }
#endif
}
#endif

#ifndef _WIN32
void BaseTcpServer::RemoveEvent_Epoll(uint64_t fd)
{
#if defined __APPLE__ || defined Macintosh
    struct kevent evt;
    EV_SET(&evt, fd, EVFILT_READ, EV_DELETE, 0, 0, (void*)(intptr_t)fd);
    int ret = kevent(my_fd_, &evt, 1, nullptr, 0, nullptr);
    if (ret < 0) {
        LogError("Failed to delete event %d", (int)fd);
        return;
    }
#else
    epoll_event evt;
    memset(&evt, 0, sizeof(evt));
    int ret = epoll_ctl(my_fd_, EPOLL_CTL_DEL, (int)fd, &evt);
    if (ret < 0)
    {
        switch(errno)
        {
        case EBADF:
            LogError("Failed to delete fd %d from epoll. Bad file descriptor");
            break;
        case EBADFD:
            LogError("Failed to delete fd %d from epoll. File descriptor in bad state");
            break;
        default:
            LogError("Failed to delete fd %d from epoll (error: %d)",
                (int)fd, errno);
        }
        return;
    }
#endif
}
#endif

#ifndef _WIN32
#if defined __APPLE__ || defined Macintosh
bool BaseTcpServer::WaitForEvents_Epoll(int timeout)
{
    struct timespec timeout_spec;
    timeout_spec.tv_sec = timeout / 1000;
    timeout_spec.tv_nsec = (timeout % 1000) * 1000 * 1000;

    while (is_running_)
    {
        int fd_count = kevent(my_fd_, nullptr, 0, event_list_, max_event_count_, &timeout_spec);
        for (int fd_idx = 0; fd_idx < fd_count; fd_idx++)
        {
            Event evt;
            evt.fd = (int)(intptr_t)event_list_[fd_idx].udata;
            int event_type = event_list_[fd_idx].filter;
            auto flags = event_list_[fd_idx].flags;
            if((flags & (EV_ERROR | EV_EOF)) != 0)
            {
                evt.type = EventType::Hup;
            }
            else
            {
                switch (event_type)
                {
                    case EVFILT_READ:
                        evt.type = EventType::ReadReady;
                        break;
                    case EVFILT_WRITE:
                        evt.type = EventType::WriteReady;
                        break;
                    default:
                        break;
                }
            }

            evt.socket_ptr = GetSocketPtr(evt.fd);
            HandleEvent(evt);
        }
    }

    return true;
}
#else
bool BaseTcpServer::WaitForEvents_Epoll(int timeout)
{
    while (is_running_)
    {
        int fd_count = epoll_wait(my_fd_, event_list_, max_event_count_, timeout);
        for (int fd_idx = 0; fd_idx < fd_count; ++fd_idx)
        {
            int fd = event_list_[fd_idx].data.fd;
            (void)fd;
            Event evt;
            uint32_t evt_types = event_list_[fd_idx].events;
            if ((evt_types & (EPOLLHUP | EPOLLRDHUP)) != 0)
            {
                //LogKeyInfo("evt_types: 0x%x", evt_types);
                //LogKeyInfo("EPOLLIN: %u; EPOLLOUT: %u; EPOLLHUP: %u; EPOLLRDHUP: %u",
                //    EPOLLIN, EPOLLOUT, EPOLLHUP, EPOLLRDHUP);
                evt.type = (evt_types & EPOLLRDHUP) != 0 ? EventType::RdHup : EventType::Hup;
            }
            else if ((evt_types & EPOLLIN) != 0)
            {
                evt.type = EventType::ReadReady;
            }
            else if ((evt_types & EPOLLOUT) != 0)
            {
                evt.type = EventType::WriteReady;
            }

            evt.fd = (uint64_t)event_list_[fd_idx].data.fd;
            //evt.ptr = event_list_[fd_idx].data.ptr;
            //evt.u32 = event_list_[fd_idx].data.u32;
            //evt.u64 = event_list_[fd_idx].data.u64;

            evt.socket_ptr = GetSocketPtr(evt.fd);
            HandleEvent(evt);

            //auto iter_find = connection_map_.find(fd);
            //if (iter_find != connection_map_.end() && iter_find->second != nullptr)
            //{
            //    iter_find->second->HandleEvent(evt);
            //}
        }
    }

    return true;
}
#endif
#endif //ifndef _WIN32

} //end of namespace
