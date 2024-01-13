#pragma once

#include <string>
#include <vector>
#include <thread>
#include <mutex>

namespace sslib
{

typedef void (*THREADPROC)(void*);

class Thread
{
protected:
    std::string name_;	//the name of this thread
    bool is_active_ = false;

    //This member variable is used by ThreadGroup class. If it's value if true, then 
    //ThreadGroup will free the memory allocated for this thread
    bool is_auto_delete_ = true;

    std::thread thread_;

    THREADPROC fn_user_;
    void *fn_user_args_;

public:
    Thread(const std::string &name = "");
    virtual ~Thread();

    std::thread::id GetHandle() {
        return thread_.get_id();
    }
    const std::string GetName() {
        return name_;
    }
    void SetName(const std::string &name);

    bool IsActive() {
        return is_active_;
    }

    //
    void SetAutoDelete(bool is_auto_delete = true) {
        is_auto_delete_ = is_auto_delete;
    }
    bool GetAutoDelete() {
        return is_auto_delete_;
    }

    //Create the thread
    //@Attention: It's up to the user to free the memory for the returned object
    static Thread* CreateThread(THREADPROC fn_start_routine, void *arg_list);
    void Create();
    void Create(THREADPROC fn_start_routine, void *arg_list);

    //This method is called by other threads, waiting for this thread to exist
    void Join();

/// Child classes can override the following methods
public:
    // main procedure for the thread
    virtual void Run();

    // Called by another thread to notify that this thread should exit as soon as quickly
    // This mechanism can make this thread quit gracefully. Tasks processed by threads can be quite different, so it's good
    //to let each thread to stop and exit by itself.
    virtual void CancelThread();

/// static methods
public:
    static void Sleep(int seconds) {
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
    }
    static void SleepMilli(int milli_seconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(milli_seconds));
    }
    static void SleepMicro(int micro_seconds) {
        std::this_thread::sleep_for(std::chrono::microseconds(micro_seconds));
    }

    static bool TryLockFor(std::mutex &lock, int timeout);

///
protected:
    //the function pointer for _beginthread
    friend void _ThreadEntry(void *arg_list);
}; //class Thread

class ThreadList
{
public:
    ThreadList() {};
    virtual ~ThreadList();

    void Clear();
    void Add(Thread *thread_ptr);
    const Thread* Get(int thread_idx) const;
    Thread* Get(int thread_idx);

    void Join();

protected:
    std::vector<Thread*> data_;
};

class StdThreadList
{
public:
    StdThreadList() {};
    virtual ~StdThreadList();

    void Clear();
    void Add(std::thread *thread_ptr);
    const std::thread* Get(int thread_idx) const;
    std::thread* Get(int thread_idx);

    void Join();

protected:
    std::vector<std::thread*> data_;
};

} //end of namespace
