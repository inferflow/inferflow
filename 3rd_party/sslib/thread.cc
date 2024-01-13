#include "thread.h"
#include "log.h"

using namespace std;

namespace sslib
{

Thread::Thread(const string &name)
{
	SetName(name);
	is_active_ = false;

	fn_user_ = nullptr;
	SetAutoDelete();
}

Thread::~Thread()
{
}

void _ThreadEntry(void *arg_list)
{
    Thread *pThread = (Thread*)arg_list;

    pThread->Run();
    pThread->is_active_ = false;
}

void Thread::SetName(const string &name)
{
    name_ = name;
}

/**
 * Create a thread and use Run() as its main procedure
 * @return 0 on success; other values otherwise
 */
void Thread::Create()
{
    thread_ = std::thread(_ThreadEntry, this);
    is_active_ = true;
}

/**
 * Create a thread and use fn_start_routine() as its main procedure
 * @param fn_start_routine: main procedure
 * @param arg_list: parameter list
 * @return 0 on success; other values otherwise
 */
void Thread::Create(THREADPROC fn_start_routine, void *arg_list)
{
    fn_user_ = fn_start_routine;
    fn_user_args_ = arg_list;
    return Create();
}

void Thread::Join()
{
    if (thread_.joinable())
    {
        thread_.join();
        is_active_ = false;
    }
}

//////////////////////////////////////////////////////////////////////////
// Virtual methods

void Thread::Run()
{
    if (fn_user_) {
        (*fn_user_)(fn_user_args_);
    }
    else {
        LogWarning("To do something meaningful, you should generate a subclass from Thread");
    }
}

void Thread::CancelThread()
{
    LogWarning("You had better to implement method CancelThread() in a subclass");
}

//////////////////////////////////////////////////////////////////////////
// Static functions

// We have this function because timed_mutex::try_lock_for has a bug on Linux
// https://stackoverflow.com/questions/17518464/why-does-stdtimed-mutextry-lock-for-not-work
// http://gcc.gnu.org/bugzilla/show_bug.cgi?id=54562
//
//static
bool Thread::TryLockFor(std::mutex &lock, int timeout)
{
    auto tp1 = chrono::steady_clock::now();
    while (!lock.try_lock())
    {
        auto tp2 = chrono::steady_clock::now();
        auto elapsedTime = chrono::duration_cast<chrono::milliseconds>(tp2 - tp1).count();
        if (elapsedTime >= timeout) {
            return false;
        }

        this_thread::sleep_for(chrono::microseconds(100));
    }

    return true;
}

//////////////////////////////////////////////////////////////////////////
// ThreadList

ThreadList::~ThreadList()
{
    Clear();
}

void ThreadList::Clear()
{
    for (auto *thread_ptr : data_)
    {
        if (thread_ptr != nullptr)
        {
            thread_ptr->Join();
            delete thread_ptr;
        }
    }

    data_.clear();
}

void ThreadList::Add(Thread *thread_ptr)
{
    data_.push_back(thread_ptr);
}

const Thread* ThreadList::Get(int thread_idx) const
{
    return thread_idx >= 0 && thread_idx < (int)data_.size() ? data_[thread_idx] : nullptr;
}

Thread* ThreadList::Get(int thread_idx)
{
    return thread_idx >= 0 && thread_idx < (int)data_.size() ? data_[thread_idx] : nullptr;
}

void ThreadList::Join()
{
    for (auto *thread_ptr : data_)
    {
        if (thread_ptr != nullptr)
        {
            thread_ptr->Join();
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// StdThreadList

StdThreadList::~StdThreadList()
{
    Clear();
}

void StdThreadList::Clear()
{
    for (auto *thread_ptr : data_)
    {
        if (thread_ptr != nullptr)
        {
            if (thread_ptr->joinable()) {
                thread_ptr->join();
            }
            delete thread_ptr;
        }
    }

    data_.clear();
}

void StdThreadList::Add(std::thread *thread_ptr)
{
    data_.push_back(thread_ptr);
}

const std::thread* StdThreadList::Get(int thread_idx) const
{
    return thread_idx >= 0 && thread_idx < (int)data_.size() ? data_[thread_idx] : nullptr;
}

std::thread* StdThreadList::Get(int thread_idx)
{
    return thread_idx >= 0 && thread_idx < (int)data_.size() ? data_[thread_idx] : nullptr;
}

void StdThreadList::Join()
{
    for (auto *thread_ptr : data_)
    {
        if (thread_ptr != nullptr && thread_ptr->joinable())
        {
            thread_ptr->join();
        }
    }
}

} //end of namespace
