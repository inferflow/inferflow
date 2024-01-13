#include "worker_pool.h"
#include "thread.h"

namespace sslib
{

WorkerPool::~WorkerPool()
{
    worker_list_.Clear(true);
}

void WorkerPool::Add(Worker *new_worker)
{
    worker_list_.push_back(new_worker);
}

WorkerPool::Worker* WorkerPool::BorrowWorker(int timeout, int worker_idx) const
{
    Worker *idle_worker = nullptr;
    auto tm1 = chrono::steady_clock::now();
    while (idle_worker == nullptr)
    {
#       ifdef __GNUC__
#       pragma GCC diagnostic push
#       pragma GCC diagnostic ignored "-Wcast-qual"
#       endif

        if (Thread::TryLockFor((mutex&)lock_, 1)) //1: 1ms
        {
            for (int idx = 0; idx < (int)worker_list_.size(); idx++)
            {
                auto *worker_ptr = worker_list_[idx];
                if ((worker_idx < 0 || worker_idx == idx) && worker_ptr->IsIdle())
                {
                    worker_ptr->SetIdle(false);
                    idle_worker = worker_ptr;
                    break;
                }
            }
            ((mutex&)lock_).unlock();
        }

#       ifdef __GNUC__
#       pragma GCC diagnostic pop
#       endif

        if (idle_worker != nullptr) {
            return idle_worker;
        }

        auto tm = chrono::steady_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(tm - tm1).count();
        if (elapsed_time > timeout)
        {
            //LogWarning("Time out in borrowing a worker (%d > %d)",
            //    elapsed_time, timeout);
            return nullptr;
        }

        Thread::SleepMicro(100);
    }

    return idle_worker;
}

void WorkerPool::ReturnWorker(Worker *the_worker) const
{
#   ifdef __GNUC__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wcast-qual"
#   endif

    ((mutex&)lock_).lock(); //unlock
    the_worker->SetIdle(true);
    ((mutex&)lock_).unlock(); //unlock

#   ifdef __GNUC__
#   pragma GCC diagnostic pop
#   endif
}

} //end of namespace
