#pragma once

#include <mutex>
#include "vector_ex.h"

namespace sslib
{

using std::mutex;

class WorkerPool
{
public:
    class Worker
    {
    public:
        virtual ~Worker() {};

    protected:
        bool IsIdle() const {
            return is_idle_;
        }

        void SetIdle(bool is_idle = true) {
            is_idle_ = is_idle;
        }

    protected:
        bool is_idle_ = true;
        friend class WorkerPool;
    };

public:
    WorkerPool() {};
    virtual ~WorkerPool();

    void Add(Worker *new_worker);
    Worker* BorrowWorker(int timeout, int worker_idx = -1) const;
    void ReturnWorker(Worker *the_worker) const;

protected:
    PtrVector<Worker> worker_list_;
    mutex lock_;
};

} //end of namespace
