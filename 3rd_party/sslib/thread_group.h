#pragma once

#include <vector>
#include <mutex>
#include "prime_types.h"
#include "thread.h"

namespace sslib
{

//ThreadGroup: A thread-safe list of threads
class ThreadGroup
{
public:
    ThreadGroup();
    virtual ~ThreadGroup();

    void Add(Thread *thread_ptr);
    Thread* Remove(Thread *thread_ptr);

    int Size() {
        return GetThreadsCount();
    }
    int GetThreadsCount();

    void RemoveZombies();
    void RemoveZombiesAndAdd(Thread *thread_ptr);

    //Notify that all threads in the group should exit
    void CancelThreads();
    //Wait for all threads in the group to exit
    void JoinThreads();

public:
    std::vector<Thread*> threads_; //Thread list
    std::mutex lock_; //A lock for thread-safe access

protected:
    void RemoveZombies_NoLock();
}; //class ThreadGroup

} //end of namespace
