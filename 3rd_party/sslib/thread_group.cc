#include "thread_group.h"

using namespace std;

namespace sslib
{

ThreadGroup::ThreadGroup()
{

}

ThreadGroup::~ThreadGroup()
{
}

void ThreadGroup::Add(Thread *thread_ptr)
{
    lock_.lock();
    threads_.push_back(thread_ptr);
    lock_.unlock();
}

Thread* ThreadGroup::Remove(Thread *thread_ptr)
{
    lock_.lock();

    for (int i = 0; i < (int)threads_.size(); i++)
    {
        Thread *pItem = threads_[i];
        if (pItem == thread_ptr)
        {
            threads_.erase(threads_.begin() + i);
            lock_.unlock();
            return pItem;
        }
    }

    lock_.unlock();
    return nullptr;
}

int ThreadGroup::GetThreadsCount()
{
	int nRes;
	lock_.lock();
	nRes = (int)threads_.size();
	lock_.unlock();
	return nRes;
}

void ThreadGroup::RemoveZombies()
{
	lock_.lock();
    RemoveZombies_NoLock();
	lock_.unlock();
}

void ThreadGroup::RemoveZombiesAndAdd(Thread *thread_ptr)
{
    lock_.lock();
    RemoveZombies_NoLock();
    threads_.push_back(thread_ptr);
    lock_.unlock();
}

void ThreadGroup::RemoveZombies_NoLock()
{
    for (int i = (int)threads_.size() - 1; i >= 0; i--)
    {
        Thread *thread_ptr = threads_[i];
        if (!thread_ptr->IsActive())
        {
            thread_ptr->Join();	//wait for exit
            if (thread_ptr->GetAutoDelete()) {
                delete thread_ptr;
            }

            //delete from list
            threads_.erase(threads_.begin() + i);
        }
    }
}

void ThreadGroup::CancelThreads()
{
    lock_.lock();

    vector<Thread*>::iterator iter = threads_.begin();
    while (iter != threads_.end())
    {
        Thread *thread_ptr = *iter;
        thread_ptr->CancelThread();

        iter++;
    }

    lock_.unlock();
}

void ThreadGroup::JoinThreads()
{
    lock_.lock();

    vector<Thread*>::iterator iter = threads_.begin();
    for (; iter != threads_.end(); iter++)
    {
        Thread *thread_ptr = *iter;
        thread_ptr->Join();
    }

    lock_.unlock();
}

} //end of namespace
