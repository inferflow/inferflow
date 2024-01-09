#include "task_queue.h"

namespace sslib
{

TaskQueue::TaskQueue()
{
}

TaskQueue::~TaskQueue()
{
    Clear();
}

void TaskQueue::Clear()
{
    lock_.lock();

    for (auto *task : task_list_) {
        delete task;
    }
    task_list_.clear();

    for (auto *res : response_list_) {
        delete res;
    }
    response_list_.clear();

    lock_.unlock();
}

void TaskQueue::AddTask(BaseTask *task)
{
    lock_.lock();
    task_list_.push_back(task);
    lock_.unlock();
}

BaseTask* TaskQueue::RetrieveTask()
{
    BaseTask *task = nullptr;
    lock_.lock();
    if (!task_list_.empty())
    {
        task = task_list_.front();
        task_list_.pop_front();
    }
    lock_.unlock();

    return task;
}

void TaskQueue::AddResponse(BaseTask *res)
{
    lock_.lock();
    response_list_.push_back(res);
    lock_.unlock();
}

BaseTask* TaskQueue::RetrieveResponse()
{
    BaseTask *res = nullptr;
    lock_.lock();
    if (!response_list_.empty())
    {
        res = response_list_.front();
        response_list_.pop_front();
    }
    lock_.unlock();

    return res;
}

} //end of namespace
