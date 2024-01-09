#pragma once

#include <list>
#include <mutex>

namespace sslib
{

using std::list;
using std::mutex;

class BaseTask
{
public:
    BaseTask() {};
    virtual ~BaseTask() {};
};

class TaskQueue
{
public:
    TaskQueue();
    virtual ~TaskQueue();
    void Clear();

    void AddTask(BaseTask *task);
    BaseTask* RetrieveTask();

    void AddResponse(BaseTask *task);
    BaseTask* RetrieveResponse();

protected:
    list<BaseTask*> task_list_;
    list<BaseTask*> response_list_;
    mutex lock_;
};

} //end of namespace
