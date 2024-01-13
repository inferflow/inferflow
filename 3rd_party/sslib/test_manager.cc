#include "test_manager.h"
#include "log.h"
#include "vector_ex.h"
#include "string_util.h"
#include "task_monitor.h"

namespace sslib
{

class TestManagerInternalData
{
public:
    PtrVector<BasicTest> test_ptr_list_;
    std::vector<string> test_name_list_;

    TestManagerInternalData() {};
};

//static
bool TestManager::initialized_ = false;
//static
void* TestManager::internal_data_ = nullptr;

//static
bool TestManager::AddTest(BasicTest *new_test, const string &test_name)
{
    if (new_test == nullptr || test_name.empty()) {
        return false;
    }

    if (!initialized_)
    {
        internal_data_ = new TestManagerInternalData;
        initialized_ = true;
    }

    auto *std_data = (TestManagerInternalData*)internal_data_;
    std_data->test_ptr_list_.push_back(new_test);
    std_data->test_name_list_.push_back(test_name);
    return true;
}

//static
bool TestManager::Run(const string &test_name_filter, const string &data_dir)
{
    if (internal_data_ == nullptr)
    {
        LogKeyInfo("No registered tests");
        return true;
    }

    auto *std_data = (TestManagerInternalData*)internal_data_;

    uint32_t test_count = (uint32_t)std_data->test_ptr_list_.size();
    if (test_count != (uint32_t)std_data->test_name_list_.size())
    {
        LogError("Invalid test settings");
        return false;
    }

    LogKeyInfo("Registered tests: %u", test_count);

    bool ret = true;
    uint32_t matching_count = 0;
    for (uint32_t idx = 0; idx < test_count; idx++)
    {
        BasicTest *test = std_data->test_ptr_list_[idx];
        const string &test_name = std_data->test_name_list_[idx];

        bool is_matching = StringUtil::WildcardMatch(test_name, test_name_filter);
        if (!is_matching) {
            continue;
        }

        matching_count++;
        LogKeyInfo("##### %s #####", test_name.c_str());
        TaskMonitor tm(0);
        ret = test->Run(data_dir);
        tm.ShowElapsedTime(L"Time cost");
        if (ret)
        {
            LogMilestone("%s: Successful", test_name.c_str());
        }
        else
        {
            LogError("%s: Failed", test_name.c_str());
        }
    }

    return ret;
}

//static
void TestManager::Finalize()
{
    if (internal_data_ != nullptr)
    {
        delete (TestManagerInternalData*)internal_data_;
        internal_data_ = nullptr;
    }

    initialized_ = false;
}

} //end of namespace
