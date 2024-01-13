#pragma once

#include <string>

namespace sslib
{

using std::string;

class BasicTest
{
public:
    virtual ~BasicTest() {};

    virtual bool Run(const string &data_dir) = 0;
};

class TestManager
{
public:
    //test_name can be in format of "<module>.<name>"
    static bool AddTest(BasicTest *new_test, const string &test_name);
    static bool Run(const string &test_name_filter, const string &data_dir);
    static void Finalize();

protected:
    static bool initialized_;
    static void *internal_data_;
};

template <typename T>
struct TestHelper
{
    TestHelper(const string &test_name)
    {
        T *new_test = new T;
        TestManager::AddTest(new_test, test_name);
    }
};

#define Macro_RegisterTest(TestType, test_name) \
    static TestHelper<TestType> *test_helper = new TestHelper<TestType>(test_name);

} //end of namespace
