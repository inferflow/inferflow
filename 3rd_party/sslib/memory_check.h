#pragma once

//refer to the MSDN document titled "Find memory leaks with the CRT library"
//https://docs.microsoft.com/en-us/visualstudio/debugger/finding-memory-leaks-using-the-crt-library?view=vs-2017

#ifdef _WIN32
#   define _CRTDBG_MAP_ALLOC   //for memory-leak checking
#   include <stdlib.h>         //for memory-leak checking
#   include <crtdbg.h>         //for memory-leak checking
#endif //def _WIN32

#ifdef _DEBUG
//#   define DEBUG_NEW new (_NORMAL_BLOCK , __FILE__ , __LINE__)
//#   define new DEBUG_NEW
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type
#endif //def _DEBUG

#include "prime_types.h"

namespace sslib
{

void EnableMemoryLeakReport(uint32_t value = 0)
{
#ifdef _WIN32
    if (value != 0 && value != uint32_t(-1)) {
        _CrtSetBreakAlloc(value);
    }
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#else
    (void)value;
#endif
}

void EnableMemoryLeakReport(uint32_t value_array[])
{
#ifdef _WIN32
    size_t num = sizeof(value_array) / sizeof(uint32_t);
    for (size_t idx = 0; idx < num; idx++) {
        _CrtSetBreakAlloc(value_array[idx]);
    }
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#else
    (void)value_array;
#endif
}

} //end of namespace
