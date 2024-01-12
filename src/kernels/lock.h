#pragma once

#include "cuda_runtime.h"

__device__ void SleepClockCycles(long long sleep_cycles)
{
    auto start = clock64();
    long long cycles_elapsed = 0;
    do
    {
        cycles_elapsed = clock64() - start;
    }
    while (cycles_elapsed < sleep_cycles);
}

__device__ void MutexLock(unsigned int *mutex)
{
    //unsigned ns = 8;
    long long sleep_cycles = 4;
    while (atomicCAS(mutex, 0, 1) != 0)
    {
        //__nanosleep(ns);
        //if (ns < 256) {
        //    ns *= 2;
        //}
        SleepClockCycles(sleep_cycles);
        if (sleep_cycles < 128) {
            sleep_cycles *= 2;
        }
    }
}

__device__ void MutexUnlock(unsigned int *mutex)
{
    atomicExch(mutex, 0);
}

