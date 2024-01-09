#pragma once

#include "sslib/thread.h"
#include "tensor/tensor_opr.h"
#include "tensor/tensor_mul.h"
#include "tensor/cublas_engine.h"

INFER_FLOW_BEGIN

using std::vector;
using sslib::Thread;

struct ParallelGemmStat
{
    bool is_succ = true;
    float time_cost = 0;
};

class ParallelGemmWorker : public Thread
{
public:
    ParallelGemmWorker();
    virtual ~ParallelGemmWorker();

    bool Init(int id, int device_id, CublasEngine &cublas_engine);
    void SetData(DeviceTensor &C, const DeviceTensor &A, const DeviceTensor &B,
        bool is_b_column_major = false);

    const ParallelGemmStat& GetStat() const {
        return stat_;
    }

    virtual void Run() override;
    virtual void CancelThread() override;

protected:
    int id_ = 0;
    int device_id_ = 0;
    CublasEngine *cublas_engine_ = nullptr;

    const DeviceTensor *A_ = nullptr;
    const DeviceTensor *B_ = nullptr;
    DeviceTensor *C_ = nullptr;
    bool is_b_column_major_ = false;
    ParallelGemmStat stat_;
};

INFER_FLOW_END

