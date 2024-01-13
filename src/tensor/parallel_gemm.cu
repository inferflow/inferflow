#include "parallel_gemm.h"
#include "sslib/task_monitor.h"
#include "common/cuda_util.h"

INFER_FLOW_BEGIN

using namespace std;
using namespace sslib;

ParallelGemmWorker::ParallelGemmWorker()
{
}

ParallelGemmWorker::~ParallelGemmWorker()
{
}

bool ParallelGemmWorker::Init(int id, int device_id, CublasEngine &cublas_engine)
{
    this->id_ = id;
    this->device_id_ = device_id;
    this->cublas_engine_ = &cublas_engine;
    return true;
}

void ParallelGemmWorker::SetData(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, bool is_b_column_major)
{
    this->A_ = &A;
    this->B_ = &B;
    this->C_ = &C;
    this->is_b_column_major_ = is_b_column_major;
}

void ParallelGemmWorker::Run()
{
    TaskMonitor tm;
    CudaUtil::SetDevice(device_id_);

    bool ret = true;
    bool is_b_column_major = is_b_column_major_;
    MatrixMulAlg matrix_mul_alg = MatrixMulAlg::Cublas;
    if (matrix_mul_alg == MatrixMulAlg::Cublas)
    {
        ret = cublas_engine_->GemmEx(*C_, *A_, *B_, 1.0f, 0, is_b_column_major);
    }
    else
    {
        ret = TensorMul::Gemm(*C_, *A_, *B_, 1.0f, 0, is_b_column_major,
            false, matrix_mul_alg);
    }

    stat_.time_cost = tm.GetElapsedTime(false) / 1000.0f;
    stat_.is_succ = ret;
}

void ParallelGemmWorker::CancelThread()
{
}

INFER_FLOW_END

