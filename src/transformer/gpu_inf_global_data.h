#pragma once

#include <mutex>
#include "sslib/vector_ex.h"
#include "tensor/device_tensor.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::map;
using std::mutex;
using sslib::PtrVector;

struct SourceTensor
{
    DeviceTensor *tensor = nullptr;
    int start_row = 0;
};

struct TensorWithDeviceId
{
    const DeviceTensor *tensor = nullptr;
    int device_id = 0;
};

class GpuInfGlobalData
{
public:
    bool Init(int worker_num); //not thread-safe
    void Clear(); //not thread-safe

    //
    bool Add(int worker_id, int phase_id, const DeviceTensor *output_tensor, int device_id);
    bool Add(int worker_id, int phase_id, const vector<SourceTensor> &tensor_list, int device_id);
    bool IsPhaseDone(int phase_id);
    bool IsInPhase(int phase_id);
    void MoveToPhase(int phase_id, const DeviceTensor *input_tensor, int device_id);

    bool GetOutputTensors(vector<TensorWithDeviceId> &tensor_list, int phase_id);
    bool GetSourceTensors(vector<SourceTensor> &tensor_list, int phase_id);
    const DeviceTensor* GetInputTensor(int worker_id, int phase_id);

    void Lock4Print() const
    {
#       ifdef __GNUC__
#           pragma GCC diagnostic push
#           pragma GCC diagnostic ignored "-Wcast-qual"
#       endif
        //lock
        ((mutex&)print_lock_).lock();
#       ifdef __GNUC__
#           pragma GCC diagnostic pop
#       endif
    }

    void Unlock4Print() const
    {
#       ifdef __GNUC__
#           pragma GCC diagnostic push
#           pragma GCC diagnostic ignored "-Wcast-qual"
#       endif
        //unlock
        ((mutex&)print_lock_).unlock();
#       ifdef __GNUC__
#           pragma GCC diagnostic pop
#       endif
    }

protected:
    struct PhaseData
    {
        int phase = 0;
        const DeviceTensor *input_tensor = nullptr;
        const DeviceTensor *output_tensor = nullptr;
        vector<SourceTensor> source_tensors;
        int device_id = 0;
        bool is_done = false;
    };

    struct WorkerData
    {
        PtrVector<PhaseData> phase_list;
        map<int, PhaseData*> phase_map;
    };

protected:
    PtrVector<WorkerData> worker_list_;
    mutex lock_;
    mutex print_lock_;

protected:
    PhaseData* AddPhase(WorkerData *worker_data, int phase_id, int device_id);
    const PhaseData* GetPhase(const WorkerData *worker_data, int phase_id) const;
};

TRANSFORMER_END
INFER_FLOW_END
