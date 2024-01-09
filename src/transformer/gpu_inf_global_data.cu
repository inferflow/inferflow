#include "gpu_inf_global_data.h"
#include "sslib/log.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

bool GpuInfGlobalData::Init(int worker_num)
{
    for (int idx = 0; idx < worker_num; idx++)
    {
        WorkerData *item = new WorkerData;
        worker_list_.push_back(item);
    }
    return true;
}

void GpuInfGlobalData::Clear()
{
    worker_list_.Clear(true);
}

bool GpuInfGlobalData::Add(int worker_id, int phase_id,
    const DeviceTensor *output_tensor, int device_id)
{
    lock_.lock();
    bool ret = worker_id >= 0 && worker_id < (int)worker_list_.size();
    if (ret)
    {
        WorkerData *worker_data = worker_list_[worker_id];
        auto *phase_ptr = AddPhase(worker_data, phase_id, device_id);
        phase_ptr->output_tensor = output_tensor;
        phase_ptr->is_done = true;
    }
    lock_.unlock();

    return ret;
}

bool GpuInfGlobalData::Add(int worker_id, int phase_id,
    const vector<SourceTensor> &tensor_list, int device_id)
{
    lock_.lock();
    bool ret = worker_id >= 0 && worker_id < (int)worker_list_.size();
    if (ret)
    {
        WorkerData *worker_data = worker_list_[worker_id];
        auto *phase_ptr = AddPhase(worker_data, phase_id, device_id);
        phase_ptr->source_tensors = tensor_list;
        phase_ptr->is_done = true;
    }
    lock_.unlock();

    return ret;
}

bool GpuInfGlobalData::IsPhaseDone(int phase_id)
{
    bool is_done = true;
    lock_.lock();
    for (const auto *worker_data : worker_list_)
    {
        const PhaseData *phase_ptr = GetPhase(worker_data, phase_id);
        if (phase_ptr == nullptr) {
            is_done = false;
            break;
        }

        bool is_item_done = (phase_ptr->phase == phase_id && phase_ptr->is_done)
            || phase_ptr->phase > phase_id;
        if (!is_item_done) {
            is_done = false;
            break;
        }
    }
    lock_.unlock();

    return is_done;
}

bool GpuInfGlobalData::IsInPhase(int phase_id)
{
    bool ret = true;
    lock_.lock();
    for (const auto *worker_data : worker_list_)
    {
        const PhaseData *phase_ptr = GetPhase(worker_data, phase_id);
        if (phase_ptr == nullptr || phase_ptr->phase < phase_id) {
            ret = false;
            break;
        }
    }
    lock_.unlock();

    return ret;
}

void GpuInfGlobalData::MoveToPhase(int phase_id, const DeviceTensor *input_tensor,
    int device_id)
{
    lock_.lock();
    for (auto *worker_data : worker_list_)
    {
        PhaseData *phase_ptr = AddPhase(worker_data, phase_id, device_id);
        phase_ptr->output_tensor = nullptr;
        phase_ptr->source_tensors.clear();
        phase_ptr->input_tensor = input_tensor;
        phase_ptr->is_done = false;
    }
    lock_.unlock();
}

bool GpuInfGlobalData::GetOutputTensors(vector<TensorWithDeviceId> &tensor_list, int phase_id)
{
    bool ret = false;
    tensor_list.clear();
    lock_.lock();

    for (const WorkerData *worker_data : worker_list_)
    {
        const PhaseData *phase_ptr = GetPhase(worker_data, phase_id);
        if (phase_ptr != nullptr)
	{
            TensorWithDeviceId tdi;
            tdi.tensor = phase_ptr->output_tensor;
            tdi.device_id = phase_ptr->device_id;
            tensor_list.push_back(tdi);
            ret = true;
        }
    }

    lock_.unlock();
    return ret;
}

bool GpuInfGlobalData::GetSourceTensors(vector<SourceTensor> &tensor_list, int phase_id)
{
    bool ret = false;
    tensor_list.clear();
    lock_.lock();

    for (const WorkerData *worker_data : worker_list_)
    {
        const PhaseData *phase_ptr = GetPhase(worker_data, phase_id);
        if (phase_ptr != nullptr)
        {
            for (const auto &source_tensor : phase_ptr->source_tensors) {
                tensor_list.push_back(source_tensor);
            }
            ret = true;
        }
    }

    lock_.unlock();
    return ret;
}

const DeviceTensor* GpuInfGlobalData::GetInputTensor(int worker_id, int phase_id)
{
    const DeviceTensor *tensor = nullptr;
    lock_.lock();
    const WorkerData *worker_data = worker_list_[worker_id];
    const PhaseData *phase_ptr = GetPhase(worker_data, phase_id);
    if (phase_ptr != nullptr) {
        tensor = phase_ptr->input_tensor;
    }
    lock_.unlock();
    return tensor;
}

GpuInfGlobalData::PhaseData* GpuInfGlobalData::AddPhase(
    WorkerData *worker_data, int phase_id, int device_id)
{
    auto iter = worker_data->phase_map.find(phase_id);
    PhaseData *phase_ptr = nullptr;
    if (iter == worker_data->phase_map.end())
    {
        phase_ptr = new PhaseData;
        worker_data->phase_list.push_back(phase_ptr);
        worker_data->phase_map[phase_id] = phase_ptr;
    }
    else
    {
        phase_ptr = iter->second;
    }

    phase_ptr->phase = phase_id;
    phase_ptr->device_id = device_id;
    return phase_ptr;
}

const GpuInfGlobalData::PhaseData* GpuInfGlobalData::GetPhase(
    const WorkerData *worker_data, int phase_id) const
{
    auto iter = worker_data->phase_map.find(phase_id);
    return iter == worker_data->phase_map.end() ? nullptr : iter->second;
}

TRANSFORMER_END
INFER_FLOW_END

