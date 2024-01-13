#pragma once

#include "sslib/thread.h"
#include "device_tensor.h"
#include "host_float_buffer.h"

INFER_FLOW_BEGIN

using std::vector;

class DeviceTensorBuilder : public sslib::Thread
{
public:
    struct Task
    {
        int id = 0;
        DeviceTensorEx *target = nullptr;
        vector<DeviceTensorEx*> target_list;
        const HostTensor *source = nullptr;
        vector<const HostTensor*> source_list;
        ElementType device_data_type = ElementType::F16;
        bool force_dequant = false;
        bool be_trans = false;
        int cx_in_merging = 0;
        TensorPartitionType partition_type = TensorPartitionType::BY_COL;
        float delta_ratio = 0;

        //the "p_" prefix means "parameter"
        void Set(DeviceTensorEx *p_target, const HostTensor *p_source,
            ElementType p_device_data_type, bool p_force_dequant, bool p_be_trans,
            TensorPartitionType p_partition_type = TensorPartitionType::BY_COL)
        {
            this->target = p_target;
            this->source = p_source;
            this->device_data_type = p_device_data_type;
            this->force_dequant = p_force_dequant;
            this->be_trans = p_be_trans;
            this->partition_type = p_partition_type;
        }

        void SetSourceTarget(DeviceTensorEx *p_target, const HostTensor *p_source)
        {
            this->target = p_target;
            this->source = p_source;
            this->source_list.clear();
        }

        void SetSourceTarget(DeviceTensorEx *p_target,
            const vector<const HostTensor*> &p_source_list)
        {
            this->target = p_target;
            this->source_list = p_source_list;
            this->source = nullptr;
        }

        void SetSourceTarget(const vector<DeviceTensorEx*> &p_target_list,
            const HostTensor *p_source)
        {
            this->target_list.clear();
            this->target_list.assign(p_target_list.begin(), p_target_list.end());
            this->source = p_source;
            this->source_list.clear();
        }
    };

public:
    DeviceTensorBuilder() {};
    virtual ~DeviceTensorBuilder();
    void Clear();

    bool Init(int device_id, int aux_buffer_capacity, int aux_tensor_size);
    bool Init(const vector<int> &devices, int aux_buffer_capacity, int aux_tensor_size);

    void AddTask(const Task &t);
    void ClearTasks();

    // main procedure for the thread
    virtual void Run() override;

    virtual void CancelThread() override;

    bool HasError() {
        return has_error_;
    }

    bool Build(Task &task);

    static bool Build(DeviceTensorEx &target, const HostTensor &cpu_tensor,
        HostHalfBuffer &aux_buffer, ElementType device_data_type, bool force_dequant,
        bool be_trans, DeviceTensor *aux_tensor, float delta_ratio = 0);
    static bool Build(vector<DeviceTensorEx*> &targets, const HostTensor &cpu_tensor,
        HostHalfBuffer &aux_buffer, const vector<int> &devices,
        ElementType device_data_type, bool force_dequant, bool be_trans,
        TensorPartitionType partition_type, DeviceTensor *aux_tensor);

protected:
    int device_id_ = 0;
    vector<int> devices_;
    vector<Task> tasks_;
    HostHalfBuffer aux_buffer_;
    DeviceTensor aux_tensor_;
    bool has_error_ = false;
    int error_index_ = -1;
    bool is_cancelled_ = false;

protected:
    bool Merge(Task &task);

    static bool Build_Quant(DeviceTensorEx &target, const HostTensor &cpu_tensor,
        HostHalfBuffer &aux_buffer, ElementType quant_type, bool be_trans,
        DeviceTensor *aux_tensor, float delta_ratio = 0);

    static bool BuildDeltaTensor(DeviceTensorEx &target, vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz, float delta_ratio);

    static bool Build_Q8_GlobalLinear(DeviceTensorEx &target,
        vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz, float delta_ratio);
    static bool Build_Q8_Log(DeviceTensorEx &target,
        const vector<inferflow_fp16> &host_array, int cx, int cy, int cz);
};

INFER_FLOW_END

