#include "device_tensor_builder.h"
#include "common/cuda_util.h"
#include "common/quantization.h"
#include "device_tensor_util.h"

INFER_FLOW_BEGIN

using namespace std;
using namespace sslib;

DeviceTensorBuilder::~DeviceTensorBuilder()
{
    Clear();
}

void DeviceTensorBuilder::Clear()
{
    aux_buffer_.Clear();
    aux_tensor_.Clear();
}

bool DeviceTensorBuilder::Init(int device_id, int aux_buffer_capacity, int aux_tensor_size)
{
    device_id_ = device_id;
    aux_buffer_.New(aux_buffer_capacity);
    if (aux_tensor_size > 0) {
        aux_tensor_.New(ElementType::F16, aux_tensor_size);
    }

    return true;
}

bool DeviceTensorBuilder::Init(const vector<int> &devices,
    int aux_buffer_capacity, int aux_tensor_size)
{
    this->devices_ = devices;
    aux_buffer_.New(aux_buffer_capacity);
    if (aux_tensor_size > 0) {
        aux_tensor_.New(ElementType::F16, aux_tensor_size);
    }

    return true;
}

void DeviceTensorBuilder::AddTask(const Task &t)
{
    tasks_.push_back(t);
}

void DeviceTensorBuilder::ClearTasks()
{
    tasks_.clear();
}

void DeviceTensorBuilder::Run()
{
    bool ret = true;
    if (device_id_ >= 0) {
        CudaUtil::SetDevice(device_id_);
    }

    int task_num = (int)tasks_.size();
    for (int idx = 0; ret && idx < task_num && ret && !is_cancelled_; idx++)
    {
        auto &task = tasks_[idx];
        ret = Build(task);
        if (!ret) {
            error_index_ = idx;
        }
    }

    has_error_ = !ret;
}

void DeviceTensorBuilder::CancelThread()
{
    is_cancelled_ = true;
}

bool DeviceTensorBuilder::Build(Task &task)
{
    bool ret = true;
    if (task.target != nullptr)
    {
        if (!task.source_list.empty())
        {
            ret = Merge(task);
        }
        else
        {
            ret = Build(*task.target, *task.source, aux_buffer_, task.device_data_type,
                task.force_dequant, task.be_trans, &aux_tensor_, task.delta_ratio);
        }
    }
    else
    {
        ret = Build(task.target_list, *task.source, aux_buffer_, devices_,
            task.device_data_type, task.force_dequant, task.be_trans,
            task.partition_type, &aux_tensor_);
    }

    //if (task.id == 10000 + 0 && task.target->tensor != nullptr)
    //{
    //    const auto *tensor = task.target->tensor;
    //    LogKeyInfo("Tensor (%d, %d, %d)", tensor->ne[0], tensor->ne[1], tensor->ne[2]);
    //    task.target->tensor->Print(cout, 8, 8, 8, "attention_norm:\n") << endl;
    //}

    return ret;
}

//static
bool DeviceTensorBuilder::Build(DeviceTensorEx &target, const HostTensor &cpu_tensor,
    HostHalfBuffer &aux_buffer, ElementType device_data_type, bool force_dequant,
    bool be_trans, DeviceTensor *aux_tensor, float delta_ratio)
{
    bool ret = true;
    int cx = cpu_tensor.ne[0], cy = cpu_tensor.ne[1], cz = cpu_tensor.ne[2];
    bool is_gpu_quant = TensorCommon::IsQuantType(device_data_type);

    int cx_new = be_trans ? cy : cx;
    int cy_new = be_trans ? cx : cy;

    if (is_gpu_quant && !force_dequant)
    {
        ret = Build_Quant(target, cpu_tensor, aux_buffer, device_data_type,
            be_trans, aux_tensor, delta_ratio);
        //device_tensor->Print(cout, 8, 8, 8, "device_tensor:\n") << endl;
    }
    else if (!be_trans && device_data_type == ElementType::F16
        && device_data_type == cpu_tensor.data_type)
    {
        const inferflow_fp16 *src_data = cpu_tensor.data_f16();
        ret = target.tensor->FromHost(src_data, cx_new, cy_new, cz);
    }
    else
    {
        //vector<half> host_array;
        //DeviceTensorUtil::GetFP16List(host_array, cpu_tensor, be_trans);
        DeviceTensorUtil::GetFP16List(aux_buffer, cpu_tensor, be_trans);

        //ret = target.FromHost(host_array.data(), cx_new, cy_new, cz);
        ret = target.tensor->FromHost(aux_buffer.data(), cx_new, cy_new, cz);
        //target.tensor->Print(cout, 8, 8, 8, "device_tensor:\n") << endl;
    }

    if (!ret)
    {
        LogError("Error occurred in building the device tensor (%d, %d, %d)",
            cx_new, cy_new, cz);
    }

    return ret;
}

//static
bool DeviceTensorBuilder::Build(vector<DeviceTensorEx*> &targets,
    const HostTensor &cpu_tensor, HostHalfBuffer &aux_buffer,
    const vector<int> &devices, ElementType device_data_type,
    bool force_dequant, bool be_trans, TensorPartitionType partition_type,
    DeviceTensor *aux_tensor)
{
    (void)aux_tensor;
    bool ret = true;
    int cx = cpu_tensor.ne[0], cy = cpu_tensor.ne[1], cz = cpu_tensor.ne[2];
    bool is_gpu_quant = TensorCommon::IsQuantType(device_data_type);
    int target_num = min((int)targets.size(), (int)devices.size());

    if (cz != 1)
    {
        LogError("Only one or two dimensional tensors are supported so far.");
        return false;
    }

    int cx_new = be_trans ? cy : cx;
    int cy_new = be_trans ? cx : cy;

    if (is_gpu_quant && !force_dequant)
    {
        //if (is_gpu_quant)
        //{
        //    LogKeyInfo("tensor (%d, %d, %d), is_gpu_quant: yes, partition_type: %d",
        //        cx, cy, cz, (int)partition_type);
        //}

        if (be_trans)
        {
            LogError("The quant version has not been impelemnted yet.");
            return false;
        }

        DeviceTensor target_tensor;
        DeviceTensorEx target;
        target.tensor = &target_tensor;
        ret = Build_Quant(target, cpu_tensor, aux_buffer, device_data_type,
            be_trans, aux_tensor);
        if (partition_type == TensorPartitionType::BY_COL)
        {
            int target_cy = cy_new / target_num;
            int bytes = (int)TensorCommon::ByteCount(device_data_type, cx_new * target_cy);
            for (int idx = 0; ret && idx < target_num; idx++)
            {
                CudaUtil::SetDevice(devices[idx]); //!!!

                const uint8_t *src_data = ((const uint8_t*)target.tensor->data) + bytes * idx;
                targets[idx]->tensor->New(device_data_type, cx_new, target_cy);
                targets[idx]->tensor->CopyFromDevice(src_data, bytes);
            }
        }
        else if (partition_type == TensorPartitionType::BY_ROW)
        {
            int target_cx = cx_new / target_num;
            int src_row_bytes = (int)TensorCommon::ByteCount(device_data_type, cx_new);
            int target_row_bytes = src_row_bytes / target_num;
            for (int idx = 0; ret && idx < target_num; idx++)
            {
                CudaUtil::SetDevice(devices[idx]); //!!!

                targets[idx]->tensor->New(device_data_type, target_cx, cy_new);
                for (int row_idx = 0; row_idx < cy_new; row_idx++)
                {
                    const uint8_t *src_data = ((const uint8_t*)target.tensor->data)
                        + src_row_bytes * row_idx + target_row_bytes * idx;
                    void *target_data = targets[idx]->tensor->RowData(row_idx);
                    CudaUtil::DeviceToDeviceMemcpy(target_data, src_data, target_row_bytes);
                }
            }
        }
        else
        {
            LogError("The quant version has not been impelemnted yet.");
            return false;
        }
    }
    else if (partition_type == TensorPartitionType::DUP)
    {
        DeviceTensorUtil::GetFP16List(aux_buffer, cpu_tensor, be_trans);
        for (int idx = 0; ret && idx < target_num; idx++)
        {
            CudaUtil::SetDevice(devices[idx]); //!!!
            ret = targets[idx]->tensor->FromHost(aux_buffer.data(), cx_new, cy_new, cz);
        }
    }
    else if (cy_new == 1)
    {
        DeviceTensorUtil::GetFP16List(aux_buffer, cpu_tensor, be_trans);
        for (int idx = 0; ret && idx < target_num; idx++)
        {
            CudaUtil::SetDevice(devices[idx]); //!!!
            int target_cx = cx_new / target_num;
            int offset = idx * target_cx;
            //LogKeyInfo("idx: %d, target_cx: %d, offset: %d", idx, target_cx, offset);
            ret = targets[idx]->tensor->FromHost(aux_buffer.data() + offset,
                target_cx, cy_new, cz);
        }
    }
    else if (partition_type == TensorPartitionType::BY_ROW) //rowwise
    {
        if (be_trans)
        {
            DeviceTensorUtil::GetFP16List(aux_buffer, cpu_tensor, be_trans);
            for (int idx = 0; ret && idx < target_num; idx++)
            {
                CudaUtil::SetDevice(devices[idx]); //!!!

                int target_cy = cy_new / target_num;
                int offset = cx_new * (idx * target_cy);
                //LogKeyInfo("idx: %d, cx: %d, cy: %d, offset: %d*%d*%d, target_cy: %d",
                //    idx, cx_new, cy_new, cx_new, idx, target_cy, target_cy);
                ret = targets[idx]->tensor->FromHost(aux_buffer.data() + offset,
                    cx_new, target_cy, cz);
            }
        }
        else
        {
            int target_cx = cx_new / target_num;
            for (int idx = 0; ret && idx < target_num; idx++)
            {
                int start_col = target_cx * idx;
                DeviceTensorUtil::GetFP16List(aux_buffer, cpu_tensor, start_col, target_cx);

                CudaUtil::SetDevice(devices[idx]); //!!!
                ret = targets[idx]->tensor->FromHost(aux_buffer.data(), target_cx, cy_new, cz);
            }
        }
    }
    else //if (be_trans) //colwise and be_trans
    {
        int src_part_rows = cy / target_num;
        for (int idx = 0; ret && idx < target_num; idx++)
        {
            CudaUtil::SetDevice(devices[idx]); //!!!

            int src_start_row = idx * src_part_rows;
            DeviceTensorUtil::GetFP16List(aux_buffer, cpu_tensor,
                be_trans, src_start_row, src_part_rows);
            ret = targets[idx]->tensor->FromHost(aux_buffer.data(),
                be_trans ? src_part_rows : cx_new,
                be_trans ? cy_new : cy_new / target_num,
                cz);
            //LogKeyInfo("idx: %d, source: (%d, %d), target: (%d, %d, %d)",
            //    idx, cx, cy, src_part_rows, cy_new, cz);
        }

        //if (cx < 10000 && cy < 10000) {
        //    targets[0]->tensor->Print(cout, 8, 8, 8, "target-0:") << endl;
        //    exit(0);
        //}
    }
    /*else //colwise and !be_trans
    {
        int target_cx = cx_new / target_num;
        for (int idx = 0; ret && idx < target_num; idx++)
        {
            int start_col = target_cx * idx;
            DeviceTensorUtil::GetFP16List(aux_buffer, cpu_tensor, start_col, target_cx);

            CudaUtil::SetDevice(devices[idx]); //!!!
            ret = targets[idx]->tensor->FromHost(aux_buffer.data(), target_cx, cy_new, cz);
        }
    }*/

    return ret;
}

bool DeviceTensorBuilder::Merge(Task &task)
{
    bool ret = true;
    auto &target = *task.target;
    const auto &cpu_tensor0 = *task.source_list[0];
    int cx = cpu_tensor0.ne[0], cy = cpu_tensor0.ne[1], cz = 1;
    int cx_new = task.be_trans ? cy : cx;
    int cy_new = task.be_trans ? cx : cy;
    int k = (int)task.source_list.size();

    if (task.be_trans)
    {
        int mcx = task.cx_in_merging;
        int mcy = cx_new / mcx * cy_new * cz;

        target.tensor->New(task.device_data_type, k * mcx, mcy);
        for (int idx = 0; ret && idx < k; idx++)
        {
            const auto &src_tensor = *task.source_list[idx];
            ret = DeviceTensorUtil::GetFP16List(aux_buffer_, src_tensor, task.be_trans);

            int bytes = cx * cy * (int)sizeof(half);
            CudaUtil::HostToDeviceMemcpy(aux_tensor_.data_f16(), aux_buffer_.data(), bytes);
            aux_tensor_.SetStructure(mcx, mcy, 1);
            //aux_tensor_.Print(cout, 8, 2, 1, "aux_tensor:\n") << endl;
            TensorOpr::AssignColumns(*target.tensor, aux_tensor_, 0, 1, idx * mcx, mcx);
        }

        //target.tensor->Print(cout, 8, 8, 1, "target_tensor:\n") << endl;
        TensorOpr::Reshape(*target.tensor, 2, k * cx_new, cy_new);
        //target.tensor->Print(cout, 8, 8, 1, "target_tensor:\n") << endl;
    }
    else
    {
        target.tensor->New(task.device_data_type, cx_new, k * cy_new);
        for (int idx = 0; ret && idx < k; idx++)
        {
            const auto &src_tensor = *task.source_list[idx];
            ret = DeviceTensorUtil::GetFP16List(aux_buffer_, src_tensor, task.be_trans);

            int bytes = (int)TensorCommon::ByteCount(task.device_data_type, cx * cy);
            uint8_t *target_data = (uint8_t*)target.tensor->data + idx * bytes;
            CudaUtil::HostToDeviceMemcpy(target_data, aux_buffer_.data(), bytes);
        }
        //target.tensor->Print(cout, 8, 8, 1, "target_tensor:\n") << endl;
    }

    return ret;
}

//static
bool DeviceTensorBuilder::Build_Quant(DeviceTensorEx &target, const HostTensor &cpu_tensor,
    HostHalfBuffer &aux_buffer, ElementType quant_type, bool be_trans,
    DeviceTensor *aux_tensor, float delta_ratio)
{
    (void)aux_buffer;
    bool ret = true;
    int cx = cpu_tensor.ne[0], cy = cpu_tensor.ne[1], cz = cpu_tensor.ne[2];
    int cx_new = be_trans ? cy : cx;
    int cy_new = be_trans ? cx : cy;

    vector<inferflow_fp16> host_array;
    DeviceTensorUtil::GetFP16List(host_array, cpu_tensor, be_trans);

    if (!be_trans && delta_ratio >= 0.00001f) {
        BuildDeltaTensor(target, host_array, cx_new, cy_new, cz, delta_ratio);
    }

    //cout << "host_array[0]: " << (float)host_array[0] << endl;
    if (aux_tensor != nullptr)
    {
        aux_tensor->SetStructure(cx_new, cy_new, cz);
        int bytes = sizeof(half) * (int)host_array.size();
        ret = aux_tensor->CopyFromHost(host_array.data(), bytes);

        if (ret)
        {
            target.tensor->New(quant_type, cx_new, cy_new, cz);
            ret = TensorOpr::Quantize(*target.tensor, *aux_tensor);
        }

        if (ret) {
            return ret;
        }
    }

    switch (quant_type)
    {
    case ElementType::Q8_GL:
        ret = Build_Q8_GlobalLinear(target, host_array, cx_new, cy_new, cz, delta_ratio);
        break;
    case ElementType::Q8_B32T1:
        ret = DeviceTensorUtil::BuildTensor_Q8_B32T1(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q8_B32T2:
        ret = DeviceTensorUtil::BuildTensor_Q8_B32T2(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q8_LOG:
        ret = Build_Q8_Log(target, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q6_B64T1:
        ret = DeviceTensorUtil::BuildTensor_Q6_B64T1(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q5_B32T1:
        ret = DeviceTensorUtil::BuildTensor_Q5_B32T1(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q5_B64T1:
        ret = DeviceTensorUtil::BuildTensor_Q5_B64T1(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q4_B16:
        ret = DeviceTensorUtil::BuildQ4B16Tensor(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q4_B32T1A:
        ret = DeviceTensorUtil::BuildTensor_Q4_B32T1A(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q4_B32T1B:
        ret = DeviceTensorUtil::BuildTensor_Q4_B32T1B(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q4_B64T1:
        ret = DeviceTensorUtil::BuildTensor_Q4_B64T1(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q3H_B64T1:
        ret = DeviceTensorUtil::BuildTensor_Q3H_B64T1(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q3_B32T1A:
        ret = DeviceTensorUtil::BuildTensor_Q3_B32T1A(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q3_B32T1B:
        ret = DeviceTensorUtil::BuildTensor_Q3_B32T1B(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q2_B32T1A:
        ret = DeviceTensorUtil::BuildTensor_Q2_B32T1A(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    case ElementType::Q2_B32T1B:
        ret = DeviceTensorUtil::BuildTensor_Q2_B32T1B(*target.tensor, host_array, cx_new, cy_new, cz);
        break;
    default:
        LogError("Quant type %d is not supported.", (int)quant_type);
        break;
    }

    return ret;
}

bool DeviceTensorBuilder::BuildDeltaTensor(DeviceTensorEx &target,
    vector<inferflow_fp16> &host_array, int cx, int cy, int cz,
    float delta_ratio)
{
    TensorStat stat;
    //int block_size = 32;
    DeviceTensorUtil::CalculateStat(stat, host_array, delta_ratio, 0);

    auto &params = target.linear_quant_params;
    params.z = 0; //z: zero-point
    params.scale1 = max(0.001f, stat.soft_max - params.z) / 127;
    params.scale2 = max(0.001f, params.z - stat.soft_min) / 127;

    int size = cx * cy * cz, size2 = (int)host_array.size();
    if (size != size2)
    {
        LogError("Inconsistent tensor size: %d vs. %d", size, size2);
        return false;
    }

    if (delta_ratio < 0.00001f || target.delta == nullptr) {
        return true;
    }

    vector<SparseMatrixCell> delta_cells;
    int rows = cy * max(1, cz);
    for (int row_idx = 0; row_idx < rows; row_idx++)
    {
        for (int col_idx = 0; col_idx < cx; col_idx++)
        {
            int offset = row_idx * cx + col_idx;
            float value = host_array[offset];
            bool b1 = value < stat.soft_min;
            bool b2 = value > stat.soft_max;
            if (b1 || b2)
            {
                float delta_value = value; //b1 ? value - stat.soft_min : value - stat.soft_max;
                SparseMatrixCell new_cell((uint16_t)row_idx, (uint16_t)col_idx, delta_value);
                delta_cells.push_back(new_cell);
                host_array[offset] = value - delta_value;
                //host_array[offset] = 0;
            }
        }
    }

    DeviceTensorUtil::CalculateStat(stat, host_array, delta_ratio, 32);
    for (const auto &excluded_item : stat.excluded)
    {
        int row_idx = excluded_item.id / cx;
        int col_idx = excluded_item.id % cx;
        SparseMatrixCell new_cell((uint16_t)row_idx, (uint16_t)col_idx, excluded_item.weight);
        delta_cells.push_back(new_cell);
        host_array[excluded_item.id] = (inferflow_fp16)0;
    }

    /*int lower = 100, upper = 0, sum = 0;
    IdWeight<int> max_bucket(-1, 0);
    for (int bucket_idx = 0; bucket_idx < (int)stat.block_mean_histogram.size(); bucket_idx++)
    {
        int num = stat.block_mean_histogram[bucket_idx];
        sum += num;
        if (num > 0 && lower > bucket_idx) {
            lower = bucket_idx;
        }
        if (num > 0 && upper < bucket_idx) {
            upper = bucket_idx;
        }

        if (max_bucket.weight < num) {
            max_bucket.Set(bucket_idx, num);
        }
    }

    LogKeyInfo("tensor: (%d, %d, %d), min_max_avg: (%.3f, %.3f, %.3f), lower-upper bucket: [%d, %d], width: %d, max_bucket: (%d, %d, %.3f%%), #excluded: %d",
        cx, cy, cz, stat.min_value, stat.max_value, stat.avg_value, lower, upper, upper - lower + 1, max_bucket.id, max_bucket.weight,
        max_bucket.weight * 100.0f / sum, (int)stat.excluded.size());*/

    std::sort(delta_cells.begin(), delta_cells.end());
    target.delta->SetSortedCells(delta_cells, cx, rows);

    return true;
}

bool DeviceTensorBuilder::Build_Q8_GlobalLinear(DeviceTensorEx &target,
    vector<inferflow_fp16> &host_array, int cx, int cy, int cz,
    float delta_ratio)
{
    TensorStat stat;
    DeviceTensorUtil::CalculateStat(stat, host_array, delta_ratio);

    auto &params = target.linear_quant_params;
    params.z = 0; //z: zero-point
    params.scale1 = max(0.001f, stat.soft_max - params.z) / 127;
    params.scale2 = max(0.001f, params.z - stat.soft_min) / 127;

    int size = cx * cy * cz, size2 = (int)host_array.size();
    if (size != size2)
    {
        LogError("Inconsistent tensor size: %d vs. %d", size, size2);
        return false;
    }

    if (delta_ratio >= 0.00001f && target.delta != nullptr)
    {
        vector<SparseMatrixCell> delta_cells;
        int rows = cy * max(1, cz);
        if (rows > 0xFFFF || cx > 0xFFFF)
        {
            LogError("Number of rows or columns should not be larger than %d", 0xFFFF);
            return false;
        }

        for (int row_idx = 0; row_idx < rows; row_idx++)
        {
            for (int col_idx = 0; col_idx < cx; col_idx++)
            {
                int offset = row_idx * cx + col_idx;
                float value = host_array[offset];
                bool b1 = value < stat.soft_min;
                bool b2 = value > stat.soft_max;
                if (b1 || b2)
                {
                    float delta_value = b1 ? value - stat.soft_min : value - stat.soft_max;
                    SparseMatrixCell new_cell((uint16_t)row_idx, (uint16_t)col_idx, delta_value);
                    //SparseMatrixCell new_cell(row_idx, col_idx, value);
                    delta_cells.push_back(new_cell);
                    //host_array[offset] = 0;
                }
            }
        }

        std::sort(delta_cells.begin(), delta_cells.end());
        target.delta->SetSortedCells(delta_cells, cx, rows);
    }

    vector<uint8_t> quant_array;
    Quantization::Quantize_Q8_Linear(quant_array, host_array, params);

    target.tensor->New(ElementType::Q8_GL, cx, cy, cz);
    int bytes = (int)(size * sizeof(uint8_t));
    bool ret = target.tensor->CopyFromHost(quant_array.data(), bytes);
    Macro_RetFalseIf(!ret);

    if (target.quant_map != nullptr)
    {
        vector<uint8_t> uint8_array(256, (uint8_t)0);
        for (int idx = 0; idx < 256; idx++)
        {
            uint8_array[idx] = (uint8_t)idx;
        }
        vector<inferflow_fp16> half_array;
        Quantization::Dequantize_Q8_Linear(half_array, uint8_array, params);

        target.quant_map->New(ElementType::F16, 256);
        bytes = (int)(half_array.size() * sizeof(half));
        target.quant_map->CopyFromHost((const half*)half_array.data(), bytes);
    }

    return ret;
}

bool DeviceTensorBuilder::Build_Q8_Log(DeviceTensorEx &target,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    auto &params = target.log_quant_params;
    params.base = 1.1f;
    params.scale = 1000;
    params.start = 10;

    vector<uint8_t> quant_array;
    Quantization::Quantize_Q8_Log(quant_array, host_array, params);

    int size = cx * cy * cz, size2 = (int)host_array.size();
    if (size != size2)
    {
        LogError("Inconsistent tensor size: %d vs. %d", size, size2);
        return false;
    }

    target.tensor->New(ElementType::Q8_GL, cx, cy, cz);
    int bytes = (int)(size * sizeof(uint8_t));
    bool ret = target.tensor->CopyFromHost(quant_array.data(), bytes);
    Macro_RetFalseIf(!ret);

    if (target.quant_map != nullptr)
    {
        vector<uint8_t> uint8_array(256, (uint8_t)0);
        for (int idx = 0; idx < 256; idx++)
        {
            uint8_array[idx] = (uint8_t)idx;
        }
        vector<inferflow_fp16> half_array;
        Quantization::Dequantize_Q8_Log(half_array, uint8_array, params);

        target.quant_map->New(ElementType::F16, 256);
        bytes = (int)(half_array.size() * sizeof(half));
        target.quant_map->CopyFromHost((const half*)half_array.data(), bytes);
    }

    return ret;
}

INFER_FLOW_END
