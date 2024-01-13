#include "device_tensor_util.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "sslib/log.h"
#include "tensor_util.h"
#include "common/quant_cuda.h"

namespace inferflow
{

using namespace std;
using namespace sslib;

//static
bool DeviceTensorUtil::LoadFromJson(map<int, DeviceTensorOprNode> &node_map,
    BlockedAllocator<DeviceTensor> &heap, const string &file_path)
{
    node_map.clear();

    BinFileStream reader;
    bool ret = reader.OpenForRead(file_path);
    if (!ret) {
        return false;
    }

    JsonParser jparser;
    jparser.Init();

    TaskMonitor tm(1);
    uint32_t count = 0;
    wstring tensor_str, line_str;
    while(reader.GetLine(line_str))
    {
        if (line_str.empty()) {
            continue;
        }

        if (line_str[0] == L' ')
        {
            tensor_str += line_str;
            continue;
        }

        if (!tensor_str.empty())
        {
            DeviceTensorOprNode node;
            ret = JsonToTensorOprNode(node, tensor_str, heap, jparser);
            if (!ret) {
                return false;
            }

            //LogKeyInfo("node id: %d", node.target->id);
            node_map[node.target->id] = node;

            count++;
            tm.Progress(count);
        }

        tensor_str.clear();
        tensor_str += line_str;
    }

    if (!tensor_str.empty())
    {
        DeviceTensorOprNode node;
        ret = JsonToTensorOprNode(node, tensor_str, heap, jparser);
        if (!ret) {
            return false;
        }

        //LogKeyInfo("node id: %d", node.target->id);
        node_map[node.target->id] = node;

        count++;
        tm.Progress(count);
    }

    tm.End();
    return ret;
}

//static
bool DeviceTensorUtil::JsonToTensorOprNode(DeviceTensorOprNode &node, const wstring &json_str,
    BlockedAllocator<DeviceTensor> &heap, JsonParser &jparser)
{
    JsonDoc jdoc;
    bool ret = jparser.Parse(jdoc, json_str);
    if (!ret) {
        LogError("Invalid JSON format");
        return false;
    }

    DeviceTensor *tensor = heap.New(1);
    node.target = tensor;

    JsonObject jobj = jdoc.GetJObject();

    JsonArray jarray;
    jobj.GetFieldValue(jarray, L"ne", jdoc);
    if (jarray.size >= 3)
    {
        int ne0 = jarray.items[0].GetIntValue();
        int ne1 = jarray.items[1].GetIntValue();
        int ne2 = jarray.items[2].GetIntValue();

        JsonArray item_array;
        jobj.GetFieldValue(item_array, L"data", jdoc);
        if ((int)item_array.size != ne0 * ne1 * ne2)
        {
            LogError("Invalid tensor data");
            return false;
        }

        vector<half> host_vector;
        host_vector.reserve(item_array.size);
        for (int item_idx = 0; item_idx < (int)item_array.size; item_idx++)
        {
            float float_value = item_array.items[item_idx].GetFloatValue();
            host_vector.push_back((half)float_value);
        }

        tensor->FromHost(host_vector.data(), ne0, ne1, ne2);
    }

    //
    jobj.GetFieldValue(tensor->id, L"id", jdoc);
    jobj.GetFieldValue(tensor->dim, L"dim", jdoc);
    jobj.GetFieldValue(node.opr, L"op", jdoc);

    jobj.GetFieldValue(jarray, L"src", jdoc);
    if (jarray.size >= 2)
    {
        node.sources[0] = jarray.items[0].GetIntValue();
        node.sources[1] = jarray.items[1].GetIntValue();
    }

    jobj.GetFieldValue(jarray, L"perf", jdoc);
    if (jarray.size >= 3)
    {
        node.perf_time = jarray.items[2].GetIntValue() / 1000.0f;
    }

    return true;
}

// Copy to host and verify equivalence.
bool DeviceTensorUtil::CopyAndCompare(const DeviceTensor &C1, const DeviceTensor &C2,
    float diff_ratio_threshold, bool is_study_mode)
{
    if (C1.size != C2.size) {
        LogWarning("Inconsistent size: %d vs. %d", C1.size, C2.size);
        return false;
    }

    int size = C1.size;
    std::vector<float> host1(size, 0);
    std::vector<float> host2(size, 0);

    //TaskMonitor tm;
    //cout << "Copying C1 to host (size = " << size  << ")..." << endl;
    bool ret = C1.CopyToHost(host1);
    //tm.ShowElapsedTime(L"Copying C1 to host");
    if (!ret) {
        std::cerr << "Failed to copy C1." << std::endl;
        return false;
    }

    //cout << "Copying C2 to host (size = " << size  << ")..." << endl;
    ret = C2.CopyToHost(host2);
    if (!ret) {
        std::cerr << "Failed to copy C2." << std::endl;
        return false;
    }

    ret = TensorUtil::Compare(host1, host2, diff_ratio_threshold, is_study_mode);
    return ret;
}

//static
bool DeviceTensorUtil::GetFP16List(vector<inferflow_fp16> &value_list,
    const HostTensor &tensor, bool be_transpose)
{
    bool ret = true;
    value_list.clear();

    if (TensorCommon::IsQuantType(tensor.data_type))
    {
        ret = GetFP16List_Quant(value_list, tensor, be_transpose);
    }
    else
    {
        switch (tensor.data_type)
        {
        case ElementType::F32:
            ret = GetFP16List_F32(value_list, tensor, be_transpose);
            break;
        case ElementType::F16:
            ret = GetFP16List_F16(value_list, tensor, be_transpose);
            break;
        default:
            LogError("Invalid tensor data type: %d", tensor.data_type);
            ret = false;
            break;
        }
    }

    //for (int idx = 0; idx < (int)value_list.size(); idx++)
    //{
    //    if (!isfinite(value_list[idx])) {
    //        value_list[idx] = 0;
    //    }
    //}

    return ret;
}

//static
bool DeviceTensorUtil::GetFP16List(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, bool be_transpose, int start_row, int rows)
{
    bool ret = true;
    if (TensorCommon::IsQuantType(tensor.data_type))
    {
        ret = GetFP16List_Quant(value_buffer, tensor, be_transpose, start_row, rows);
    }
    else
    {
        switch (tensor.data_type)
        {
        case ElementType::F32:
            ret = GetFP16List_F32(value_buffer, tensor, be_transpose, start_row, rows);
            break;
        case ElementType::F16:
            ret = GetFP16List_F16(value_buffer, tensor, be_transpose, start_row, rows);
            break;
        default:
            LogError("Invalid tensor data type: %d", tensor.data_type);
            ret = false;
            break;
        }
    }

    return ret;
}

bool DeviceTensorUtil::GetFP16List(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, int start_col, int cols)
{
    bool ret = true;
    if (TensorCommon::IsQuantType(tensor.data_type))
    {
        ret = GetFP16List_Quant(value_buffer, tensor, start_col, cols);
    }
    else
    {
        switch (tensor.data_type)
        {
        case ElementType::F32:
            ret = GetFP16List_F32(value_buffer, tensor, start_col, cols);
            break;
        case ElementType::F16:
            ret = GetFP16List_F16(value_buffer, tensor, start_col, cols);
            break;
        default:
            LogError("Invalid tensor data type: %d", tensor.data_type);
            ret = false;
            break;
        }
    }

    return ret;
}

//static
void DeviceTensorUtil::CalculateStat(TensorStat &stat, const vector<inferflow_fp16> &value_list,
    float ratio, int block_size)
{
    stat.Clear();
    if (value_list.empty()) {
        return;
    }

    int value_count = (int)value_list.size();
    double sum = 0;
    stat.min_value = value_list[0];
    stat.max_value = value_list[1];
    for (float v : value_list)
    {
        if (stat.min_value > v) {
            stat.min_value = v;
        }
        if (stat.max_value < v) {
            stat.max_value = v;
        }
        sum += v;
    }

    stat.avg_value = float(sum / value_count);

    const int bucket_count = 100;
    stat.histogram.resize(bucket_count, 0);
    float global_width = stat.max_value - stat.min_value;
    float width = max(0.00001f, global_width / bucket_count);
    for (float v : value_list)
    {
        int bucket_id = (int)((v - stat.min_value) / width);
        bucket_id = min(bucket_count - 1, max(0, bucket_id));
        stat.histogram[bucket_id]++;
    }

    if (block_size >= 8)
    {
        stat.block_mean_histogram.resize(bucket_count, 0);
        int block_num = value_count / block_size;
        for (int block_idx = 0; block_idx < block_num; block_idx++)
        {
            int v_idx = block_idx * block_size;
            sum = 0;
            IdWeight<float> min_item(v_idx, (float)value_list[v_idx]);
            IdWeight<float> max_item(v_idx, (float)value_list[v_idx]);
            for (v_idx++; v_idx < (block_idx + 1) * block_size; v_idx++)
            {
                const float v = (float)value_list[v_idx];
                sum += v;
                if (min_item.weight > v) {
                    min_item.Set(v_idx, v);
                }
                if (max_item.weight < v) {
                    max_item.Set(v_idx, v);
                }
            }

            float avg_value = float(sum / block_size);
            float block_len = max_item.weight - min_item.weight;
            IdWeight<float> excluded_item(UINT32_MAX, 0);
            if (block_len >= 0.7f * global_width)
            {
                bool b1 = max_item.weight - avg_value > avg_value - min_item.weight;
                if (b1 && max_item.weight - avg_value > 0.4f * global_width) {
                    excluded_item = max_item;
                }
                else if (!b1 && avg_value - min_item.weight > 0.4f * global_width) {
                    excluded_item = min_item;
                }
            }

            if (excluded_item.id != UINT32_MAX && abs(excluded_item.weight) >= 0.0001f)
            {
                v_idx = (int)excluded_item.id == block_idx * block_size ? block_idx * block_size + 1
                    : block_idx * block_size;
                sum = 0;
                min_item.Set(v_idx, (float)value_list[v_idx]);
                max_item.Set(v_idx, (float)value_list[v_idx]);
                for (v_idx++; v_idx < (block_idx + 1) * block_size; v_idx++)
                {
                    if (v_idx == (int)excluded_item.id) {
                        continue;
                    }

                    const float v = (float)value_list[v_idx];
                    sum += v;
                    if (min_item.weight > v) {
                        min_item.Set(v_idx, v);
                    }
                    if (max_item.weight < v) {
                        max_item.Set(v_idx, v);
                    }
                }

                if (max_item.weight - min_item.weight < 0.7f * block_len)
                {
                    block_len = max_item.weight - min_item.weight;
                    stat.excluded.push_back(excluded_item);
                }
            }

            //float mid_value = (max_value + min_value) / 2;
            //int bucket_id = (int)((min_value - stat.min_value) / width);
            //int bucket_id = (int)((mid_value - stat.min_value) / width);
            int bucket_id = (int)(block_len / width);
            bucket_id = min(bucket_count - 1, max(0, bucket_id));
            stat.block_mean_histogram[bucket_id]++;
        }
    }

    int max_discarding = (int)(value_count * ratio);
    int start = 0, end = bucket_count;
    int discarded_count = 0;
    while (discarded_count < max_discarding)
    {
        int n1 = stat.histogram[start];
        int n2 = stat.histogram[end - 1];

        if (n1 <= n2 && discarded_count + n1 <= max_discarding)
        {
            start++;
            discarded_count += n1;
        }
        else if (n2 <= n1 && discarded_count + n2 <= max_discarding)
        {
            end--;
            discarded_count += n2;
        }
        else
        {
            break;
        }
    }

    stat.soft_min = stat.min_value + width * start;
    stat.soft_max = stat.max_value - width * (bucket_count - end);
}

//static
float DeviceTensorUtil::Rmsd(const vector<half> &vec1, const vector<half> &vec2)
{
    const auto *v1 = (inferflow_fp16*)vec1.data();
    const auto *v2 = (inferflow_fp16*)vec2.data();
    uint64_t n1 = (uint64_t)vec1.size();
    uint64_t n2 = (uint64_t)vec2.size();

    return TensorUtil::Rmsd(v1, n1, v2, n2);
}

//static
float DeviceTensorUtil::NormRmsd(const vector<half> &vec1, const vector<half> &vec2)
{
    const auto *v1 = (inferflow_fp16*)vec1.data();
    const auto *v2 = (inferflow_fp16*)vec2.data();
    uint64_t n1 = (uint64_t)vec1.size();
    uint64_t n2 = (uint64_t)vec2.size();

    return TensorUtil::NormRmsd(v1, n1, v2, n2);
}

bool DeviceTensorUtil::BuildTensor_Q8_B32T1(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q8B32_CAPACITY;
    vector<BlockQ8_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ8_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q8_B32T1<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q8_B32T1, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ8_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q8_B32T2(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q8B32_CAPACITY;
    vector<BlockQ8_B32T2> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ8_B32T2 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q8_B32T2<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q8_B32T2, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ8_B32T2));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q6_B64T1(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q6_B64_CAPACITY;
    vector<BlockQ6_B64T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ6_B64T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q6_B64T1<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q6_B64T1, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ6_B64T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildQ5Tensor(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q5B32_CAPACITY;
    vector<BlockQ5_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ5_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        DeviceQuantization::QuantizeQ5Row<inferflow_fp16>(target_blocks, source_row, cx);
    }

    //int start = 1069248;
    //for (int idx = start; idx < start + 32; idx++) {
    //    cout << (idx > 0 ? ", " : "###") << (float)host_array[idx];
    //}
    //cout << endl;

    //half temp[Q5_CAPACITY];
    //const auto &first_block = block_array[1];
    //Quantization::DequantizeQ5Block(temp, &first_block);
    //LogKeyInfo("e0: %f, base: %f, delta: %f", (float)temp[0], (float)first_block.base, (float)first_block.delta);

    tensor.New(ElementType::Q5, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ5_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildQ4B16Tensor(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q4B16_CAPACITY;
    vector<BlockQ4_B16> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ4_B16 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q4B16<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q4_B16, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ4_B16));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q4_B32T1A(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q4B32_CAPACITY;
    vector<BlockQ4_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ4_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q4_B32T1A<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q4_B32T1A, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ4_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q4_B32T1B(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q4B32_CAPACITY;
    vector<BlockQ4_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ4_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q4_B32T1B<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q4_B32T1B, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ4_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q3H_B64T1(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q3H_B64_CAPACITY;
    vector<BlockQ3H_B64T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ3H_B64T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q3H_B64T1<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q3H_B64T1, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ3H_B64T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q3_B32T1A(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q3B32_CAPACITY;
    vector<BlockQ3_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ3_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q3_B32T1A<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q3_B32T1A, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ3_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q3_B32T1B(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q3B32_CAPACITY;
    vector<BlockQ3_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ3_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q3_B32T1B<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q3_B32T1B, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ3_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q2_B32T1A(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q2B32_CAPACITY;
    vector<BlockQ2_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ2_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q2_B32T1A<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q2_B32T1A, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ2_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

bool DeviceTensorUtil::BuildTensor_Q2_B32T1B(DeviceTensor &tensor,
    const vector<inferflow_fp16> &host_array, int cx, int cy, int cz)
{
    int blocks_per_row = cx / Q2B32_CAPACITY;
    vector<BlockQ2_B32T1> block_array(blocks_per_row * cy);
    for (int row_id = 0; row_id < cy; row_id++)
    {
        const inferflow_fp16 *source_row = host_array.data() + row_id * cx;
        BlockQ2_B32T1 *target_blocks = block_array.data() + row_id * blocks_per_row;
        Quantization::QuantizeRow_Q2_B32T1B<inferflow_fp16>(target_blocks,
            blocks_per_row, source_row, cx);
    }

    tensor.New(ElementType::Q2_B32T1B, cx, cy, cz);
    int bytes = (int)(block_array.size() * sizeof(BlockQ2_B32T1));
    bool ret = tensor.CopyFromHost(block_array.data(), bytes);
    return ret;
}

//static
bool DeviceTensorUtil::GetFP16List_F32(vector<inferflow_fp16> &value_list,
    const HostTensor &tensor, bool be_transpose)
{
    value_list.clear();

    int element_num = (int)TensorUtil::ElementCount(tensor);
    int row_num = (int)TensorUtil::RowCount(tensor);
    int row_size = tensor.ne[0];

    value_list.resize(element_num);
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const float *row_data = (const float*)tensor.RowData(row_idx);
        for (int col_idx = 0; col_idx < row_size; col_idx++)
        {
            int offset = be_transpose ? (col_idx * row_num + row_idx)
                : (row_idx * row_size + col_idx);
            value_list[offset] = row_data[col_idx];
        }
    }

    return true;
}

//static
bool DeviceTensorUtil::GetFP16List_F32(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, bool be_transpose,
    int start_row, int rows)
{
    int element_num = (int)TensorUtil::ElementCount(tensor);
    int row_num = (int)TensorUtil::RowCount(tensor);
    int row_size = tensor.ne[0];

    if (value_buffer.capacity() < element_num) {
        return false;
    }

    int selected_rows = rows >= 0 ? rows : row_num;
    int end_row = rows >= 0 ? min(start_row + rows, row_num) : row_num;
    for (int row_idx = start_row; row_idx < end_row; row_idx++)
    {
        const float *row_data = (const float*)tensor.RowData(row_idx);
        for (int col_idx = 0; col_idx < row_size; col_idx++)
        {
            int offset = be_transpose ? (col_idx * selected_rows + row_idx - start_row)
                : ((row_idx - start_row) * row_size + col_idx);
            value_buffer.Set(offset, row_data[col_idx]);
        }
    }

    return true;
}

//static
bool DeviceTensorUtil::GetFP16List_F16(vector<inferflow_fp16> &value_list,
    const HostTensor &tensor, bool be_transpose)
{
    value_list.clear();

    int element_num = (int)TensorUtil::ElementCount(tensor);
    int row_num = (int)TensorUtil::RowCount(tensor);
    int row_size = tensor.ne[0];

    value_list.resize(element_num);
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const inferflow_fp16 *row_data = (const inferflow_fp16*)tensor.RowData(row_idx);
        if (be_transpose)
        {
            for (int col_idx = 0; col_idx < row_size; col_idx++)
            {
                int offset = be_transpose ? (col_idx * row_num + row_idx)
                    : (row_idx * row_size + col_idx);
                value_list[offset] = row_data[col_idx];
            }
        }
        else
        {
            auto *target_row_data = value_list.data() + row_idx * row_size;
            memcpy(target_row_data, row_data, row_size * sizeof(inferflow_fp16));
        }
    }

    if ((int)value_list.size() != element_num) {
        LogError("Something is wrong");
        return false;
    }

    return true;
}

//static
bool DeviceTensorUtil::GetFP16List_F16(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, bool be_transpose,
    int start_row, int rows)
{
    int element_num = (int)TensorUtil::ElementCount(tensor);
    int row_num = (int)TensorUtil::RowCount(tensor);
    int row_size = tensor.ne[0];

    if (value_buffer.capacity() < element_num) {
        return false;
    }

    int selected_rows = rows >= 0 ? rows : row_num;
    int end_row = rows >= 0 ? min(start_row + rows, row_num) : row_num;
    for (int row_idx = start_row; row_idx < end_row; row_idx++)
    {
        const inferflow_fp16 *row_data = (const inferflow_fp16*)tensor.RowData(row_idx);
        if (be_transpose)
        {
            for (int col_idx = 0; col_idx < row_size; col_idx++)
            {
                int offset = col_idx * selected_rows + row_idx - start_row;
                value_buffer.Set(offset, row_data[col_idx]);
            }

        }
        else
        {
            int offset = (row_idx - start_row) * row_size;
            value_buffer.Set(offset, row_data, row_size);
            //for (int col_idx = 0; col_idx < row_size; col_idx++)
            //{
            //    int offset = (row_idx - start_row) * row_size + col_idx;
            //    value_buffer.Set(offset, row_data[col_idx]);
            //}
        }
    }

    return true;
}

//static
bool DeviceTensorUtil::GetFP16List_Quant(vector<inferflow_fp16> &value_list,
    const HostTensor &tensor, bool be_transpose)
{
    bool ret = true;
    value_list.clear();
    ggml_type data_type_ggml = TensorUtil::ToGgmlType(tensor.data_type);
    const quantize_fns_t &fns = ggml_internal_get_quantize_fn(data_type_ggml);
    dequantize_row_q_t const dequantize_row_q = fns.dequantize_row_q;

    int element_num = (int)TensorUtil::ElementCount(tensor);
    int row_num = (int)TensorUtil::RowCount(tensor);
    int row_size = tensor.ne[0];

    value_list.resize(element_num);
    float *buf = new float[row_size];
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const void *row_data = (const void*)tensor.RowData(row_idx);
        dequantize_row_q(row_data, buf, row_size);
        if (!ret) {
            return false;
        }

        for (int col_idx = 0; col_idx < row_size; col_idx++)
        {
            int offset = be_transpose ? (col_idx * row_num + row_idx)
                : (row_idx * row_size + col_idx);
            value_list[offset] = buf[col_idx];
        }
    }

    delete[] buf;
    return ret;
}

//static
bool DeviceTensorUtil::GetFP16List_Quant(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, bool be_transpose, int start_row, int rows)
{
    bool ret = true;
    ggml_type data_type_ggml = TensorUtil::ToGgmlType(tensor.data_type);
    const quantize_fns_t &fns = ggml_internal_get_quantize_fn(data_type_ggml);
    dequantize_row_q_t const dequantize_row_q = fns.dequantize_row_q;

    int element_num = (int)TensorUtil::ElementCount(tensor);
    int row_num = (int)TensorUtil::RowCount(tensor);
    int row_size = tensor.ne[0];

    if (value_buffer.capacity() < element_num) {
        return false;
    }

    float *buf = new float[row_size];
    int selected_rows = rows >= 0 ? rows : row_num;
    int end_row = rows >= 0 ? min(start_row + rows, row_num) : row_num;
    for (int row_idx = start_row; row_idx < end_row; row_idx++)
    {
        const void *row_data = (const void*)tensor.RowData(row_idx);
        dequantize_row_q(row_data, buf, row_size);
        if (!ret) {
            return false;
        }

        for (int col_idx = 0; col_idx < row_size; col_idx++)
        {
            int offset = be_transpose ? (col_idx * selected_rows + row_idx - start_row)
                : ((row_idx - start_row) * row_size + col_idx);
            value_buffer.Set(offset, buf[col_idx]);
        }
    }

    delete[] buf;
    return ret;
}

bool DeviceTensorUtil::GetFP16List_F32(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, int start_col, int cols)
{
    int row_num = (int)TensorUtil::RowCount(tensor);
    int src_cols = tensor.ne[0];

    int target_cols = cols >= 0 ? min(cols, src_cols - start_col) : src_cols - start_col;
    if (value_buffer.capacity() < target_cols * row_num) {
        return false;
    }

    int end_col = start_col + target_cols;
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const float *row_data = (const float*)tensor.RowData(row_idx);
        for (int col_idx = start_col; col_idx < end_col; col_idx++)
        {
            int offset = row_idx * target_cols + col_idx - start_col;
            value_buffer.Set(offset, (inferflow_fp16)row_data[col_idx]);
        }
    }

    return true;
}

bool DeviceTensorUtil::GetFP16List_F16(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, int start_col, int cols)
{
    int row_num = (int)TensorUtil::RowCount(tensor);
    int src_cols = tensor.ne[0];

    int target_cols = cols >= 0 ? min(cols, src_cols - start_col) : src_cols - start_col;
    if (value_buffer.capacity() < target_cols * row_num) {
        return false;
    }

    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const inferflow_fp16 *row_data = (const inferflow_fp16*)tensor.RowData(row_idx);
        const inferflow_fp16 *src_buf = row_data + start_col;
        int offset = row_idx * target_cols;
        value_buffer.Set(offset, src_buf, target_cols);
    }

    return true;
}

bool DeviceTensorUtil::GetFP16List_Quant(HostHalfBuffer &value_buffer,
    const HostTensor &tensor, int start_col, int cols)
{
    (void)value_buffer; (void)tensor; (void)start_col; (void)cols;
    bool ret = false;
    LogError("The column selection version of GetFP16List_Quant has not been implemented yet.");
    return ret;
}

} //end of namespace

