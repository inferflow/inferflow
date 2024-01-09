#include "tensor_util.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include "sslib/log.h"

namespace inferflow
{

using namespace std;
using namespace sslib;

//static
uint64_t TensorUtil::ElementCount(const HostTensor &tensor)
{
    return tensor.ne[0] * tensor.ne[1] * tensor.ne[2];
}

//static
uint64_t TensorUtil::RowCount(const HostTensor &tensor)
{
    return tensor.ne[1] * tensor.ne[2];
}

//static
uint64_t TensorUtil::ElementCount(const struct ggml_tensor &tensor)
{
    return tensor.ne[0] * tensor.ne[1] * tensor.ne[2] * tensor.ne[3];
}

//static
uint64_t TensorUtil::RowCount(const struct ggml_tensor &tensor)
{
    return tensor.ne[1] * tensor.ne[2] * tensor.ne[3];
}

//static
void TensorUtil::TensorToJson(ostream &strm, const HostTensor &tensor,
    bool include_data, bool is_ext, bool include_stat)
{
    strm << "{\"id\":" << tensor.id;
    strm << ", \"type\":" << (int)tensor.data_type;
    strm << ", \"dim\":" << tensor.dim;
    strm << ", \"ne\":[" << tensor.ne[0] << "," << tensor.ne[1]
        << ", " << tensor.ne[2] << "]";

    vector<float> value_list;
    if (include_data || include_stat)
    {
        bool ret = GetFloatList(value_list, tensor);
        if (!ret) {
            value_list.clear();
            LogError("Failed to extract tensor data");
        }
    }

    if (include_stat)
    {
        TensorStat tensor_stat;
        CalculateStat(tensor_stat, value_list);

        strm << ", \"stat\":[" << tensor_stat.min_value
            << ", " << tensor_stat.max_value
            << ", " << tensor_stat.avg_value
            << ", " << tensor_stat.soft_min
            << ", " << tensor_stat.soft_max << "]";

        strm << ", \"histogram\":[";
        for (int bin_id = 0; bin_id < (int)tensor_stat.histogram.size(); bin_id++)
        {
            strm << (bin_id > 0 ? ", " : "") << tensor_stat.histogram[bin_id];
        }
        strm << "]";
    }

    //data
    if (include_data)
    {
        int cx = tensor.ne[0];
        strm << ",\n  \"data\":[";
        int size = (int)value_list.size();
        //strm << std::scientific << std::setw(3) << std::setprecision(4);
        strm << std::setprecision(4);
        for (int idx = 0; idx < size; idx++)
        {
            if (idx > 0)
            {
                if (is_ext && idx % cx == 0) {
                    strm << ", \"row_start\"";
                }

                if (!is_ext || idx % cx < 8) {
                    strm << (idx % 100 == 0 ? ",\n  " : ",");
                }
            }

            if (!is_ext || idx % cx < 8) {
                strm << value_list[idx];
            }
        }
        strm << "]";
    }

    strm << "}";
}

//static
void TensorUtil::TensorToJson(ostream &strm, const struct ggml_tensor &tensor,
    bool include_data, bool is_ext, bool include_stat)
{
    strm << "{\"id\":" << tensor.id;
    strm << ", \"type\":" << tensor.type;
    strm << ", \"dim\":" << tensor.n_dims;
    strm << ", \"ne\":[" << tensor.ne[0] << "," << tensor.ne[1]
        << ", " << tensor.ne[2] << ", " << tensor.ne[3] << "]";
    strm << ", \"nb\":[" << tensor.nb[0] << ", " << tensor.nb[1]
        << ", " << tensor.nb[2] << ", " << tensor.nb[3] << "]";
    strm << ", \"op\":" << tensor.op;
    int src0_id = tensor.src0 != nullptr ? tensor.src0->id : -1;
    int src1_id = tensor.src1 != nullptr ? tensor.src1->id : -1;
    strm << ", \"src\":[" << src0_id << "," << src1_id << "]";
    strm << ", \"perf\":[" << tensor.perf_runs << ", "  << tensor.perf_cycles
        << ", "  << tensor.perf_time_us << "]";

    vector<float> value_list;
    if (include_data || include_stat)
    {
        bool ret = GetFloatList(value_list, tensor);
        if (!ret) {
            value_list.clear();
            LogError("Failed to extract tensor data");
        }
    }

    if (include_stat)
    {
        TensorStat tensor_stat;
        CalculateStat(tensor_stat, value_list);

        strm << ", \"stat\":[" << tensor_stat.min_value
            << ", " << tensor_stat.max_value
            << ", " << tensor_stat.avg_value
            << ", " << tensor_stat.soft_min
            << ", " << tensor_stat.soft_max << "]";

        strm << ", \"histogram\":[";
        for (int bin_id = 0; bin_id < (int)tensor_stat.histogram.size(); bin_id++)
        {
            strm << (bin_id > 0 ? ", " : "") << tensor_stat.histogram[bin_id];
        }
        strm << "]";
    }

    //data
    if (include_data)
    {
        int cx = (int)tensor.ne[0];
        strm << ",\n  \"data\":[";
        int size = (int)value_list.size();
        //strm << std::scientific << std::setw(3) << std::setprecision(4);
        strm << std::setprecision(4);
        for (int idx = 0; idx < size; idx++)
        {
            if (idx > 0)
            {
                if (is_ext && idx % cx == 0) {
                    strm << ", \"row_start\"";
                }

                if (!is_ext || idx % cx < 8) {
                    strm << (idx % 100 == 0 ? ",\n  " : ",");
                }
            }

            if (!is_ext || idx % cx < 8) {
                strm << value_list[idx];
            }
        }
        strm << "]";
    }

    strm << "}";
}

//static
bool TensorUtil::PrintNetwork(const vector<const struct ggml_tensor*> &tensor_list,
    const string &file_path, bool is_ext)
{
    ofstream writer(file_path);

    for (const ggml_tensor *tensor : tensor_list)
    {
        LogKeyInfo("Printing tensor %d (%d * %d * %d * %d)...",
            tensor->id, tensor->ne[0], tensor->ne[1],
            tensor->ne[2], tensor->ne[3]);
        stringstream ss;
        TensorToJson(ss, *tensor, true, is_ext);
        writer << ss.str() << "\n";
    }

    writer.close();
    return writer.good();
}

//static
bool TensorUtil::PrintGraph(const ggml_cgraph &graph, const string &file_path,
    int layer_idx, const struct ggml_tensor *graph_input)
{
    ofstream writer(file_path);

    const auto *tensor = graph_input;
    LogKeyInfo("Printing the input tensor %d (%d * %d * %d * %d)...",
            tensor->id, tensor->ne[0], tensor->ne[1],
            tensor->ne[2], tensor->ne[3]);

    {
        stringstream ss;
        TensorToJson(ss, *tensor, true);
        writer << ss.str() << "\n";
    }

    for (int idx = 0; idx < graph.n_nodes; idx++)
    {
        tensor = graph.nodes[idx];
        if (tensor != nullptr && tensor->id != 0)
        {
            int node_layer_idx = tensor->id / 10000 - 1;
            if (node_layer_idx < 0 || layer_idx == node_layer_idx
                || layer_idx >= 99)
            {
                LogKeyInfo("Printing tensor %d (%d * %d * %d * %d)...",
                    tensor->id, tensor->ne[0], tensor->ne[1],
                    tensor->ne[2], tensor->ne[3]);

                stringstream ss;
                TensorToJson(ss, *tensor, true);
                writer << ss.str() << "\n";
            }
        }
    }

    writer.close();
    return writer.good();
}

//static
void TensorUtil::CalculateStat(TensorStat &stat, const vector<float> &value_list)
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
    float width = max(0.00001f, (stat.max_value - stat.min_value) / bucket_count);
    for (float v : value_list)
    {
        int bucket_id = (int)((v - stat.min_value) / width);
        bucket_id = min(bucket_count - 1, max(0, bucket_id));
        stat.histogram[bucket_id]++;
    }

    int max_discarding = (int)(value_count * 0.001f);
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
void TensorUtil::NormalizeByRow(HostTensor &tensor)
{
    bool is_f16 = tensor.data_type == ElementType::F16;
    bool is_f32 = tensor.data_type == ElementType::F32;
    if (!is_f16 && !is_f32) {
        return;
    }

    int cx = (int)tensor.ne[0], cy = (int)tensor.ne[1];
    float value = 0;
    if (is_f16)
    {
        host_fp16_t *f16_array = (host_fp16_t*)tensor.data;
        for (int row = 0; row < cy; row++)
        {
            double sum = 0;
            for (int col = 0; col < cx; col++)
            {
                value = (float)f16_array[row * cx + col];
                sum += (value * value);
            }

            float scale = 1.0f / max(0.000001f, (float)sqrt(sum));
            //if (row <= 5) {
            //    LogKeyInfo("row: %d, scale: %.3f", row, scale);
            //}

            for (int col = 0; col < cx; col++)
            {
                value = (float)f16_array[row * cx + col];
                f16_array[row * cx + col] = value * scale;
            }
        }
    }
    else
    {
        float *f32_array = (float*)tensor.data;
        for (int row = 0; row < cy; row++)
        {
            double sum = 0;
            for (int col = 0; col < cx; col++)
            {
                value = (float)f32_array[row * cx + col];
                sum += (value * value);
            }

            float scale = 1.0f / max(0.000001f, (float)sqrt(sum));
            //if (row <= 5) {
            //    LogKeyInfo("row: %d, scale: %.3f", row, scale);
            //}

            for (int col = 0; col < cx; col++)
            {
                value = (float)f32_array[row * cx + col];
                f32_array[row * cx + col] = value * scale;
            }
        }
    }
}

//static
bool TensorUtil::Compare(const vector<float> &vec1, const vector<float> &vec2,
    float diff_ratio_threshold, bool is_study_mode)
{
    int size1 = (int)vec1.size(), size2 = (int)vec2.size();
    bool ret = Compare(vec1.data(), size1, vec2.data(), size2,
        diff_ratio_threshold, is_study_mode);
    return ret;
}

//static
bool TensorUtil::Compare(const float *vec1, uint64_t size1, const float *vec2,
    uint64_t size2, float diff_ratio_threshold, bool is_study_mode)
{
    (void)diff_ratio_threshold;
    if (size1 != size2)
    {
        if (is_study_mode) {
            LogWarning("Different size: %d vs. %d", size1, size2);
        }
        return false;
    }

    if (size1 == 0)
    {
        if (is_study_mode) {
            LogKeyInfo("The two empty vectors are equal.");
        }
        return true;
    }

    if (is_study_mode) {
        LogKeyInfo("Comparing (first value: %f vs. %f)...", vec1[0], vec2[0]);
    }

    uint64_t size = size1;
    int subtle_diff_num = 0;
    for (uint64_t idx = 0; idx < size; idx++)
    {
        float larger_value = max(max(fabs(vec1[idx]), fabs(vec2[idx])), 0.01f);
        float diff = fabs(vec1[idx] - vec2[idx]);
        if (diff >= 0.01f && diff >= 0.1f * larger_value)
        {
            if (is_study_mode)
            {
                LogWarning("Inconsistent results on position %d: %f vs. %f",
                    idx, vec1[idx], vec2[idx]);
            }
            return false;
        }

        if (diff >= 0.05f * larger_value)
        {
            subtle_diff_num++;
            if (is_study_mode && subtle_diff_num <= 1)
            {
                LogWarning("Subtle diff on position %d: %f vs. %f",
                    idx, vec1[idx], vec2[idx]);
            }
        }
    }

    if (is_study_mode && subtle_diff_num > 0) {
        LogWarning("Number of subtle diffs: %d/%d", subtle_diff_num, size);
    }
    if (subtle_diff_num > 0.05 * size) {
        return false;
    }

    if (is_study_mode) {
        LogKeyInfo("They are almost equal.");
    }
    return true;
}

//static
bool TensorUtil::Compare(const HostTensor &tensor_a, const HostTensor &tensor_b,
    float diff_ratio_threshold, bool is_study_mode)
{
    (void)diff_ratio_threshold;
    bool ret = true;
    int rows_a = tensor_a.Rows(), cols_a = tensor_a.Columns();
    int rows_b = tensor_b.Rows(), cols_b = tensor_b.Columns();
    if (rows_a != rows_b || cols_a != cols_b)
    {
        if (is_study_mode)
        {
            LogWarning("Different numbers of rows or columns: (%d, %d) vs. (%d, %d)",
                rows_a, cols_a, rows_b, cols_b);
        }
        return false;
    }

    if (rows_a == 0 || cols_a == 0)
    {
        if (is_study_mode) {
            LogKeyInfo("The two empty tensors are equal.");
        }
        return true;
    }

    uint64_t size = tensor_a.size;
    uint64_t total_subtle_diff_num = 0;
    vector<float> vec1, vec2;
    for (int row_idx = 0; row_idx < rows_a; row_idx++)
    {
        tensor_a.CopyRow(vec1, row_idx);
        tensor_b.CopyRow(vec2, row_idx);

        int subtle_diff_num = 0;
        for (int col_idx = 0; col_idx < cols_a; col_idx++)
        {
            float larger_value = max(max(fabs(vec1[col_idx]), fabs(vec2[col_idx])), 0.01f);
            float diff = fabs(vec1[col_idx] - vec2[col_idx]);
            if (diff >= 0.01f && diff >= 0.1f * larger_value)
            {
                if (is_study_mode)
                {
                    LogWarning("Inconsistent results on row %d and column %d: %f vs. %f",
                        row_idx, col_idx, vec1[col_idx], vec2[col_idx]);
                }
                return false;
            }

            if (diff >= 0.05f * larger_value)
            {
                subtle_diff_num++;
                total_subtle_diff_num++;
                if (is_study_mode && total_subtle_diff_num <= 1)
                {
                    LogWarning("Subtle diff on row %d and column %d: %f vs. %f",
                        row_idx, col_idx, vec1[col_idx], vec2[col_idx]);
                }
            }
        }
    }

    if (total_subtle_diff_num > 0.05 * size) {
        return false;
    }

    if (is_study_mode) {
        LogKeyInfo("They are almost equal.");
    }
    return ret;
}

//root mean square deviation
float TensorUtil::Rmsd(const vector<inferflow_fp16> &vec1, const vector<inferflow_fp16> &vec2)
{
    uint64_t size1 = (uint64_t)vec1.size();
    uint64_t size2 = (uint64_t)vec2.size();
    return Rmsd(vec1.data(), size1, vec2.data(), size2);
}

float TensorUtil::Rmsd(const inferflow_fp16 *vec1, uint64_t size1,
    const inferflow_fp16 *vec2, uint64_t size2)
{
    double sum = 0;
    uint64_t size = size1 >= size2 ? size1 : size2; //max(size1, size2)
    if (size <= 0) {
        return 0;
    }

    float v1 = 0, v2 = 0;
    for (uint64_t idx = 0; idx < size; idx++)
    {
        v1 = idx < size1 ? (float)vec1[idx] : 0;
        v2 = idx < size2 ? (float)vec2[idx] : 0;
        sum += (v1 - v2) * (v1 - v2);
    }

    sum /= size;
    float rmsd = (float)sqrt(sum);
    return rmsd;
}

//normalized root mean square deviation
float TensorUtil::NormRmsd(const vector<inferflow_fp16> &vec1, const vector<inferflow_fp16> &vec2)
{
    uint64_t size1 = (uint64_t)vec1.size();
    uint64_t size2 = (uint64_t)vec2.size();
    return NormRmsd(vec1.data(), size1, vec2.data(), size2);
}

float TensorUtil::NormRmsd(const inferflow_fp16 *vec1, uint64_t size1,
    const inferflow_fp16 *vec2, uint64_t size2)
{
    double sum = 0, sum1 = 0, sum2 = 0;
    uint64_t size = size1 >= size2 ? size1 : size2; //max(size1, size2)
    if (size <= 0) {
        return 0;
    }

    float v1 = 0, v2 = 0;
    for (uint64_t idx = 0; idx < size; idx++)
    {
        v1 = idx < size1 ? (float)vec1[idx] : 0;
        v2 = idx < size2 ? (float)vec2[idx] : 0;
        sum += (v1 - v2) * (v1 - v2);
        sum1 += v1 * v1;
        sum2 += v2 * v2;
    }

    sum /= size;
    double rmsd = sqrt(sum);

    sum1 /= size;
    sum2 /= size;
    double m = max(0.0000001, max(sqrt(sum1), sqrt(sum2)));

    return (float)(rmsd / m);
}

float TensorUtil::MeanDeviation(const vector<inferflow_fp16> &vec1,
    const vector<inferflow_fp16> &vec2)
{
    uint64_t size1 = (uint64_t)vec1.size();
    uint64_t size2 = (uint64_t)vec2.size();
    return MeanDeviation(vec1.data(), size1, vec2.data(), size2);
}

float TensorUtil::MeanDeviation(const inferflow_fp16 *vec1, uint64_t size1,
    const inferflow_fp16 *vec2, uint64_t size2)
{
    double sum = 0;
    uint64_t size = size1 >= size2 ? size1 : size2; //max(size1, size2)
    if (size <= 0) {
        return 0;
    }

    float v1 = 0, v2 = 0;
    for (uint64_t idx = 0; idx < size; idx++)
    {
        v1 = idx < size1 ? (float)vec1[idx] : 0;
        v2 = idx < size2 ? (float)vec2[idx] : 0;
        sum += abs(v1 - v2);
    }

    return float(sum / size);
}

//static
bool TensorUtil::GetFloatList(vector<float> &value_list, const HostTensor &tensor,
    bool be_transpose)
{
    bool ret = true;
    value_list.clear();

    if (TensorCommon::IsQuantType(tensor.data_type))
    {
        ret = GetFloatList_Quant(value_list, tensor, be_transpose);
    }
    else
    {
        switch (tensor.data_type)
        {
        case ElementType::F32:
            ret = GetFloatList_F32(value_list, tensor, be_transpose);
            break;
        case ElementType::F16:
            ret = GetFloatList_F16(value_list, tensor, be_transpose);
            break;
        default:
            LogError("Invalid tensor data type: %d", tensor.data_type);
            ret = false;
            break;
        }
    }

    for (int idx = 0; idx < (int)value_list.size(); idx++)
    {
        if (!isfinite(value_list[idx])) {
            value_list[idx] = 0;
        }
    }

    return ret;
}

//static
bool TensorUtil::GetFloatList(vector<float> &value_list, const ggml_tensor &tensor,
    bool be_transpose)
{
    bool ret = true;
    value_list.clear();

    switch (tensor.type)
    {
        case GGML_TYPE_F32:
            ret = GetFloatList_F32(value_list, tensor, be_transpose);
            break;
        case GGML_TYPE_F16:
            ret = GetFloatList_F16(value_list, tensor, be_transpose);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            ret = GetFloatList_Quant(value_list, tensor, be_transpose);
            break;
        default:
            LogError("Invalid tensor data type: %d", tensor.type);
            ret = false;
            break;
    }

    for (int idx = 0; idx < (int)value_list.size(); idx++)
    {
        if (!isfinite(value_list[idx])) {
            value_list[idx] = 0;
        }
    }

    return ret;
}

//static
bool TensorUtil::GetFloatList_F32(vector<float> &value_list,
    const HostTensor &tensor, bool be_transpose)
{
    value_list.clear();

    int row_num = tensor.Rows();
    int col_num = tensor.Columns();

    value_list.resize(tensor.size);
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const float *row_data = (const float*)tensor.RowData(row_idx);
        if (be_transpose)
        {
            for (int col_idx = 0; col_idx < col_num; col_idx++)
            {
                int offset = col_idx * row_num + row_idx;
                value_list[offset] = row_data[col_idx];
            }
        }
        else
        {
            float *target_array = value_list.data() + row_idx * col_num;
            memcpy(target_array, row_data, col_num * sizeof(float));
        }
    }

    return true;
}

//static
bool TensorUtil::GetFloatList_F16(vector<float> &value_list,
    const HostTensor &tensor, bool be_transpose)
{
    value_list.clear();

    int row_num = tensor.Rows();
    int col_num = tensor.Columns();

    value_list.resize(tensor.size);
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const inferflow_fp16 *row_data = (const inferflow_fp16*)tensor.RowData(row_idx);
        for (int col_idx = 0; col_idx < col_num; col_idx++)
        {
            int offset = be_transpose ? (col_idx * row_num + row_idx)
                : (row_idx * col_num + col_idx);
            value_list[offset] = (float)row_data[col_idx];
        }
    }

    return true;
}

//static
bool TensorUtil::GetFloatList_Quant(vector<float> &value_list,
    const HostTensor &tensor, bool be_transpose)
{
    bool ret = true;
    value_list.clear();
    ggml_type data_type_ggml = ToGgmlType(tensor.data_type);
    const quantize_fns_t &fns = ggml_internal_get_quantize_fn(data_type_ggml);
    dequantize_row_q_t const dequantize_row_q = fns.dequantize_row_q;

    int element_num = (int)ElementCount(tensor);
    int row_num = (int)RowCount(tensor);
    int row_size = tensor.ne[0];
    //const int ne1 = (int)tensor.ne[1];
    //const int ne2 = (int)tensor.ne[2];

    value_list.resize(element_num);
    float *buf = new float[row_size];
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const float *row_data = (const float*)tensor.RowData(row_idx);
        dequantize_row_q(row_data, buf, row_size);

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
bool TensorUtil::GetFloatList_F32(vector<float> &value_list,
    const ggml_tensor &tensor, bool be_transpose)
{
    value_list.clear();

    int element_num = (int)ElementCount(tensor);
    int row_num = (int)RowCount(tensor);
    int row_size = (int)tensor.ne[0];
    const int ne1 = (int)tensor.ne[1];
    const int ne2 = (int)tensor.ne[2];
    const int ne3 = (int)tensor.ne[3];
    const int nb0 = (int)tensor.nb[0];
    const int nb1 = (int)tensor.nb[1];
    const int nb2 = (int)tensor.nb[2];
    const int nb3 = (int)tensor.nb[3];

    if (nb0 != sizeof(float))
    {
        LogWarning("Pay attention: ne = (%d, %d, %d, %d), nb0 = %d!!!",
            row_size, ne1, ne2, ne3, nb0);
    }

    value_list.resize(element_num);
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const int i3 = row_idx / (ne2 * ne1);
        const int i2 = (row_idx - i3 * ne2 * ne1) / ne1;
        const int i1 = (row_idx - i3 * ne2 * ne1 - i2 * ne1);

        if (nb0 == sizeof(float))
        {
            const float *row_data = (const float*)((const char*)tensor.data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
            for (int col_idx = 0; col_idx < row_size; col_idx++)
            {
                int offset = be_transpose ? (col_idx * row_num + row_idx)
                    : (row_idx * row_size + col_idx);
                value_list[offset] = row_data[col_idx];
            }
        }
        else
        {
            for (int i0 = 0; i0 < row_size; i0++)
            {
                int offset = i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;
                const float *src_ptr = (const float*)((const char*)tensor.data + offset);
                int target_offset = be_transpose ? (i0 * row_num + row_idx) : (row_idx * row_size + i0);
                value_list[target_offset] = *src_ptr;
            }
        }
    }

    return true;
}

//static
bool TensorUtil::GetFloatList_F16(vector<float> &value_list,
    const ggml_tensor &tensor, bool be_transpose)
{
    value_list.clear();

    int element_num = (int)ElementCount(tensor);
    int row_num = (int)RowCount(tensor);
    int row_size = (int)tensor.ne[0];
    const int ne1 = (int)tensor.ne[1];
    const int ne2 = (int)tensor.ne[2];
    const int nb1 = (int)tensor.nb[1];
    const int nb2 = (int)tensor.nb[2];
    const int nb3 = (int)tensor.nb[3];

    value_list.resize(element_num);
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const int i3 = row_idx / (ne2 * ne1);
        const int i2 = (row_idx - i3 * ne2 * ne1) / ne1;
        const int i1 = (row_idx - i3 * ne2 * ne1 - i2 * ne1);
        const int offset = i1 * nb1 + i2 * nb2 + i3 * nb3;

        const ggml_fp16_t *row_data = (const ggml_fp16_t*)((const char*)tensor.data + offset);
        for (int col_idx = 0; col_idx < row_size; col_idx++)
        {
            int target_offset = be_transpose ? (col_idx * row_num + row_idx)
                : (row_idx * row_size + col_idx);
            value_list[target_offset] = ggml_fp16_to_fp32(row_data[col_idx]);
        }
    }

    return true;
}

//static
bool TensorUtil::GetFloatList_Quant(vector<float> &value_list,
    const ggml_tensor &tensor, bool be_transpose)
{
    bool ret = true;
    value_list.clear();
    const quantize_fns_t &fns = ggml_internal_get_quantize_fn(tensor.type);
    dequantize_row_q_t const dequantize_row_q = fns.dequantize_row_q;

    int element_num = (int)ElementCount(tensor);
    int row_num = (int)RowCount(tensor);
    int row_size = (int)tensor.ne[0];
    const int ne1 = (int)tensor.ne[1];
    const int ne2 = (int)tensor.ne[2];
    const int nb1 = (int)tensor.nb[1];
    const int nb2 = (int)tensor.nb[2];
    const int nb3 = (int)tensor.nb[3];

    value_list.resize(element_num);
    float *buf = new float[row_size];
    for (int row_idx = 0; row_idx < row_num; row_idx++)
    {
        const int i3 = row_idx / (ne2 * ne1);
        const int i2 = (row_idx - i3 * ne2 * ne1) / ne1;
        const int i1 = (row_idx - i3 * ne2 * ne1 - i2 * ne1);

        const void *row_data = (const void*)((const char*)tensor.data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
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

ggml_type TensorUtil::ToGgmlType(ElementType t)
{
    switch (t)
    {
    case ElementType::F32:
        return GGML_TYPE_F32;
        break;
    case ElementType::F16:
    case ElementType::BF16:
        return GGML_TYPE_F16;
        break;
    case ElementType::Q8_B32T2:
        return GGML_TYPE_Q8_0;
        break;
    case ElementType::Q5:
        return GGML_TYPE_Q5_1;
        break;
    case ElementType::Q4_B32T1A:
    case ElementType::Q4_B32T1B:
        return GGML_TYPE_Q4_1;
        break;
    default: break;
    }

    return GGML_TYPE_F16;
}

ElementType TensorUtil::ToElementType(ggml_type t)
{
    switch (t)
    {
    case GGML_TYPE_F32:
        return ElementType::F32;
        break;
    case GGML_TYPE_F16:
        return ElementType::F16;
        break;
    case GGML_TYPE_Q8_0:
        return ElementType::Q8_B32T2;
        break;
    case GGML_TYPE_Q5_1:
        return ElementType::Q5;
        break;
    case GGML_TYPE_Q4_1:
        return ElementType::Q4_B32T1A;
        break;
    default: break;
    }

    return ElementType::Invalid;
}

} //end of namespace
