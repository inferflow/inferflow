#pragma once

#include <vector>
#include <map>
#include "sslib/blocked_allocator.h"
#include "sslib/json.h"
#include "sslib/prime_types.h"
#include "ggml/ggml.h"
#include "host_tensor.h"

namespace inferflow
{

using std::string;
using std::wstring;
using std::vector;
using std::map;
using std::ostream;
using std::wostream;
using sslib::IdWeight;
using sslib::BlockedAllocator;
using sslib::JsonParser;
using sslib::JsonBuilder;

struct TensorStat
{
    float min_value = 0;
    float max_value = 0;
    float avg_value = 0;
    float soft_min = 0;
    float soft_max = 0;
    vector<int> histogram;
    vector<int> block_mean_histogram;
    vector<IdWeight<float>> excluded;

    void Clear()
    {
        min_value = 0;
        max_value = 0;
        avg_value = 0;
        soft_min = 0;
        soft_max = 0;
        histogram.clear();
        block_mean_histogram.clear();
        excluded.clear();
    }
};

class TensorUtil
{
public:
    static uint64_t ElementCount(const HostTensor &tensor);
    static uint64_t RowCount(const HostTensor &tensor);
    static uint64_t ElementCount(const struct ggml_tensor &tensor);
    static uint64_t RowCount(const struct ggml_tensor &tensor);

    static void TensorToJson(ostream &strm, const HostTensor &tensor,
        bool include_data = true, bool is_ext = false, bool include_stat = false);
    static void TensorToJson(ostream &strm, const struct ggml_tensor &tensor,
        bool include_data = true, bool is_ext = false, bool include_stat = false);

    static bool PrintNetwork(const vector<const struct ggml_tensor*> &tensor_list,
        const string &file_path, bool is_ext = false);
    static bool PrintGraph(const ggml_cgraph &graph, const string &file_path,
        int layer_idx, const struct ggml_tensor *graph_input);

    static bool GetFloatList(vector<float> &value_list, const HostTensor &tensor,
        bool be_transpose = false);
    static bool GetFloatList(vector<float> &value_list, const ggml_tensor &tensor,
        bool be_transpose = false);

    static void CalculateStat(TensorStat &stat, const vector<float> &value_list);

    static void NormalizeByRow(HostTensor &tensor);

    static bool Compare(const vector<float> &vec1, const vector<float> &vec2,
        float diff_ratio_threshold = 0.05f, bool is_study_mode = false);
    static bool Compare(const float *vec1, uint64_t size1, const float *vec2, uint64_t size2,
        float diff_ratio_threshold = 0.05f, bool is_study_mode = false);
    static bool Compare(const HostTensor &tensor_a, const HostTensor &tensor_b,
        float diff_ratio_threshold = 0.05f, bool is_study_mode = false);

    //root mean square deviation
    static float Rmsd(const vector<inferflow_fp16> &vec1, const vector<inferflow_fp16> &vec2);
    static float Rmsd(const inferflow_fp16 *vec1, uint64_t size1,
        const inferflow_fp16 *vec2, uint64_t size2);
    //normalized root mean square deviation
    static float NormRmsd(const vector<inferflow_fp16> &vec1, const vector<inferflow_fp16> &vec2);
    static float NormRmsd(const inferflow_fp16 *vec1, uint64_t size1,
        const inferflow_fp16 *vec2, uint64_t size2);

    static float MeanDeviation(const vector<inferflow_fp16> &vec1,
        const vector<inferflow_fp16> &vec2);
    static float MeanDeviation(const inferflow_fp16 *vec1, uint64_t size1,
        const inferflow_fp16 *vec2, uint64_t size2);

    static ggml_type ToGgmlType(ElementType t);
    static ElementType ToElementType(ggml_type t);

protected:
    static bool GetFloatList_F32(vector<float> &value_list,
        const HostTensor &tensor, bool be_transpose);
    static bool GetFloatList_F16(vector<float> &value_list,
        const HostTensor &tensor, bool be_transpose);
    static bool GetFloatList_Quant(vector<float> &value_list,
        const HostTensor &tensor, bool be_transpose);

    static bool GetFloatList_F32(vector<float> &value_list,
        const ggml_tensor &tensor, bool be_transpose);
    static bool GetFloatList_F16(vector<float> &value_list,
        const ggml_tensor &tensor, bool be_transpose);
    static bool GetFloatList_Quant(vector<float> &value_list,
        const ggml_tensor &tensor, bool be_transpose);
};

} //end of namespace
