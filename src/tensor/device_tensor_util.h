#pragma once

#include <vector>
#include <map>
#include "sslib/blocked_allocator.h"
#include "sslib/json.h"
#include "tensor_opr.h"
#include "tensor_util.h"
#include "host_float_buffer.h"

namespace inferflow
{

using std::string;
using std::wstring;
using std::vector;
using std::map;
using std::wostream;
using sslib::BlockedAllocator;
using sslib::JsonParser;
using sslib::JsonBuilder;

class DeviceTensorUtil
{
public:
    static bool LoadFromJson(map<int, DeviceTensorOprNode> &node_map,
        BlockedAllocator<DeviceTensor> &heap, const string &file_path);

    static bool CopyAndCompare(const DeviceTensor &C1, const DeviceTensor &C2,
        float diff_ratio_threshold = 0.05f, bool is_study_mode = true);

    static bool GetFP16List(vector<inferflow_fp16> &value_list, const HostTensor &tensor,
        bool be_transpose = false);
    static bool GetFP16List(HostHalfBuffer &value_buffer, const HostTensor &tensor,
        bool be_transpose = false, int start_row = 0, int rows = -1);
    static bool GetFP16List(HostHalfBuffer &value_buffer, const HostTensor &tensor,
        int start_col, int cols);

    static bool BuildTensor_Q8_B32T1(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q8_B32T2(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q6_B64T1(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q5_B32T1(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q5_B64T1(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildQ4B16Tensor(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q4_B32T1A(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q4_B32T1B(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q4_B64T1(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q3H_B64T1(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q3_B32T1A(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q3_B32T1B(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q2_B32T1A(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);
    static bool BuildTensor_Q2_B32T1B(DeviceTensor &tensor, const vector<inferflow_fp16> &host_array,
        int cx, int cy, int cz);

    static void CalculateStat(TensorStat &stat, const vector<inferflow_fp16> &value_list,
        float ratio = 0.0001f, int block_size = 0);

    //root mean square deviation
    static float Rmsd(const vector<half> &vec1, const vector<half> &vec2);
    //normalized root mean square deviation
    static float NormRmsd(const vector<half> &vec1, const vector<half> &vec2);

protected:
    static bool JsonToTensorOprNode(DeviceTensorOprNode &node, const wstring &json_str,
        BlockedAllocator<DeviceTensor> &heap, JsonParser &jparser);

    static bool GetFP16List_F32(vector<inferflow_fp16> &value_list,
        const HostTensor &tensor, bool be_transpose);
    static bool GetFP16List_F16(vector<inferflow_fp16> &value_list,
        const HostTensor &tensor, bool be_transpose);
    static bool GetFP16List_Quant(vector<inferflow_fp16> &value_list,
        const HostTensor &tensor, bool be_transpose);

    static bool GetFP16List_F32(HostHalfBuffer &value_buffer,
        const HostTensor &tensor, bool be_transpose,
        int start_row = 0, int rows = -1);
    static bool GetFP16List_F16(HostHalfBuffer &value_buffer,
        const HostTensor &tensor, bool be_transpose,
        int start_row = 0, int rows = -1);
    static bool GetFP16List_Quant(HostHalfBuffer &value_buffer,
        const HostTensor &tensor, bool be_transpose,
        int start_row = 0, int rows = -1);

    static bool GetFP16List_F32(HostHalfBuffer &value_buffer,
        const HostTensor &tensor, int start_col, int cols);
    static bool GetFP16List_F16(HostHalfBuffer &value_buffer,
        const HostTensor &tensor, int start_col, int cols);
    static bool GetFP16List_Quant(HostHalfBuffer &value_buffer,
        const HostTensor &tensor, int start_col, int cols);
};

} //end of namespace

