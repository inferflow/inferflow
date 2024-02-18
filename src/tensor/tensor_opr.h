#pragma once

#include "sslib/string.h"
#include "sslib/blocked_allocator.h"
#include "device_tensor.h"
#include "device_memory_heap.h"
#include "tensor_common.h"
#include "host_tensor_opr.h"

INFER_FLOW_BEGIN

using std::string;
using std::vector;
using std::map;
using sslib::StrLessNoCase;
using sslib::BlockedAllocator;

struct DeviceTensorOprNode
{
    DeviceTensor *target = nullptr;
    int sources[2];
    //DeviceTensor *sources[2];
    int opr = 0;
    float perf_time = 0;

    DeviceTensorOprNode()
    {
        sources[0] = 0;
        sources[1] = 0;
    }
};

class TensorOpr
{
public:
    TensorOpr();
    virtual ~TensorOpr();

    static bool Assign(DeviceTensor &A, float value);
    static bool Assign(DeviceTensor &B, const DeviceTensor &A,
        int a_start_row = 0, int b_start_row = 0, int row_num = -1);

    static bool RepeatKV(DeviceTensor &B, const DeviceTensor &A,
        int heads_per_group);

    static bool AssignColumns(DeviceTensor &B, const DeviceTensor &A,
        int start_col_a, int step = 1, int start_col_b = 0, int col_num = -1);
    static bool AssignColumns(DeviceTensor &A, float value,
        int start_col, int col_num = -1);

    static bool Reshape(DeviceTensor &A, int new_dim_count, int new_ne0,
        int new_ne1, int new_ne2 = 0);
    static bool Reshape(DeviceTensor &A, int new_dim_count,
        int dims[DeviceTensor::MaxDimCount]);

    //transpose the tensor
    static bool Transpose(DeviceTensor &T, const DeviceTensor &S, int alg_id = 0);
    //transpose the row matrix of the tensor
    static bool TransposeYZ(DeviceTensor &A);

    static bool LayerNormalization(DeviceTensor &T, const DeviceTensor &S,
        TensorNormAlg alg_id = TensorNormAlg::STD,
        const DeviceTensor *M = nullptr, const DeviceTensor *A = nullptr);
    static bool LinearNorm(DeviceTensor &T, const DeviceTensor &S);
    static bool StdNorm(DeviceTensor &T, const DeviceTensor &S,
        const DeviceTensor *M = nullptr, const DeviceTensor *A = nullptr);
    static bool RmsNorm(DeviceTensor &T, const DeviceTensor &S,
        const DeviceTensor *M = nullptr, const DeviceTensor *A = nullptr);
    static bool RmsNorm_Naive(DeviceTensor &T, const DeviceTensor &S);

    static bool VectorDotProduct(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B);

    // Elementwise unary operations
    static bool AggregateByRow(DeviceTensor &B, const DeviceTensor &A,
        RowAggregationType agg_type = RowAggregationType::Sum);
    static bool SumByRow(DeviceTensor &B, const DeviceTensor &A);
    static bool MaxByRow(DeviceTensor &B, const DeviceTensor &A);
    static bool MinByRow(DeviceTensor &B, const DeviceTensor &A);

    static bool Activation(DeviceTensor &B, const DeviceTensor &A, ActivationFn fn);
    static bool ReluActivation(DeviceTensor &B, const DeviceTensor &A);
    static bool SiluActivation(DeviceTensor &B, const DeviceTensor &A, bool is_glu);
    static bool GeluActivation(DeviceTensor &B, const DeviceTensor &A);

    static bool SoftMax(DeviceTensor &A,
        int diag_mask_prefix_len = -1, float mask_value = -1e9f,
        float scale = 1.0f, DeviceTensor *aux_tensor = nullptr);
    static bool SoftMax(DeviceTensor &B, const DeviceTensor &A,
        int diag_mask_prefix_len = -1, float mask_value = -1e9f,
        float scale = 1.0f, DeviceTensor *aux_tensor = nullptr);

    // Elementwise binary operations
    static bool Scale(DeviceTensor &A, float scale);
    static bool Scale(DeviceTensor &C, const DeviceTensor &A, float scale);
    static bool Scale(DeviceTensor &A, const DeviceTensor &scale,
        bool is_reverse = false);
    static bool Scale(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &scale, bool is_reverse = false);

    static bool Add(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, int alg_id = 0);
    static bool Mul(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B);

    static bool AddByRowIndex(DeviceTensor &B, const DeviceTensor &A,
        const DeviceTensor &idx_tensor, const DeviceTensor *weight_tensor);

    //Supports ROPE and ALIBI
    static bool PositionEmbedding(DeviceTensor &A, const PosEmbeddingParams &params,
        int start_z = 0, int z_num = -1, int base_z = 0);
    static bool PositionEmbedding(DeviceTensor &B, const DeviceTensor &A,
        const PosEmbeddingParams &params, int start_z = 0, int z_num = -1,
        int base_z = 0);

    static bool DiagMask(DeviceTensor &A, int context_len, float value);
    static bool DiagMask(DeviceTensor &B, const DeviceTensor &A,
        int context_len, float value);

    //check each element for not being nan or inf
    //return: invalid element count
    static int CheckElements(int &invalid_pos, const DeviceTensor &tensor);

    static bool Dequantize(DeviceTensor &B, const DeviceTensorEx &Ax,
        bool be_transpose = false, bool be_sync = true, int alg_id = 0);
    static bool Dequantize(DeviceTensor &B, const DeviceTensor &A, int alg_id = 0);

    static bool Quantize(DeviceTensor &B, const DeviceTensor &A);

protected:
    static bool Transpose_Alg1(DeviceTensor &T, const DeviceTensor &S);
    static bool Transpose_Alg2(DeviceTensor &T, const DeviceTensor &S);

    static bool Add_Alg1(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B);
    static bool Add_Alg2(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B);
    static bool Add_Alg3(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B);

    static bool SoftMax_Alg1(DeviceTensor &B, const DeviceTensor &A,
        DeviceTensor *aux_tensor = nullptr);
    static bool SoftMax_Alg2(DeviceTensor &B, const DeviceTensor &A,
        int diag_mask_prefix_len = -1, float mask_value = -1e9f,
        float scale = 1.0f);

    //A: source; B: target
    static bool IsCompatible_AB(const DeviceTensor &A,
        const DeviceTensor &B, bool be_transpose = false);
    //A, B: sources; B: target
    static bool IsCompatible_ABC(const DeviceTensor &A,
        const DeviceTensor &B, const DeviceTensor &C);

    static bool DequantizeQ5_Alg1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ5_Alg2(DeviceTensor &B, const DeviceTensor &A, bool be_transpose);

protected:
    static bool QuantizeQ8_B32T1(DeviceTensor &B, const DeviceTensor &A, int alg_id = 0);
    static bool QuantizeQ8_B32T2(DeviceTensor &B, const DeviceTensor &A, int alg_id = 0);
    static bool QuantizeQ6_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ5_B32T1(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ5_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ4B16(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ4_B32T1(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ4_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ3H_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ3_B32T1(DeviceTensor &B, const DeviceTensor &A);
    static bool QuantizeQ2_B32T1(DeviceTensor &B, const DeviceTensor &A);

    static bool DequantizeQ8_B32T1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ8_B32T2(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ6_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ5_B32T1(DeviceTensor &B, const DeviceTensor &A,
        bool be_transpose = false, int alg_id = 0);
    static bool DequantizeQ5_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ4B16(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ4_B32T1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ4_B32T1_Alg1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ4_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ3H_B64T1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ3_B32T1(DeviceTensor &B, const DeviceTensor &A);
    static bool DequantizeQ2_B32T1(DeviceTensor &B, const DeviceTensor &A);

    static bool DequantizeQ8_GlobalLinear(DeviceTensor &B, const DeviceTensorEx &Ax);
    static bool DequantizeQ8_Log(DeviceTensor &B, const DeviceTensorEx &Ax);
};

INFER_FLOW_END
