#pragma once

#include "sslib/blocked_allocator.h"
#include "host_tensor.h"

INFER_FLOW_BEGIN

using std::vector;
using std::map;
using sslib::IdWeight;
using sslib::BlockedAllocator;

// Parameters of position embedding
struct PosEmbeddingParams
{
    PositionEmbeddingAlg alg = PositionEmbeddingAlg::ROPE;
    const int *device_token_id_array = nullptr;
    int heads = 0;
    int context_len = 0;
    int dims = 0;
    int mode = 0;
    int order_type = 0;
    float rope_theta = 10000.0f;
    float partial_rotary_factor = 1.0f;
};

struct RowItemForMoe
{
    static const int MAX_SIZE = 8;
    int size = 0;
    IdWeight<float> arr[MAX_SIZE];
};

typedef map<string, TensorNormAlg, StrLessNoCase> TensorNormAlgMap;
typedef map<string, ActivationFn, StrLessNoCase> ActivationFnMap;
typedef map<string, PositionEmbeddingAlg, StrLessNoCase> PositionEmbeddingAlgMap;

class HostTensorOpr
{
public:
    // Elementwise binary operations
    static bool Scale(HostTensor &A, float scale);
    static bool Scale(HostTensor &C, const HostTensor &A, float scale);
    static bool Scale(HostTensor &A, const HostTensor &scale);
    static bool Scale(HostTensor &C, const HostTensor &A, const HostTensor &scale);

    static bool Add(HostTensor &C, const HostTensor &A,
        const HostTensor &B);
    static bool Mul(HostTensor &C, const HostTensor &A,
        const HostTensor &B);

    static bool SoftMax(HostTensor &A);
    static bool SoftMax(HostTensor &B, const HostTensor &A);

    static bool BuildRowsForMoE(vector<RowItemForMoe> &row_items,
        const HostTensor &router_logits, int moe_top_k,
        bool norm_top_k_prob = true);

    //GEMM
    static bool Gemm(HostTensor &C, const HostTensor &A, const HostTensor &B,
        float alpha = 1.0f, float beta = 0, bool is_b_column_major = false);
    static bool GemmSparse(HostTensor &C, const HostTensor &A, const HostSparseMatrix &B,
        float alpha = 1.0f, float beta = 0);

    //GEMV
    static bool Gemv_AX(HostTensor &Y, const HostTensor &A, const HostTensor &X,
        float alpha = 1.0f, float beta = 0);

public:
    //A: source; B: target
    static bool IsCompatible_AB(const HostTensor &A,
        const HostTensor &B, bool be_transpose = false);
    //A, B: sources; B: target
    static bool IsCompatible_ABC(const HostTensor &A,
        const HostTensor &B, const HostTensor &C);

    static bool IsCompatible_MulMat(int a_rows, int a_cols, int b_rows, int b_cols,
        int c_rows, int c_cols, bool is_b_column_major);

    static void InitTensorNormAlgMap(TensorNormAlgMap &the_map);
    static void InitActivationFnMap(ActivationFnMap &the_map);
    static void InitPositionEmbeddingAlgMap(PositionEmbeddingAlgMap &the_map);

protected:
    static bool Gemm_F16_RowMajor(inferflow_fp16 *C, const inferflow_fp16 *A,
        const inferflow_fp16 *B, int M, int N, int K,
        float alpha = 1.0f, float beta = 0);
    static bool Gemm_F16_ColMajor(inferflow_fp16 *C, const inferflow_fp16 *A,
        const inferflow_fp16 *B, int M, int N, int K,
        float alpha = 1.0f, float beta = 0);

    static bool Gemv_AX_F16(inferflow_fp16 *Y, const inferflow_fp16 *A, const inferflow_fp16 *X,
        int rows, int cols, float alpha = 1.0f, float beta = 0);
    static bool Gemv_AX_F32(float *Y, const float *A, const float *X,
        int rows, int cols, float alpha = 1.0f, float beta = 0);
};

INFER_FLOW_END
