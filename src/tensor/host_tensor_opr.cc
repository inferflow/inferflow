#include "host_tensor_opr.h"
#include "sslib/log.h"

using namespace sslib;

INFER_FLOW_BEGIN

//static
bool HostTensorOpr::Scale(HostTensor &A, float scale)
{
    bool ret = Scale(A, A, scale);
    return ret;
}

//static
bool HostTensorOpr::Scale(HostTensor &C, const HostTensor &A, float scale)
{
    bool ret = IsCompatible_AB(A, C);
    if (!ret) {
        return false;
    }

    int element_num = (int)A.size;
    if (A.data_type == ElementType::F16)
    {
        const inferflow_fp16 *data_a = A.data_f16();
        inferflow_fp16 *data_c = C.data_f16();
        for (int idx = 0; idx < element_num; idx++)
        {
            data_c[idx] = (inferflow_fp16)((float)data_a[idx] * scale);
        }
    }
    else
    {
        const float *data_a = A.data_f32();
        float *data_c = C.data_f32();
        for (int idx = 0; idx < element_num; idx++)
        {
            data_c[idx] = data_a[idx] * scale;
        }
    }

    return true;
}

//static
bool HostTensorOpr::Scale(HostTensor &A, const HostTensor &scale)
{
    bool ret = Scale(A, A, scale);
    return ret;
}

//static
bool HostTensorOpr::Scale(HostTensor &C, const HostTensor &A, const HostTensor &scale)
{
    (void)C; (void)A; (void)scale;
    /*bool ret = IsCompatible_AB(A, C);
    if (!ret) {
        return false;
    }

    int M = A.Rows(), N = A.Columns();
    if (scale.size < M)
    {
        LogError("A is not compatible with the scale tensor");
        return false;
    }

    if (A.data_type == ElementType::F16)
    {
    }
    else
    {
    }*/

    return true;
}

//static
bool HostTensorOpr::Add(HostTensor &C, const HostTensor &A, const HostTensor &B)
{
    bool ret = IsCompatible_AB(C, A);
    if (!ret) {
        LogError("Add: C is not compatible with A");
        return false;
    }

    ret = A.ne[0] == B.ne[0] && A.ne[1] % B.ne[1] == 0 && A.ne[2] % B.ne[2] == 0;
    if (!ret)
    {
        LogError("Add: A is not compatible with B: (%d, %d, %d) vs. (%d, %d, %d)",
            A.ne[0], A.ne[1], A.ne[2], B.ne[0], B.ne[1], B.ne[2]);
        return false;
    }


    return ret;
}

//static
bool HostTensorOpr::Mul(HostTensor &C, const HostTensor &A, const HostTensor &B)
{
    (void)C; (void)A; (void)B;
    /*bool ret = IsCompatible_AB(C, A);
    if (!ret) {
        LogError("Mul: C is not compatible with A");
        return false;
    }

    ret = A.ne[0] == B.ne[0] && A.ne[1] % B.ne[1] == 0 && A.ne[2] % B.ne[2] == 0;
    if (!ret)
    {
        LogError("Mul: A is not compatible with B: (%d, %d, %d) vs. (%d, %d, %d)",
            A.ne[0], A.ne[1], A.ne[2], B.ne[0], B.ne[1], B.ne[2]);
        return false;
    }

    int M1 = A.Rows(), M2 = B.Rows(), N = A.Columns();

    if (A.data_type == ElementType::F16)
    {
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();
    }*/

    return true;
}

//static
bool HostTensorOpr::SoftMax(HostTensor &A)
{
    return SoftMax(A, A);
}

//static
bool HostTensorOpr::SoftMax(HostTensor &B, const HostTensor &A)
{
    int rows = A.Rows(), cols = A.Columns();
    int rows_b = B.Rows(), cols_b = B.Columns();
    if (rows != rows_b || cols != cols_b)
    {
        LogError("A and B are not compatible: (%d, %d) vs. (%d, %d)",
            cols, rows, cols_b, rows_b);
        return false;
    }

    if (A.data_type != ElementType::F16 || B.data_type != ElementType::F16)
    {
        LogError("Only FP16 type is supported so far");
        return false;
    }

    for (int row = 0; row < rows; row++)
    {
        const inferflow_fp16 *src_row = (const inferflow_fp16*)A.RowData(row);
        inferflow_fp16 *target_row = (inferflow_fp16*)B.RowData(row);
        float max_value = src_row[0];
        for (int col = 1; col < cols; col++)
        {
            const float val = src_row[col];
            if (max_value < val) {
                max_value = val;
            }
        }

        float sum = 0;
        for (int col = 0; col < cols; col++)
        {
            float val = src_row[col];
            val = expf(val - max_value);
            sum += val;
            target_row[col] = val;
        }

        const float scale = 1.0f / sum;
        for (int col = 0; col < cols; col++)
        {
            target_row[col] = target_row[col] * (inferflow_fp16)scale;
        }
    }

    return true;
}

//static
bool HostTensorOpr::BuildRowsForMoE(vector<RowItemForMoe> &row_items,
    const HostTensor &router_logits, int moe_top_k, bool norm_top_k_prob)
{
    int token_num = router_logits.ne[1];
    int expert_num = router_logits.ne[0];
    row_items.clear();
    row_items.resize(token_num);
    moe_top_k = std::min(moe_top_k, RowItemForMoe::MAX_SIZE);

    for (int row_idx = 0; row_idx < token_num; row_idx++)
    {
        const auto *src_row = (const inferflow_fp16*)router_logits.RowData(row_idx);
        auto &item = row_items[row_idx];

        int arr_len = 0;
        for (int expert_id = 0; expert_id < expert_num; expert_id++)
        {
            const float score = src_row[expert_id];
            if (arr_len >= moe_top_k && item.arr[arr_len - 1].weight >= score
                || score < 0.00001f)
            {
                continue;
            }

            if (arr_len < moe_top_k)
            {
                item.arr[arr_len].Set(UINT32_MAX, 0);
                arr_len++;
            }

            for (int e_idx = arr_len - 1; e_idx >= 0; e_idx--)
            {
                if (item.arr[e_idx].weight >= score) {
                    break;
                }

                if (e_idx + 1 < arr_len) {
                    item.arr[e_idx + 1] = item.arr[e_idx];
                }
                item.arr[e_idx].Set(expert_id, score);
            }
        }

        item.size = arr_len;
        if (norm_top_k_prob)
        {
            float sum = 0;
            for (int idx = 0; idx < arr_len; idx++) {
                sum += item.arr[idx].weight;
            }
            for (int idx = 0; idx < arr_len; idx++) {
                item.arr[idx].weight /= sum;
            }
        }
    }

    /*for (int row_idx = 0; row_idx < token_num; row_idx++)
    {
        const auto *src_row = (const inferflow_fp16*)router_logits.RowData(row_idx);
        auto &item = row_items[row_idx];

        int arr_len = 0;
        for (int expert_id = 0; expert_id < expert_num; expert_id++)
        {
            const float score = src_row[expert_id];
            if (arr_len == 0)
            {
                item.arr[0].Set(expert_id, score);
                arr_len = 1;
            }
            else if (arr_len == 1)
            {
                if (item.arr[0].weight < score)
                {
                    if (moe_top_k == 1)
                    {
                        item.arr[0].Set(expert_id, score);
                    }
                    else
                    {
                        item.arr[1] = item.arr[0];
                        item.arr[0].Set(expert_id, score);
                        arr_len++;
                    }
                }
                else
                {
                    if (moe_top_k > 1)
                    {
                        item.arr[1].Set(expert_id, score);
                        arr_len++;
                    }
                }
            }
            else
            {
                if (item.arr[0].weight < score)
                {
                    item.arr[1] = item.arr[0];
                    item.arr[0].Set(expert_id, score);
                }
                else if (item.arr[1].weight < score)
                {
                    item.arr[1].Set(expert_id, score);
                }
            }
        }

        item.size = arr_len;
        if (norm_top_k_prob)
        {
            float sum = 0;
            for (int idx = 0; idx < arr_len; idx++) {
                sum += item.arr[idx].weight;
            }
            for (int idx = 0; idx < arr_len; idx++) {
                item.arr[idx].weight /= sum;
            }
        }
    }*/

    return true;
}

//static
bool HostTensorOpr::Gemm(HostTensor &C, const HostTensor &A, const HostTensor &B,
    float alpha, float beta, bool is_b_column_major)
{
    (void)C; (void)A; (void)B; (void)alpha; (void)beta; (void)is_b_column_major;
    bool ret = true;
    /*int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();

    bool is_compatible = IsCompatible_MulMat(a_rows, a_cols,
        b_rows, b_cols, c_rows, c_cols, is_b_column_major);
    if (!is_compatible) {
        return false;
    }

    if (C.data == nullptr || c_rows * c_cols != C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    int M = a_rows, K = a_cols;
    int N = is_b_column_major ? b_rows : b_cols;

    if (A.data_type == ElementType::F16)
    {
        const auto *data_a = A.data_f16();
        const auto *data_b = B.data_f16();
        auto *data_c = C.data_f16();
        if (is_b_column_major) {
            ret = Gemm_F16_ColMajor(data_c, data_a, data_b, M, N, K, alpha, beta);
        }
        else {
            ret = Gemm_F16_RowMajor(data_c, data_a, data_b, M, N, K, alpha, beta);
        }
    }
    else
    {
        const auto *data_a = A.data_f32();
        const auto *data_b = B.data_f32();
        auto *data_c = C.data_f32();
        if (is_b_column_major) {
            //ret = Gemm_F32_ColMajor(data_c, data_a, data_b, M, N, K, alpha, beta);
        }
        else {
            //ret = Gemm_F32_RowMajor(data_c, data_a, data_b, M, N, K, alpha, beta);
        }
    }*/

    return ret;
}

// A (M, K), B (K, N), C(M, N)
bool HostTensorOpr::Gemm_F16_RowMajor(inferflow_fp16 *C, const inferflow_fp16 *A,
    const inferflow_fp16 *B, int M, int N, int K, float alpha, float beta)
{
    (void)alpha; (void)beta;
    int row_idx = 0, col_idx = 0; //row and column of C
//#   pragma omp parallel for private(row_idx)
    for (row_idx = 0; row_idx < M; row_idx++)
    {
//#       pragma omp parallel for private(col_idx)
        for (col_idx = 0; col_idx < N; col_idx++)
        {
            float value = 0.0f;
            for (int k_idx = 0; k_idx < K; k_idx++)
            {
                value += (float)A[row_idx * K + k_idx] * (float)B[k_idx * N + col_idx];
            }
            C[row_idx * N + col_idx] = value;
        }
    }

    return true;
}

// A (M, K), B (N, K), C(M, N)
bool HostTensorOpr::Gemm_F16_ColMajor(inferflow_fp16 *C, const inferflow_fp16 *A,
    const inferflow_fp16 *B, int M, int N, int K, float alpha, float beta)
{
    (void)alpha; (void)beta;
    int row_idx = 0, col_idx = 0; //row and column of C
//#   pragma omp parallel for private(row_idx)
    for (row_idx = 0; row_idx < M; row_idx++)
    {
//#       pragma omp parallel for private(col_idx)
        for (col_idx = 0; col_idx < N; col_idx++)
        {
            float value = 0.0f;
            for (int k_idx = 0; k_idx < K; k_idx++)
            {
                value += (float)A[row_idx * K + k_idx] * (float)B[col_idx * K + k_idx];
            }
            C[row_idx * N + col_idx] = value;
        }
    }

    return true;
}

//static
bool HostTensorOpr::GemmSparse(HostTensor &C, const HostTensor &A,
    const HostSparseMatrix &B, float alpha, float beta)
{
    (void)alpha; (void)beta;
    int a_rows = A.Rows(), a_cols = A.Columns();
    int b_rows = B.Rows(), b_cols = B.Columns();
    int c_rows = C.Rows(), c_cols = C.Columns();
    bool is_compatible = a_cols == b_rows && a_rows == c_rows && b_cols == c_cols;
    if (!is_compatible)
    {
        LogError("GemmSparse: A (%d, %d), B (%d, %d), and C (%d, %d) are not compatible.",
            a_cols, a_rows, b_cols, b_rows, c_cols, c_rows);
        return false;
    }

    bool is_beta_zero = beta > -0.00001f && beta < 0.00001f;
    bool is_beta_one = beta > 0.9999f && beta < 1.00001f;
    if (is_beta_zero) {
        C.AssignZero();
    }
    else if (!is_beta_one) {
        HostTensorOpr::Scale(C, beta);
    }

    int cell_num = (int)B.Size();
    const SparseMatrixCell *cell_array = B.Cells();

    float value = 0;
    if (A.data_type == ElementType::F16 && C.data_type == ElementType::F16)
    {
        const auto *data_a = A.data_f16();
        auto *data_c = C.data_f16();
        for (int cell_idx = 0; cell_idx < cell_num; cell_idx++)
        {
            const SparseMatrixCell &cell = cell_array[cell_idx];
            for (int row_idx = 0; row_idx < a_rows; row_idx++)
            {
                value = (float)data_a[row_idx * a_cols + cell.row] * cell.score;
                int offset = row_idx * c_cols + cell.col;
                data_c[offset] = data_c[offset] + (inferflow_fp16)value;
            }
        }
    }
    else if (A.data_type == ElementType::F32 && C.data_type == ElementType::F32)
    {
        const auto *data_a = A.data_f32();
        auto *data_c = C.data_f32();
        for (int cell_idx = 0; cell_idx < cell_num; cell_idx++)
        {
            const SparseMatrixCell &cell = cell_array[cell_idx];
            for (int row_idx = 0; row_idx < a_rows; row_idx++)
            {
                value = data_a[row_idx * a_cols + cell.row] * cell.score;
                data_c[row_idx * c_cols + cell.col] += value;
            }
        }
    }

    return true;
}

//static
bool HostTensorOpr::Gemv_AX(HostTensor &Y, const HostTensor &A, const HostTensor &X,
    float alpha, float beta)
{
    bool ret = true;
    int rows = A.Rows(), cols = A.Columns();
    if ((int)X.size != cols)
    {
        LogError("Gemv_AX: A (%d, %d, %d) and X (%d) are not compatible",
            A.ne[0], A.ne[1], A.ne[2], X.size);
        return false;
    }
    if ((int)Y.size != rows)
    {
        LogError("Gemv_AX: A (%d, %d, %d) and Y (%d) are not compatible",
            A.ne[0], A.ne[1], A.ne[2], Y.size);
        return false;
    }

    if (A.data_type == ElementType::F16)
    {
        const auto *data_a = A.data_f16();
        const auto *data_x = X.data_f16();
        auto *data_y = Y.data_f16();
        ret = Gemv_AX_F16(data_y, data_a, data_x, rows, cols, alpha, beta);
    }
    else
    {
        const auto *data_a = A.data_f32();
        const auto *data_x = X.data_f32();
        auto *data_y = Y.data_f32();
        ret = Gemv_AX_F32(data_y, data_a, data_x, rows, cols, alpha, beta);
    }

    return ret;
}

//static
bool HostTensorOpr::Gemv_AX_F16(inferflow_fp16 *Y, const inferflow_fp16 *A,
    const inferflow_fp16 *X, int rows, int cols, float alpha, float beta)
{
    (void)alpha; (void)beta;
    int row_idx = 0;
//#   pragma omp parallel for private(row_idx)
    for (row_idx = 0; row_idx < rows; row_idx++)
    {
        float value = 0.0f;
        for (int col_idx = 0; col_idx < cols; col_idx++)
        {
            value += (float)A[row_idx * cols + col_idx] * (float)X[col_idx];
        }
        Y[row_idx] = value;
    }

    return true;
}

//static
bool HostTensorOpr::Gemv_AX_F32(float *Y, const float *A, const float *X,
    int rows, int cols, float alpha, float beta)
{
    (void)alpha; (void)beta;
    int row_idx = 0;
//#   pragma omp parallel for private(row_idx)
    for (row_idx = 0; row_idx < rows; row_idx++)
    {
        float value = 0.0f;
        for (int col_idx = 0; col_idx < cols; col_idx++)
        {
            value += A[row_idx * cols + col_idx] * X[col_idx];
        }
        Y[row_idx] = value;
    }

    return true;
}

//static
bool HostTensorOpr::IsCompatible_AB(const HostTensor &A, const HostTensor &B,
    bool be_transpose)
{
    int b_rows = B.Rows(), b_cols = B.Columns();
    if (B.data == nullptr || b_rows * b_cols != (int)B.size) {
        LogWarning("Invalid tensor B");
        return false;
    }

    if (!B.HasCompatibleStructureWith(A, be_transpose))
    {
        LogKeyInfo("Tensor B (%d, %d, %d) is not compatible with A (%d, %d, %d) (transpose: %s)",
            B.ne[0], B.ne[1], B.ne[2], A.ne[0], A.ne[1], A.ne[2], be_transpose ? "Y" : "N");
        return false;
    }

    return true;
}

//static
bool HostTensorOpr::IsCompatible_ABC(const HostTensor &A,
    const HostTensor &B, const HostTensor &C)
{
    int c_rows = C.Rows(), c_cols = C.Columns();
    if (C.data == nullptr || c_rows * c_cols != (int)C.size) {
        LogWarning("Invalid tensor C");
        return false;
    }

    if (!C.HasCompatibleStructureWith(A) || !C.HasCompatibleStructureWith(B)) {
        LogKeyInfo("Tensor C is not compatible with A or B");
        return false;
    }

    return true;
}

//static
bool HostTensorOpr::IsCompatible_MulMat(int a_rows, int a_cols,
    int b_rows, int b_cols, int c_rows, int c_cols,
    bool is_b_column_major)
{
    if (is_b_column_major)
    {
        if (a_cols != b_cols)
        {
            LogWarning("A (%d, %d) and B (%d, %d) are not compatible.",
                a_rows, a_cols, b_rows, b_cols);
            return false;
        }

        if (a_rows != c_rows || b_rows != c_cols)
        {
            LogWarning("C (%d, %d) is not compatible with A (%d, %d) and B (%d, %d).",
                c_rows, c_cols, a_rows, a_cols, b_rows, b_cols);
            return false;
        }
    }
    else
    {
        if (a_cols != b_rows)
        {
            LogWarning("A (%d, %d) and B (%d, %d) are not compatible.",
                a_rows, a_cols, b_rows, b_cols);
            return false;
        }

        if (a_rows != c_rows || b_cols != c_cols)
        {
            LogWarning("C (%d, %d) is not compatible with A (%d, %d) and B (%d, %d).",
                c_rows, c_cols, a_rows, a_cols, b_rows, b_cols);
            return false;
        }
    }

    return true;
}

void HostTensorOpr::InitTensorNormAlgMap(TensorNormAlgMap &the_map)
{
    the_map["std"] = TensorNormAlg::STD;
    the_map["rms"] = TensorNormAlg::RMS;
    the_map["linear"] = TensorNormAlg::LINEAR;
}

void HostTensorOpr::InitActivationFnMap(ActivationFnMap &the_map)
{
    the_map["sigmoid"] = ActivationFn::SIGMOID;
    the_map["elu"] = ActivationFn::ELU;
    the_map["relu"] = ActivationFn::RELU;
    the_map["gelu"] = ActivationFn::GELU;
    the_map["silu"] = ActivationFn::SILU;
    the_map["glu_silu"] = ActivationFn::GLU_SILU;
}

void HostTensorOpr::InitPositionEmbeddingAlgMap(PositionEmbeddingAlgMap &the_map)
{
    the_map[""] = PositionEmbeddingAlg::EMPTY;
    the_map["empty"] = PositionEmbeddingAlg::EMPTY;
    the_map["alibi"] = PositionEmbeddingAlg::ALIBI;
    the_map["rope"] = PositionEmbeddingAlg::ROPE;
    the_map["sinusoidal"] = PositionEmbeddingAlg::SINUSOIDAL;
    the_map["sinusoidal2"] = PositionEmbeddingAlg::SINUSOIDAL2;
}

INFER_FLOW_END
