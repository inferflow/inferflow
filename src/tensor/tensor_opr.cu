#include "tensor_opr.h"
#include "sslib/log.h"
#include <cublas_v2.h>
//#include "cuPrintf.cu"
#include "common/cuda_util.h"
#include "kernels/unary_tensor_opr.h"
#include "kernels/binary_tensor_opr.h"
#include "kernels/vector_opr.h"
#include "kernels/tensor_quant.h"
#include "kernels/misc_tensor_opr.h"

using namespace sslib;

INFER_FLOW_BEGIN

///////////////////////////////////////////////////////////////////////////////
// class TensorOpr

TensorOpr::TensorOpr()
{
}

TensorOpr::~TensorOpr()
{
}

//static
bool TensorOpr::Assign(DeviceTensor &A, float value)
{
    int batch_size = 1;
    int block_dim = 256;
    int block_grid = (A.size + batch_size * block_dim - 1) / (batch_size * block_dim);

    float *a_data = A.data_f32();
    Tensor_Assign_Kernel<<<block_grid, block_dim>>>(a_data, A.size, value, batch_size);
    cudaDeviceSynchronize(); //Wait for kernels to finish

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::Assign");
    return ret;
}

//static
bool TensorOpr::Assign(DeviceTensor &B, const DeviceTensor &A,
    int a_start_row, int b_start_row, int row_num)
{
    int cx = A.ne[0], cy = A.ne[1], cz = A.ne[2];
    int source_rows = cy * cz;
    int target_rows = B.ne[1] * B.ne[2];
    if (a_start_row < 0 || a_start_row >= source_rows) {
        LogError("Invalid source_start_row: %d", a_start_row);
        return false;
    }
    if (b_start_row < 0 || b_start_row >= target_rows) {
        LogError("Invalid target_start_row: %d", b_start_row);
        return false;
    }

    if (row_num < 0) {
        row_num = source_rows - a_start_row;
    }

    int r1 = row_num * cx;
    int r2 = (target_rows - b_start_row) * B.ne[0];
    if (a_start_row + row_num > source_rows || r1 > r2) {
        LogError("TensorOpr.Assign: row_num %d is too large.", row_num);
        return false;
    }

    //Allow that A and B have different collumn count
    //if (cx != B.ne[0]) {
    //    LogError("A and B have different column count: %d vs. %d.", cx, B.ne[0]);
    //    return false;
    //}

    bool a_is_zy_data = A.is_zy_data;

    int block_size_y = cy ? 4 : 1;
    int block_size_x = 64 / block_size_y;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((cx + block_size_x - 1) / block_size_x, (cy + block_size_y - 1) / block_size_y, cz);
    //cout << "block: " << block << "; grid: " << grid << endl;

    int copy_element_num  = row_num * cx;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16() + a_start_row * cx;
        half *b_data = B.data_f16() + b_start_row * B.ne[0];
        Tensor_Assign2_Kernel<half, half><<<grid, block>>>(a_data, A.ne[0], A.ne[1], A.ne[2],
            a_is_zy_data, b_data, copy_element_num);
    }
    else
    {
        const float *a_data = A.data_f32() + a_start_row * cx;
        float *b_data = B.data_f32() + b_start_row * B.ne[0];
        Tensor_Assign2_Kernel<float, float><<<grid, block>>>(a_data, A.ne[0], A.ne[1], A.ne[2],
            a_is_zy_data, b_data, copy_element_num);
    }

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::Assign");
    return ret;
}

//static
bool TensorOpr::RepeatKV(DeviceTensor &B, const DeviceTensor &A,
    int heads_per_group)
{
    int cx = A.ne[0], cy = A.ne[1], cz = A.ne[2];
    if (B.ne[0] != cx || B.ne[1] != cy || B.ne[2] != heads_per_group * cz)
    {
        LogError("RepeatKV: A and B are not compatible");
        return false;
    }

    bool a_is_zy_data = A.is_zy_data;

    int block_size_y = 4;
    int block_size_x = 64 / block_size_y;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((cx + block_size_x - 1) / block_size_x, (cy + block_size_y - 1) / block_size_y, cz);
    //cout << "block: " << block << "; grid: " << grid << endl;

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        half *b_data = B.data_f16();
        RepeatKV_Kernel<<<grid, block>>>(a_data, cx, cy, cz, a_is_zy_data,
            b_data, heads_per_group);
    }
    else
    {
        const float *a_data = A.data_f32();
        float *b_data = B.data_f32();
        RepeatKV_Kernel<<<grid, block>>>(a_data, cx, cy, cz, a_is_zy_data,
            b_data, heads_per_group);
    }

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::RepeatKV");
    return ret;
}

//static
bool TensorOpr::AssignColumns(DeviceTensor &B, const DeviceTensor &A,
    int start_col_a, int step, int start_col_b, int col_num)
{
    int cx = A.ne[0], cy = A.ne[1], cz = A.ne[2];
    int cx_b = B.ne[0];
    if (col_num <= 0) {
        col_num = cx_b - start_col_b;
    }
    int source_rows = cy * cz;
    int target_rows = B.ne[1] * B.ne[2];
    if (source_rows != target_rows) {
        LogError("A and B have different rows: %d vs. %d", source_rows, target_rows);
        return false;
    }

    if (start_col_a < 0 || start_col_a >= cx) {
        LogError("Invalid start_col_a: %d", start_col_a);
        return false;
    }
    if (start_col_a + col_num > cx) {
        LogError("col_num or start_col_a is too large: %d, %d", col_num, start_col_a);
        return false;
    }

    int block_size_y = cy > 1 ? 4 : 1;
    int block_size_x = 64 / block_size_y;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((col_num + block_size_x - 1) / block_size_x,
        (source_rows + block_size_y - 1) / block_size_y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16() + start_col_a;
        half *b_data = B.data_f16() + start_col_b;
        Tensor_AssignColumns_Kernel<half, half><<<grid, block>>>(a_data, b_data,
            cx, cx_b, source_rows, step);
    }
    else
    {
        const float *a_data = A.data_f32() + start_col_a;
        float *b_data = B.data_f32() + start_col_b;
        Tensor_AssignColumns_Kernel<float, float><<<grid, block>>>(a_data, b_data,
            cx, cx_b, source_rows, step);
    }

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::AssignColumns");
    return ret;
}

//static
bool TensorOpr::AssignColumns(DeviceTensor &A, float value,
    int start_col, int col_num)
{
    int cx = A.ne[0], cy = A.ne[1], cz = A.ne[2];
    if (col_num <= 0) {
        col_num = cx - start_col;
    }
    int rows = cy * cz;

    if (start_col < 0 || start_col >= cx) {
        LogError("Invalid start_col_a: %d", start_col);
        return false;
    }
    if (start_col + col_num > cx) {
        LogError("col_num or start_col is too large: %d, %d", col_num, start_col);
        return false;
    }

    int block_size_y = cy > 1 ? 4 : 1;
    int block_size_x = 64 / block_size_y;
    dim3 block(block_size_x, block_size_y);
    dim3 grid((col_num + block_size_x - 1) / block_size_x,
        (rows + block_size_y - 1) / block_size_y);

    if (A.data_type == ElementType::F16)
    {
        half *a_data = A.data_f16() + start_col;
        Tensor_AssignColumns2_Kernel<half><<<grid, block>>>(a_data, value,
            cx, rows, col_num);
    }
    else
    {
        float *a_data = A.data_f32() + start_col;
        Tensor_AssignColumns2_Kernel<float><<<grid, block>>>(a_data, value,
            cx, rows, col_num);
    }

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::AssignColumns");
    return ret;
}

//static
bool TensorOpr::Reshape(DeviceTensor &A, int new_dim_count,
    int new_ne0, int new_ne1, int new_ne2)
{
    int dims[DeviceTensor::MaxDimCount] = {new_ne0, new_ne1, new_ne2};
    bool ret = Reshape(A, new_dim_count, dims);
    return ret;
}

//static
bool TensorOpr::Reshape(DeviceTensor &A, int new_dim_count,
    int dims[DeviceTensor::MaxDimCount])
{
    if (new_dim_count <= 0 || new_dim_count > DeviceTensor::MaxDimCount) {
        LogError("Invalid new dim_count");
        return false;
    }

    int new_size = 1;
    for (int dim_idx = 0; dim_idx < new_dim_count; dim_idx++)
    {
        if (dims[dim_idx] < 1) {
            LogError("Invalid parameters: dims[%d] = %d", dim_idx, dims[dim_idx]);
            return false;
        }

        new_size *= dims[dim_idx];
    }

    if (A.size != new_size)
    {
        LogError("Inconsistent size: %d (%d, %d, %d) vs. %d (%d, %d, %d)",
            A.size, A.ne[0], A.ne[1], A.ne[2],
            new_size, dims[0], new_dim_count > 1 ? dims[1] : 0,
            new_dim_count > 2 ? dims[2] : 0);
        return false;
    }

    bool ret = A.SetStructure(new_dim_count, dims);
    return ret;
}

//static
bool TensorOpr::Transpose(DeviceTensor &T, const DeviceTensor &S, int alg_id)
{
    bool ret = true;
    switch (alg_id)
    {
    case 1:
        ret = Transpose_Alg1(T, S);
        break;
    case 2:
    default:
        ret = Transpose_Alg2(T, S);
        break;
    }

    return ret;
}

//static
bool TensorOpr::Transpose_Alg1(DeviceTensor &T, const DeviceTensor &S)
{
    if (S.size != T.size) {
        LogError("transpose: S and T are not compatible: %d vs. %d", S.size, T.size);
        return false;
    }

    T.dim = max(2, S.dim);
    T.ne[1] = S.ne[0];
    T.ne[0] = S.ne[1];

    int size = S.size;
    int ne0 = S.ne[0], ne1 = S.ne[1], ne2 = max(1, S.ne[2]);

    //block_dim: threads per block; block_grid: grid of blocks
    int cell_len = 8, block_len = 2;
    int unit_len = cell_len * block_len;
    dim3 block_dim(block_len, block_len);
    dim3 block_grid((ne1 * ne2 + unit_len - 1) / unit_len, (ne0 + unit_len - 1) / unit_len);

    if (S.data_type == ElementType::F16)
    {
        const half *src_data = S.data_f16();
        half *target_data = T.data_f16();

        Tensor_Transpose_Kernel<half, half><<<block_grid, block_dim>>>(
            size, ne0, ne1, ne2, src_data, target_data, cell_len);
    }
    else if (S.data_type == ElementType::F32)
    {
        const float *src_data = S.data_f32();
        float *target_data = T.data_f32();

        Tensor_Transpose_Kernel<float, float><<<block_grid, block_dim>>>(
            size, ne0, ne1, ne2, src_data, target_data, cell_len);
    }
    else
    {
        LogError("Unsupported data type: %d", S.data_type);
        return false;
    }

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::Transpose_Alg1");
    return ret;
}

//static
bool TensorOpr::Transpose_Alg2(DeviceTensor &T, const DeviceTensor &S)
{
    if (S.size != T.size) {
        LogError("transpose: S and T are not compatible: %d vs. %d", S.size, T.size);
        return false;
    }

    int cx = S.Columns(), cy = S.Rows();
    T.SetStructure(S.Rows(), S.Columns());

    int tile_dim = 32, block_rows = 8;
    dim3 block(tile_dim, block_rows);
    dim3 grid((cx + tile_dim - 1) / tile_dim, (cy + tile_dim - 1) / tile_dim);

    if (S.data_type == ElementType::F16)
    {
        const half *src_data = S.data_f16();
        half *target_data = T.data_f16();
        Tensor_Transpose_Alg2_Kernel<half><<<grid, block>>>(src_data, target_data, cx, cy);
    }
    else if (S.data_type == ElementType::F32)
    {
        const float *src_data = S.data_f32();
        float *target_data = T.data_f32();
        Tensor_Transpose_Alg2_Kernel<float><<<grid, block>>>(src_data, target_data, cx, cy);
    }
    else
    {
        LogError("Unsupported data type: %d", S.data_type);
        return false;
    }

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::Transpose_Alg2");
    return ret;
}

bool TensorOpr::TransposeYZ(DeviceTensor &A)
{
    if (A.dim <= 1) {
        return true;
    }

    if (A.is_zy_data) {
        A.is_zy_data = false;
        return true;
    }

    if (A.dim == 2) {
        A.ne[2] = 1;
    }

    int ne_bak = A.ne[1];
    A.ne[1] = A.ne[2];
    A.ne[2] = ne_bak;

    if (A.is_zy_data) {
        A.is_zy_data = false;
        return true;
    }

    if (A.ne[1] > 1 && A.ne[2] > 1) {
        A.is_zy_data = true;;
    }

    return true;
}

bool TensorOpr::VectorDotProduct(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B)
{
    if (A.ne[1] != 1 || A.ne[2] != 1 || B.ne[1] != 1 || B.ne[2] != 1
        || C.ne[1] != 1 || C.ne[2] != 1)
    {
        LogError("The input and output tensors should be vectors");
        return false;
    }

    if (A.size != B.size) {
        LogError("A and B should have the same size");
        return false;
    }

    if (C.size != 1) {
        LogError("C should have size 1");
        return false;
    }

    const float *a_data = A.data_f32();
    const float *b_data = B.data_f32();

    int thread_per_block = 256;
    int cell_len = 4;
    int unit_len = cell_len * thread_per_block;
    int block_per_grid = (A.size + unit_len - 1) / unit_len;

    DeviceTensor target;
    target.New(ElementType::F32, block_per_grid);

    VectorDotProduct_Kernel<<<block_per_grid, thread_per_block>>>(
        target.data_f32(), a_data, b_data, A.size, cell_len);

    bool ret = CudaUtil::DeviceSynchronize("TensorOpr::VectorDotProduct");
    Macro_RetFalseIf(!ret);

    vector<float> host_vec;
    target.CopyToHost(host_vec);

    float sum = 0;
    for (int idx = 0 ; idx < host_vec.size(); idx++) {
        sum += host_vec[idx];
    }

    C.FromHost(&sum, 1);
    return true;
}

//static
bool TensorOpr::LayerNormalization(DeviceTensor &T, const DeviceTensor &S,
    TensorNormAlg alg_id, const DeviceTensor *M, const DeviceTensor *A,
    float multi_base)
{
    bool ret = true;
    switch (alg_id)
    {
    case TensorNormAlg::STD:
        ret = StdNorm(T, S, M, A);
        break;
    case TensorNormAlg::RMS:
        ret = RmsNorm(T, S, M, A, multi_base);
        break;
    case TensorNormAlg::LINEAR:
        ret = LinearNorm(T, S);
        break;
    default:
        break;
    }

    return ret;
}

//static
bool TensorOpr::LinearNorm(DeviceTensor &T, const DeviceTensor &S, float scale)
{
    if (T.dim != S.dim || T.ne[0] != S.ne[0] || T.ne[1] != S.ne[1]
        || T.ne[2] != S.ne[2])
    {
        LogError("linear_norm: S and T are not compatible: %d (%d, %d, %d) vs. %d (%d, %d, %d)",
            T.dim, T.ne[0], T.ne[1], T.ne[2], S.dim, S.ne[0], S.ne[1], S.ne[2]);
        return false;
    }

    if (scale <= 0.0001) {
        scale = (float)sqrt(S.ne[0]);
    }
    bool ret = Scale(T, S, scale);
    return ret;
}

//static
bool TensorOpr::StdNorm(DeviceTensor &T, const DeviceTensor &S,
    const DeviceTensor *M, const DeviceTensor *A)
{
    if (T.dim != S.dim || T.ne[0] != S.ne[0] || T.ne[1] != S.ne[1]
        || T.ne[2] != S.ne[2])
    {
        LogError("std_norm: S and T are not compatible: %d (%d, %d, %d) vs. %d (%d, %d, %d)",
            T.dim, T.ne[0], T.ne[1], T.ne[2], S.dim, S.ne[0], S.ne[1], S.ne[2]);
        return false;
    }

    const float eps = 1e-5f;
    int rows = S.ne[1] * S.ne[2];
    int cols = S.ne[0];
    int rows_mul = M != nullptr ? M->Rows() : 0;
    int rows_add = A != nullptr ? A->Rows() : 0;

    int thread_per_block = min(128, Inferflow_MaxThreadPerBlock);
    int thread_per_row = thread_per_block;
    dim3 block(thread_per_row, 1);
    dim3 grid(1, (rows + block.y - 1) / block.y);

    if (S.data_type == ElementType::F16)
    {
        const half *src_data = S.data_f16();
        half *target_data = T.data_f16();
        const half *mul_data = M != nullptr ? M->data_f16() : nullptr;
        const half *add_data = A != nullptr ? A->data_f16() : nullptr;

        /*if (cols <= 1024 && cols % (4 * WARP_SIZE) == 0)
        {
            block.x = WARP_SIZE;
            Tensor_StdNorm_Half_Alg2S_Kernel<<<grid, block>>>(rows, cols, src_data,
                target_data, eps, rows_mul, mul_data, rows_add, add_data);
        }
        else*/
        {
            Tensor_StdNorm_Kernel<half, half><<<grid, block>>>(rows, cols, src_data,
                target_data, eps, rows_mul, mul_data, rows_add, add_data);
        }
    }
    else
    {
        const float *src_data = S.data_f32();
        float *target_data = T.data_f32();
        const float *mul_data = M != nullptr ? M->data_f32() : nullptr;
        const float *add_data = A != nullptr ? A->data_f32() : nullptr;

        Tensor_StdNorm_Kernel<float, float><<<grid, block>>>(rows, cols, src_data,
            target_data, eps, rows_mul, mul_data, rows_add, add_data);
    }

    bool ret = CudaUtil::DeviceSynchronize("StdNorm");
    return ret;
}

//static
bool TensorOpr::RmsNorm(DeviceTensor &T, const DeviceTensor &S,
    const DeviceTensor *M, const DeviceTensor *A, float multi_base)
{
    if (T.dim != S.dim || T.ne[0] != S.ne[0] || T.ne[1] != S.ne[1]
        || T.ne[2] != S.ne[2])
    {
        LogError("rms_norm: S and T are not compatible: %d (%d, %d, %d) vs. %d (%d, %d, %d)",
            T.dim, T.ne[0], T.ne[1], T.ne[2], S.dim, S.ne[0], S.ne[1], S.ne[2]);
        return false;
    }

    const float eps = 1e-5f;
    int rows = S.ne[1] * S.ne[2];
    int cols = S.ne[0];
    int rows_mul = M != nullptr ? M->Rows() : 0;
    int rows_add = A != nullptr ? A->Rows() : 0;

    int thread_per_block = min(128, Inferflow_MaxThreadPerBlock);
    int thread_per_row = thread_per_block;
    dim3 block(thread_per_row, 1);
    dim3 grid(1, (rows + block.y - 1) / block.y);

    if (S.data_type == ElementType::F16)
    {
        const half *src_data = S.data_f16();
        half *target_data = T.data_f16();
        const half *mul_data = M != nullptr ? M->data_f16() : nullptr;
        const half *add_data = A != nullptr ? A->data_f16() : nullptr;

        Tensor_RmsNorm_Kernel<half, half><<<grid, block>>>(rows, cols, src_data,
            target_data, eps, rows_mul, mul_data, rows_add, add_data, multi_base);
    }
    else
    {
        const float *src_data = S.data_f32();
        float *target_data = T.data_f32();
        const float *mul_data = M != nullptr ? M->data_f32() : nullptr;
        const float *add_data = A != nullptr ? A->data_f32() : nullptr;

        Tensor_RmsNorm_Kernel<float, float><<<grid, block>>>(rows, cols, src_data,
            target_data, eps, rows_mul, mul_data, rows_add, add_data, multi_base);
    }

    bool ret = CudaUtil::DeviceSynchronize("RmsNorm");
    return ret;
}

//static
bool TensorOpr::RmsNorm_Naive(DeviceTensor &T, const DeviceTensor &S)
{
    if (T.dim != S.dim || T.ne[0] != S.ne[0] || T.ne[1] != S.ne[1]
        || T.ne[2] != S.ne[2])
    {
        LogError("rms_norm: S and T are not compatible: %d (%d, %d, %d) vs. %d (%d, %d, %d)",
            T.dim, T.ne[0], T.ne[1], T.ne[2], S.dim, S.ne[0], S.ne[1], S.ne[2]);
        return false;
    }

    //vector<float> host_s, host_t;
    //S.CopyToHost(host_s);
    //T.CopyToHost(host_t);
    //LogKeyInfo("S: [%f, %f, %f...]", host_s[0], host_s[1], host_s[2]);
    //LogKeyInfo("T: [%f, %f, %f...]", host_t[0], host_t[1], host_t[2]);

    const float eps = 1e-6f;

    int rows = S.ne[1] * S.ne[2];
    int cols = S.ne[0];
    //LogKeyInfo("rows: %d, cols: %d", rows, cols);

    int thread_per_block = 64;
    int block_num = (rows + thread_per_block - 1) / thread_per_block;

    if (S.data_type == ElementType::F16)
    {
        const half *src_data = S.data_f16();
        half *target_data = T.data_f16();

        Tensor_RmsNorm_Naive_Kernel<half, half><<<block_num, thread_per_block>>>(
            rows, cols, src_data, target_data, eps);
    }
    else
    {
        const float *src_data = S.data_f32();
        float *target_data = T.data_f32();

        Tensor_RmsNorm_Naive_Kernel<float, float><<<block_num, thread_per_block>>>(
            rows, cols, src_data, target_data, eps);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }

    //host-side codes:
    /*LogKeyInfo("S size: %d, ne0: %d", S.size, ne0);
    for (int i2 = 0; i2 < ne2; i2++)
    {
        for (int i1 = 0; i1 < ne1; i1++)
        {
            LogKeyInfo("offset: %d", i2 * ne0 * ne1 + i1 * ne0);
            const float *src_row = src_data + i2 * ne0 * ne1 + i1 * ne0;

            float sum = 0.0f;
            for (int i0 = 0; i0 < ne0; i0++) {
                sum += (src_row[i0] * src_row[i0]);
            }

            LogKeyInfo("sum: %f", sum);
            float mean = sum / ne0;
            float scale = 1.0f / sqrtf(mean + eps);

            float *target_row = target_data + i2 * ne0 * ne1 + i1 * ne0;
            for (int i0 = 0; i0 < ne0; i0++) {
                target_row[i0] = src_row[i0] * scale;
            }
        }
    }*/

    return true;
}

//static
bool TensorOpr::PositionEmbedding(DeviceTensor &A, const PosEmbeddingParams &params,
    int start_z, int z_num, int base_z)
{
    bool ret = PositionEmbedding(A, A, params, start_z, z_num, base_z);
    return ret;
}

//static
//Please note that A and B may be the same tensor
bool TensorOpr::PositionEmbedding(DeviceTensor &B, const DeviceTensor &A,
    const PosEmbeddingParams &params, int start_z, int z_num, int base_z)
{
    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        return false;
    }

    int rope_cols = (int)(A.ne[0] * params.partial_rotary_factor + 0.5f);
    int rope_dims = (int)(params.dims * params.partial_rotary_factor + 0.5f);
    //LogKeyInfo("params.dims: %d, rope dims: %d", params.dims, rope_dims);

    int y_num = A.ne[1];
    if (z_num < 0) {
        z_num = A.ne[2] - start_z;
    }

    int M = y_num * z_num, N = A.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + 2 * block.x - 1) / (2 * block.x), (M + block.y - 1) / block.y);
    dim3 grid_alibi((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    //LogKeyInfo("is_rope: %s, y_num: %d, z_num: %d, start_z: %d, context_len: %d, dims: %d",
    //    is_rope ? "Y" : "N", y_num, z_num, start_z, params.context_len, params.dims);
    int base_offset = N * y_num * start_z;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16() + base_offset;
        half *b_data = B.data_f16() + base_offset;

        switch (params.alg)
        {
        case PositionEmbeddingAlg::ROPE:
            if (params.order_type == 2)
            {
                PosEmbedding_Rope_Order2_Kernel<half, half><<<grid, block>>>(
                    a_data, b_data, A.ne[0], y_num, z_num, params.context_len,
                    rope_dims, params.mode, params.rope_theta, rope_cols);
            }
            else
            {
                PosEmbedding_Rope_Std_Kernel<half, half><<<grid, block>>>(
                    a_data, b_data, A.ne[0], y_num, z_num, params.context_len,
                    params.dims, params.mode, params.rope_theta);
            }
            break;
        case PositionEmbeddingAlg::ALIBI:
            PosEmbedding_Alibi_Std_Kernel<half, half><<<grid_alibi, block>>>(
                a_data, b_data, A.ne[0], y_num, z_num, base_z,
                params.context_len, params.dims, params.heads);
            break;
        case PositionEmbeddingAlg::SINUSOIDAL:
            PosEmbedding_Sinusoidal1_Order2_Kernel<half, half><<<grid, block>>>(
                a_data, b_data, params.device_token_id_array, A.ne[0],
                y_num, z_num, params.context_len, params.dims);
            break;
        case PositionEmbeddingAlg::SINUSOIDAL2:
        default:
            PosEmbedding_Sinusoidal2_Order2_Kernel<half, half><<<grid, block>>>(
                a_data, b_data, params.device_token_id_array, A.ne[0],
                y_num, z_num, params.context_len, params.dims);
            break;
        }
    }
    else
    {
        const float *a_data = A.data_f32() + base_offset;
        float *b_data = B.data_f32() + base_offset;

        switch (params.alg)
        {
        case PositionEmbeddingAlg::ROPE:
            if (params.order_type == 2)
            {
                PosEmbedding_Rope_Order2_Kernel<float, float><<<grid, block>>>(
                    a_data, b_data, A.ne[0], y_num, z_num, params.context_len,
                    params.dims, params.mode, params.rope_theta, rope_cols);
            }
            else
            {
                PosEmbedding_Rope_Std_Kernel<float, float><<<grid, block>>>(
                    a_data, b_data, A.ne[0], y_num, z_num, params.context_len,
                    params.dims, params.mode, params.rope_theta);
            }
            break;
        case PositionEmbeddingAlg::ALIBI:
            PosEmbedding_Alibi_Std_Kernel<float, float><<<grid_alibi, block>>>(
                a_data, b_data, A.ne[0], y_num, z_num, base_z,
                params.context_len, params.dims, params.heads);
            break;
        case PositionEmbeddingAlg::SINUSOIDAL:
            PosEmbedding_Sinusoidal1_Order2_Kernel<float, float><<<grid, block>>>(
                a_data, b_data, params.device_token_id_array, A.ne[0],
                y_num, z_num, params.context_len, params.dims);
            break;
        case PositionEmbeddingAlg::SINUSOIDAL2:
        default:
            PosEmbedding_Sinusoidal2_Order2_Kernel<float, float><<<grid, block>>>(
                a_data, b_data, params.device_token_id_array, A.ne[0],
                y_num, z_num, params.context_len, params.dims);
            break;
        }
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

//static
bool TensorOpr::DiagMask(DeviceTensor &A, int context_len, float value)
{
    bool ret = DiagMask(A, A, context_len, value);
    return ret;
}

//static
bool TensorOpr::DiagMask(DeviceTensor &B, const DeviceTensor &A,
    int context_len, float value)
{
    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        return false;
    }

    int M = A.Rows(), N = A.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / (block.x), (M + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        half *b_data = B.data_f16();

        Transformer_DiagMask_Kernel<half, half><<<grid, block>>>(a_data, b_data,
            A.ne[0], A.ne[1], A.ne[2], context_len, value);
    }
    else
    {
        const float *a_data = A.data_f32();
        float *b_data = B.data_f32();

        Transformer_DiagMask_Kernel<float, float><<<grid, block>>>(a_data, b_data,
            A.ne[0], A.ne[1], A.ne[2], context_len, value);
    }

    ret = CudaUtil::DeviceSynchronize("DiagMask");
    return ret;
}

//static
bool TensorOpr::AggregateByRow(DeviceTensor &B, const DeviceTensor &A,
    RowAggregationType agg_type)
{
    int b_rows = B.Rows(), b_cols = B.Columns();
    if (B.data == nullptr || b_rows * b_cols != B.size) {
        LogWarning("Invalid tensor B");
        return false;
    }

    int M = A.Rows(), N = A.Columns();
    if (B.size < M) {
        LogKeyInfo("The size of tensor B is too small (%d vs. %d)", B.size, M);
        return false;
    }

    dim3 block(8, 8);
    if (N >= 64)
    {
        block.x = 64;
        block.y = 1;
    }
    dim3 grid(1, (M + block.y - 1) / block.y);

    //unsigned int *mutex = nullptr;
    //cudaMalloc((void**)&mutex, 1 * sizeof(int));

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        half *b_data = B.data_f16();

        Tensor_AggregateByRow_Kernel<half, half><<<grid, block>>>(
            M, N, a_data, b_data, agg_type);
    }
    else
    {
        const float *a_data = A.data_f32();
        float *b_data = B.data_f32();

        Tensor_AggregateByRow_Kernel<float, float><<<grid, block>>>(
            M, N, a_data, b_data, agg_type);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish
    //cudaFree(mutex);
    //mutex = nullptr;

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

//static
bool TensorOpr::SumByRow(DeviceTensor &B, const DeviceTensor &A)
{
    bool ret = AggregateByRow(B, A, RowAggregationType::Sum);
    return ret;
}

//static
bool TensorOpr::MaxByRow(DeviceTensor &B, const DeviceTensor &A)
{
    bool ret = AggregateByRow(B, A, RowAggregationType::Max);
    return ret;
}

//static
bool TensorOpr::MinByRow(DeviceTensor &B, const DeviceTensor &A)
{
    bool ret = AggregateByRow(B, A, RowAggregationType::Min);
    return ret;
}

//static
bool TensorOpr::Activation(DeviceTensor &B, const DeviceTensor &A, ActivationFn fn)
{
    bool ret = false;
    switch (fn)
    {
    case ActivationFn::RELU:
        ret = ReluActivation(B, A);
        break;
    case ActivationFn::GELU:
        ret = GeluActivation(B, A);
        break;
    case ActivationFn::SILU:
        ret = SiluActivation(B, A, false);
        break;
    case ActivationFn::GLU_SILU:
        ret = SiluActivation(B, A, true);
        break;
    default:
        LogError("Activation function %d is not implemented yet.", (int)fn);
        break;
    }

    return ret;
}

//static
bool TensorOpr::ReluActivation(DeviceTensor &B, const DeviceTensor &A)
{
    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        return false;
    }

    int M = A.Rows(), N = A.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        if (B.data_type == ElementType::F16)
        {
            half *b_data = B.data_f16();
            ReluActivation_Kernel<half, half><<<grid, block>>>(
                M, N, a_data, b_data);
        }
        else
        {
            float *b_data = B.data_f32();
            ReluActivation_Kernel<half, float><<<grid, block>>>(
                M, N, a_data, b_data);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        float *b_data = B.data_f32();

        ReluActivation_Kernel<float, float><<<grid, block>>>(
            M, N, a_data, b_data);
    }

    ret = CudaUtil::DeviceSynchronize("ReluActivation");
    return ret;
}

//static
bool TensorOpr::SiluActivation(DeviceTensor &B, const DeviceTensor &A, bool is_glu)
{
    int r = is_glu ? 2 : 1;
    bool is_compatible = A.ne[0] == r * B.ne[0] && A.ne[1] == B.ne[1]
            && A.ne[2] == B.ne[2];
    if (!is_compatible) {
        return false;
    }

    int M = A.Rows(), N = A.Columns() / r;
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        if (B.data_type == ElementType::F16)
	{
            half *b_data = B.data_f16();
            SiluActivation_Kernel<half, half><<<grid, block>>>(
                M, N, a_data, b_data, is_glu);
        }
        else
        {
            float *b_data = B.data_f32();
            SiluActivation_Kernel<half, float><<<grid, block>>>(
                M, N, a_data, b_data, is_glu);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        float *b_data = B.data_f32();

        SiluActivation_Kernel<float, float><<<grid, block>>>(
            M, N, a_data, b_data, is_glu);
    }

    bool ret = CudaUtil::DeviceSynchronize("SiluActivation");
    return ret;
}

//static
bool TensorOpr::GeluActivation(DeviceTensor &B, const DeviceTensor &A)
{
    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        return false;
    }

    int M = A.Rows(), N = A.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        if (B.data_type == ElementType::F16)
        {
            half *b_data = B.data_f16();
            GeluActivation_Kernel<half, half><<<grid, block>>>(
                M, N, a_data, b_data);
        }
        else
        {
            float *b_data = B.data_f32();
            GeluActivation_Kernel<half, float><<<grid, block>>>(
                M, N, a_data, b_data);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        float *b_data = B.data_f32();

        GeluActivation_Kernel<float, float><<<grid, block>>>(
            M, N, a_data, b_data);
    }

    ret = CudaUtil::DeviceSynchronize("GeluActivation");
    return ret;
}

//static
bool TensorOpr::SoftMax(DeviceTensor &A, int diag_mask_prefix_len,
    float mask_value, float scale, DeviceTensor *aux_tensor)
{
    bool ret = SoftMax(A, A, diag_mask_prefix_len, mask_value, scale, aux_tensor);
    return ret;
}

//static
bool TensorOpr::SoftMax(DeviceTensor &B, const DeviceTensor &A,
    int diag_mask_prefix_len, float mask_value, float scale,
    DeviceTensor *aux_tensor)
{
    (void)aux_tensor;
    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        return false;
    }

    //ret = SoftMax_Alg1(B, A, aux_tensor);
    ret = SoftMax_Alg2(B, A, diag_mask_prefix_len, mask_value, scale);
    return ret;
}

//static
bool TensorOpr::SoftMax_Alg1(DeviceTensor &B, const DeviceTensor &A, DeviceTensor *aux_tensor)
{
    DeviceTensor agg_tensor; //agg: aggregation
    DeviceTensor *agg_tensor_ptr = aux_tensor;
    if (aux_tensor == nullptr)
    {
        agg_tensor.New(A.data_type, A.ne[1], A.ne[2]);
        agg_tensor_ptr = &agg_tensor;
    }

    bool ret = MaxByRow(*agg_tensor_ptr, A);
    if (!ret)
    {
        LogError("Failed in calling MaxByRow");
        return false;
    }

    //A.Print(cout, 8, 8, 8, "A:\n") << endl;
    //agg_tensor_ptr->Print(cout, 8, 8, 8, "max:\n") << endl;

    //bool is_neg_infinite = fpclassify(x) == FP_INFINITE && signbit(x) != 0;
    float neg_infinity = -std::numeric_limits<float>::infinity();

    int M = A.Rows(), N = A.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *max_data = agg_tensor_ptr->data_f16();

        if (B.data_type == ElementType::F16)
        {
            half *b_data = B.data_f16();
            Tensor_SoftMaxPre_Kernel<half, half><<<grid, block>>>(
                M, N, a_data, b_data, max_data, neg_infinity);
        }
        else
        {
            float *b_data = B.data_f32();
            Tensor_SoftMaxPre_Kernel<half, float><<<grid, block>>>(
                M, N, a_data, b_data, max_data, neg_infinity);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *max_data = agg_tensor_ptr->data_f32();
        float *b_data = B.data_f32();

        Tensor_SoftMaxPre_Kernel<float, float><<<grid, block>>>(
            M, N, a_data, b_data, max_data, neg_infinity);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    //B.Print(cout, 8, 8, 8, "B:\n") << endl;

    ret = SumByRow(*agg_tensor_ptr, B);
    if (!ret) {
        return false;
    }

    //agg_tensor_ptr->Print(cout, 8, 8, 8, "sum:\n") << endl;

    bool is_reverse = true;
    ret = Scale(B, *agg_tensor_ptr, is_reverse);
    if (!ret) {
        return false;
    }

    //B.Print(cout, 8, 8, 8, "B:\n") << endl;

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool TensorOpr::SoftMax_Alg2(DeviceTensor &B, const DeviceTensor &A,
    int diag_mask_prefix_len, float mask_value, float scale)
{
    float neg_infinity = -std::numeric_limits<float>::infinity();

    int cx = A.ne[0], cy = A.ne[1], cz = A.ne[2];
    dim3 block(WARP_SIZE, 1, 1);
    dim3 grid(1, cy, cz);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();

        if (B.data_type == ElementType::F16)
        {
            half *b_data = B.data_f16();
            Tensor_SoftMax_Alg2_Kernel<<<grid, block>>>(cx, cy, cz, a_data, b_data,
                neg_infinity, diag_mask_prefix_len, mask_value, scale);
        }
        else
        {
            float *b_data = B.data_f32();
            Tensor_SoftMax_Alg2_Kernel<<<grid, block>>>(cx, cy, cz, a_data, b_data,
                neg_infinity, diag_mask_prefix_len, mask_value, scale);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        float *b_data = B.data_f32();
        Tensor_SoftMax_Alg2_Kernel<<<grid, block>>>(cx, cy, cz, a_data, b_data,
            neg_infinity, diag_mask_prefix_len, mask_value, scale);
    }

    bool ret = CudaUtil::DeviceSynchronize("SoftMax_Alg2");
    return ret;
}

//static
bool TensorOpr::Scale(DeviceTensor &A, float scale)
{
    bool ret = Scale(A, A, scale);
    return ret;
}

//static
bool TensorOpr::Scale(DeviceTensor &C, const DeviceTensor &A, float scale)
{
    bool ret = IsCompatible_AB(A, C);
    if (!ret) {
        return false;
    }

    int M = A.Rows(), N = A.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / (block.x), (M + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        half *c_data = C.data_f16();

        Tensor_Scale_Kernel<half, half><<<grid, block>>>(M, N, a_data, scale, c_data);
    }
    else
    {
        const float *a_data = A.data_f32();
        float *c_data = C.data_f32();

        Tensor_Scale_Kernel<float, float><<<grid, block>>>(M, N, a_data, scale, c_data);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

//static
bool TensorOpr::Scale(DeviceTensor &A, const DeviceTensor &scale, bool is_reverse)
{
    bool ret = Scale(A, A, scale, is_reverse);
    return ret;
}

//static
bool TensorOpr::Scale(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &scale, bool is_reverse)
{
    bool ret = IsCompatible_AB(A, C);
    if (!ret) {
        return false;
    }

    int M = A.Rows(), N = A.Columns();
    if (scale.size < M)
    {
        LogError("A is not compatible with the scale tensor");
        return false;
    }

    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / (block.x), (M + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *scale_data = scale.data_f16();
        half *c_data = C.data_f16();

        Tensor_ScaleV_Kernel<half, half, half><<<grid, block>>>(
            M, N, a_data, scale_data, c_data, is_reverse);
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *scale_data = scale.data_f32();
        float *c_data = C.data_f32();

        Tensor_ScaleV_Kernel<float, float, float><<<grid, block>>>(
            M, N, a_data, scale_data, c_data, is_reverse);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

//static
bool TensorOpr::Add(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B, int alg_id)
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

    switch (alg_id)
    {
    case 1:
        ret = Add_Alg1(C, A, B);
        break;
    case 2:
        ret = Add_Alg2(C, A, B);
        break;
    case 3:
    default:
        if (A.data_type == ElementType::F16 && A.ne[0] % 4 == 0) {
            ret = Add_Alg3(C, A, B);
        }
        else {
            ret = Add_Alg1(C, A, B);
        }
        break;
    }

    return ret;
}

//static
bool TensorOpr::Add_Alg1(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B)
{
    int M1 = A.Rows(), M2 = B.Rows(), N = A.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M1 + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        ElementwiseAdd_Alg1_Kernel<half, half, half><<<grid, block>>>(
            M1, M2, N, a_data, b_data, c_data);
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();

        ElementwiseAdd_Alg1_Kernel<float, float, float><<<grid, block>>>(
            M1, M2, N, a_data, b_data, c_data);
    }

    bool ret = CudaUtil::DeviceSynchronize("Add_Alg1");
    return ret;
}

//static
bool TensorOpr::Add_Alg2(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B)
{
    int M1 = A.Rows(), M2 = B.Rows(), N = A.Columns();
    int tile_dim = 32, block_rows = 8;
    dim3 block(tile_dim, block_rows);
    dim3 grid((N + tile_dim - 1) / tile_dim, (M1 + tile_dim - 1) / tile_dim);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        ElementwiseAdd_Alg2_Kernel<half, half, half> <<<grid, block >>>(
            c_data, a_data, b_data, M1, M2, N, tile_dim, block_rows);
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();

        ElementwiseAdd_Alg2_Kernel<float, float, float> << <grid, block >> > (
            c_data, a_data, b_data, M1, M2, N, tile_dim, block_rows);
    }

    bool ret = CudaUtil::DeviceSynchronize("Add_Alg2");
    return ret;
}

//static
bool TensorOpr::Add_Alg3(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B)
{
    int M1 = A.Rows(), M2 = B.Rows(), N = A.Columns();
    int tile_len = 4;
    dim3 block, grid;
    block.y = M1 >= 32 ? 8 : (M1 >= 16 ? 4 : (M1 >= 8 ? 2 : 1));
    block.x = 128 / block.y;
    grid.x = (N / tile_len + block.x - 1) / block.x;
    grid.y = (M1 + block.y - 1) / block.y;

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        ElementwiseAdd_Alg3_Half_Kernel<<<grid, block>>> (
            c_data, a_data, b_data, M1, M2, N, tile_len);
    }
    else
    {
        //const float *a_data = A.data_f32();
        //const float *b_data = B.data_f32();
        //float *c_data = C.data_f32();

        //ElementwiseAdd_Alg3_Float_Kernel<<<grid, block>>> (
        //    c_data, a_data, b_data, M1, M2, N, tile_len);
    }

    bool ret = CudaUtil::DeviceSynchronize("Add_Alg3");
    return ret;
}

//static
bool TensorOpr::Mul(DeviceTensor &C, const DeviceTensor &A,
    const DeviceTensor &B)
{
    bool ret = IsCompatible_AB(C, A);
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
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M1 + block.y - 1) / block.y);

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        const half *b_data = B.data_f16();
        half *c_data = C.data_f16();

        ElementwiseMul_Alg1_Kernel<half, half, half><<<grid, block>>>(
            M1, M2, N, a_data, b_data, c_data);
    }
    else
    {
        const float *a_data = A.data_f32();
        const float *b_data = B.data_f32();
        float *c_data = C.data_f32();

        ElementwiseMul_Alg1_Kernel<float, float, float><<<grid, block>>>(
            M1, M2, N, a_data, b_data, c_data);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool TensorOpr::AddByRowIndex(DeviceTensor &B, const DeviceTensor &A,
    const DeviceTensor &idx_tensor, const DeviceTensor *weight_tensor)
{
    int cx = A.Columns(), cy = A.Rows();
    int tile_len = 4;
    dim3 block, grid;
    block.y = cy >= 32 ? 8 : (cy >= 16 ? 4 : (cy >= 8 ? 2 : 1));
    block.x = 128 / block.y;
    grid.x = (cx / tile_len + block.x - 1) / block.x;
    grid.y = (cy + block.y - 1) / block.y;

    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        half *b_data = B.data_f16();
        const int *idx_data = (const int*)idx_tensor.data;
        const half *weights = weight_tensor != nullptr ? weight_tensor->data_f16() : nullptr;

        AddByRowIdx_Kernel<<<grid, block>>>(b_data, a_data,
            cx, cy, tile_len, idx_data, weights);
    }
    else
    {
        LogError("Not implemented yet.");
    }

    bool ret = CudaUtil::DeviceSynchronize("AddByRowIndex");
    return ret;
}

//static
bool TensorOpr::IsCompatible_AB(const DeviceTensor &A, const DeviceTensor &B,
    bool be_transpose)
{
    int b_rows = B.Rows(), b_cols = B.Columns();
    if (B.data == nullptr || b_rows * b_cols != B.size) {
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
bool TensorOpr::IsCompatible_ABC(const DeviceTensor &A,
    const DeviceTensor &B, const DeviceTensor &C)
{
    int c_rows = C.Rows(), c_cols = C.Columns();
    if (C.data == nullptr || c_rows * c_cols != C.size) {
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
int TensorOpr::CheckElements(int &invalid_pos, const DeviceTensor &tensor)
{
    int M = tensor.Rows(), N = tensor.Columns();
    int block_size = 8;
    dim3 block(block_size, block_size);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    int invalid_count = 0;
    if (tensor.data_type == ElementType::F16)
    {
        const half *a_data = tensor.data_f16();

        Tensor_CheckElements_Kernel<half><<<grid, block>>>(M, N, a_data,
            &invalid_count, &invalid_pos);
    }
    else
    {
        const float *a_data = tensor.data_f32();

        Tensor_CheckElements_Kernel<float><<<grid, block>>>(M, N, a_data,
            &invalid_count, &invalid_pos);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return invalid_count;
    }

    return invalid_count;
}

//static
bool TensorOpr::Quantize(DeviceTensor &B, const DeviceTensor &A)
{
    if (A.data_type != ElementType::F32 && A.data_type != ElementType::F16) {
        LogError("Quantize: The data type of A should be F32 or F16");
        return false;
    }

    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        LogKeyInfo("Quantize: Tensor B is not compatible with A");
        return false;
    }

    switch (B.data_type)
    {
    case ElementType::Q8_GL:
        ret = false;
        break;
    case ElementType::Q8_LOG:
        ret = false;
        break;
    case ElementType::Q8_B32T1:
        ret = QuantizeQ8_B32T1(B, A);
        break;
    case ElementType::Q8_B32T2:
        ret = QuantizeQ8_B32T2(B, A, 2);
        break;
    case ElementType::Q6_B64T1:
        ret = QuantizeQ6_B64T1(B, A);
        break;
    case ElementType::Q5_B32T1:
        ret = QuantizeQ5_B32T1(B, A);
        break;
    case ElementType::Q5_B64T1:
        ret = QuantizeQ5_B64T1(B, A);
        break;
    case ElementType::Q4_B16:
        ret = QuantizeQ4B16(B, A);
        break;
    case ElementType::Q4_B32T1A:
    case ElementType::Q4_B32T1B:
        ret = QuantizeQ4_B32T1(B, A);
        break;
    case ElementType::Q4_B64T1:
        ret = QuantizeQ4_B64T1(B, A);
        break;
    case ElementType::Q3H_B64T1:
        ret = QuantizeQ3H_B64T1(B, A);
        break;
    case ElementType::Q3_B32T1A:
    case ElementType::Q3_B32T1B:
        ret = QuantizeQ3_B32T1(B, A);
        break;
    case ElementType::Q2_B32T1A:
    case ElementType::Q2_B32T1B:
        ret = QuantizeQ2_B32T1(B, A);
        break;
    default:
        ret = false;
        break;
    }

    return ret;
}

//static
bool TensorOpr::QuantizeQ8_B32T1(DeviceTensor &B, const DeviceTensor &A, int alg_id)
{
    (void)alg_id;
    if (B.data_type != ElementType::Q8_B32T1) {
        LogError("QuantizeQ8_B32T1: The data type of B should be Q8_B32T1");
        return false;
    }

    const int quant_block_capacity = Q8B32_CAPACITY;

    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        Tensor_QuantizeQ8_B32T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }
    else
    {
        const float *a_data = A.data_f32();
        Tensor_QuantizeQ8_B32T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ8_B32T1");
    return ret;
}

//static
bool TensorOpr::QuantizeQ8_B32T2(DeviceTensor &B, const DeviceTensor &A, int alg_id)
{
    if (B.data_type != ElementType::Q8_B32T2) {
        LogError("QuantizeQ8_B32T2: The data type of B should be Q8_B32T2");
        return false;
    }

    const int quant_block_capacity = Q8B32_CAPACITY;

    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    switch (alg_id)
    {
    case 1:
        break;
    case 2:
    default:
        block.x = 128;
        block.y = 1;
        block.z = 1;
        grid.x = (N + block.x - 1) / block.x;
        grid.y = M > 0xFFFF ? 4096 : M;
        grid.z = M > 0xFFFF ? ((M + 4095) / 4096) : 1;
        break;
    }

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        switch (alg_id)
        {
        case 1:
            Tensor_QuantizeQ8_B32T2_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
            break;
        case 2:
        default:
            Tensor_QuantizeQ8_B32T2_Alg2_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
            break;
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        switch (alg_id)
        {
        case 1:
            Tensor_QuantizeQ8_B32T2_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
            break;
        case 2:
        default:
            Tensor_QuantizeQ8_B32T2_Alg2_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
            break;
        }
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ8_B32T2");
    return ret;
}

//static
bool TensorOpr::QuantizeQ6_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q6_B64T1) {
        LogError("QuantizeQ6_B64T1: The data type of B should be Q6_B64T1");
        return false;
    }

    const int quant_block_capacity = Q6_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        Tensor_QuantizeQ6_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }
    else
    {
        const float *a_data = A.data_f32();
        Tensor_QuantizeQ6_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ6_B64T1");
    return ret;
}

//static
bool TensorOpr::QuantizeQ5_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q5_B32T1) {
        LogError("QuantizeQ5_B32T1: The data type of B should be Q5_B32T1");
        return false;
    }

    if (A.data_type != ElementType::F32 && A.data_type != ElementType::F16) {
        LogError("QuantizeQ5: The data type of A should be F32 or F16");
        return false;
    }

    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        LogKeyInfo("QuantizeQ5: Tensor B is not compatible with A");
        return false;
    }

    //const int quant_block_size = sizeof(BlockQ5);
    const int quant_block_capacity = Q5B32_CAPACITY;

    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        Tensor_QuantizeQ5_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }
    else
    {
        const float *a_data = A.data_f32();
        Tensor_QuantizeQ5_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

//static
bool TensorOpr::QuantizeQ5_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q5_B64T1) {
        LogError("QuantizeQ5_B64T1: The data type of B should be Q5_B64T1");
        return false;
    }

    const int quant_block_capacity = Q5_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        Tensor_QuantizeQ5_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }
    else
    {
        const float *a_data = A.data_f32();
        Tensor_QuantizeQ5_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ5_B64T1");
    return ret;
}

//static
bool TensorOpr::QuantizeQ4B16(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q4_B16) {
        LogError("QuantizeQ4B16: The data type of B should be Q4_B16");
        return false;
    }

    if (A.data_type != ElementType::F32 && A.data_type != ElementType::F16) {
        LogError("QuantizeQ4B16: The data type of A should be F32 or F16");
        return false;
    }

    bool ret = IsCompatible_AB(A, B);
    if (!ret) {
        LogKeyInfo("QuantizeQ4B16: Tensor B is not compatible with A");
        return false;
    }

    const int quant_block_capacity = Q4B16_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        Tensor_QuantizeQ4B16_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }
    else
    {
        const float *a_data = A.data_f32();
        Tensor_QuantizeQ4B16_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }

    cudaDeviceSynchronize(); //Wait for kernels to finish

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LogKeyInfo("Error in calling kernels: %d (%s)", status, cudaGetErrorString(status));
        return false;
    }
    return true;
}

//static
bool TensorOpr::QuantizeQ4_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q4_B32T1A && B.data_type != ElementType::Q4_B32T1B) {
        LogError("QuantizeQ4B32P16: The data type of B should be Q4_B32T1A or Q4_B32T1B");
        return false;
    }

    const int quant_block_capacity = Q4B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        if (B.data_type == ElementType::Q4_B32T1A)
        {
            Tensor_QuantizeQ4_B32T1A_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_QuantizeQ4_B32T1B_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        if (B.data_type == ElementType::Q4_B32T1A)
        {
            Tensor_QuantizeQ4_B32T1A_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_QuantizeQ4_B32T1B_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ4_B32T1");
    return ret;
}

//static
bool TensorOpr::QuantizeQ4_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q4_B64T1) {
        LogError("QuantizeQ4_B64T1: The data type of B should be Q4_B64T1");
        return false;
    }

    const int quant_block_capacity = Q4_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        Tensor_QuantizeQ4_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }
    else
    {
        const float *a_data = A.data_f32();
        Tensor_QuantizeQ4_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ4_B64T1");
    return ret;
}

//static
bool TensorOpr::QuantizeQ3H_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q3H_B64T1) {
        LogError("QuantizeQ3H_B64T1: The data type of B should be Q3H_B64T1");
        return false;
    }

    const int quant_block_capacity = Q3H_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        Tensor_QuantizeQ3H_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }
    else
    {
        const float *a_data = A.data_f32();
        Tensor_QuantizeQ3H_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)B.bytes_per_row, blocks_per_row);
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ3H_B64T1");
    return ret;
}

//static
bool TensorOpr::QuantizeQ3_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q3_B32T1A && B.data_type != ElementType::Q3_B32T1B) {
        LogError("QuantizeQ3_B32T1: The data type of B should be Q3_B32T1A or Q3_B32T1B");
        return false;
    }

    const int quant_block_capacity = Q3B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        if (B.data_type == ElementType::Q3_B32T1A)
        {
            Tensor_QuantizeQ3_B32T1A_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_QuantizeQ3_B32T1B_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        if (B.data_type == ElementType::Q3_B32T1A)
        {
            Tensor_QuantizeQ3_B32T1A_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_QuantizeQ3_B32T1B_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ3_B32T1");
    return ret;
}

//static
bool TensorOpr::QuantizeQ2_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    if (B.data_type != ElementType::Q2_B32T1A && B.data_type != ElementType::Q2_B32T1B) {
        LogError("QuantizeQ2_B32T1: The data type of B should be Q2_B32T1A or Q2_B32T1B");
        return false;
    }

    const int quant_block_capacity = Q2B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    uint8_t *b_data = (uint8_t*)B.data;
    if (A.data_type == ElementType::F16)
    {
        const half *a_data = A.data_f16();
        if (B.data_type == ElementType::Q2_B32T1A)
        {
            Tensor_QuantizeQ2_B32T1A_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_QuantizeQ2_B32T1B_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
    }
    else
    {
        const float *a_data = A.data_f32();
        if (B.data_type == ElementType::Q2_B32T1A)
        {
            Tensor_QuantizeQ2_B32T1A_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_QuantizeQ2_B32T1B_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)B.bytes_per_row, blocks_per_row);
        }
    }

    bool ret = CudaUtil::DeviceSynchronize("QuantizeQ2_B32T1");
    return ret;
}

//static
bool TensorOpr::Dequantize(DeviceTensor &B, const DeviceTensorEx &Ax,
    bool be_transpose, bool be_sync, int alg_id)
{
    const auto &A = *Ax.tensor;

    if (B.data_type != ElementType::F32 && B.data_type != ElementType::F16) {
        LogError("The data type of B should be F32 or F16");
        return false;
    }

    bool ret = IsCompatible_AB(A, B, be_transpose);
    if (!ret) {
        LogError("Dequantize: Tensor B is not compatible with A");
        return false;
    }

    switch (A.data_type)
    {
    case ElementType::Q8_GL:
        ret = DequantizeQ8_GlobalLinear(B, Ax);
        break;
    case ElementType::Q8_LOG:
        ret = DequantizeQ8_Log(B, Ax);
        break;
    case ElementType::Q8_B32T1:
        ret = DequantizeQ8_B32T1(B, A);
        break;
    case ElementType::Q8_B32T2:
        ret = DequantizeQ8_B32T2(B, A);
        break;
    case ElementType::Q6_B64T1:
        ret = DequantizeQ6_B64T1(B, A);
        break;
    case ElementType::Q5_B32T1:
        ret = DequantizeQ5_B32T1(B, A, be_transpose, alg_id);
        break;
    case ElementType::Q5_B64T1:
        ret = DequantizeQ5_B64T1(B, A);
        break;
    case ElementType::Q4_B16:
        ret = DequantizeQ4B16(B, A);
        break;
    case ElementType::Q4_B32T1A:
    case ElementType::Q4_B32T1B:
        if (alg_id == 1) {
            ret = DequantizeQ4_B32T1_Alg1(B, A);
        }
        else {
            ret = DequantizeQ4_B32T1(B, A);
        }
        break;
    case ElementType::Q4_B64T1:
        ret = DequantizeQ4_B64T1(B, A);
        break;
    case ElementType::Q3H_B64T1:
        ret = DequantizeQ3H_B64T1(B, A);
        break;
    case ElementType::Q3_B32T1A:
    case ElementType::Q3_B32T1B:
        ret = DequantizeQ3_B32T1(B, A);
        break;
    case ElementType::Q2_B32T1A:
    case ElementType::Q2_B32T1B:
        ret = DequantizeQ2_B32T1(B, A);
        break;
    default:
        ret = false;
        break;
    }

    if (ret && be_sync)
    {
        ///Wait for kernels to finish
        ret = CudaUtil::DeviceSynchronize();
    }
    return ret;
}

//static
bool TensorOpr::Dequantize(DeviceTensor &B, const DeviceTensor &A, int alg_id)
{
    DeviceTensorEx Ax;
    Ax.tensor = (DeviceTensor*)&A;
    bool be_transpose = false;
    bool be_sync = true;
    bool ret = Dequantize(B, Ax, be_transpose, be_sync, alg_id);
    return ret;
}

//static
bool TensorOpr::DequantizeQ8_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q8_B32T1) {
        LogError("The data type of A should be Q8_B32T1");
        return false;
    }

    const int quant_block_capacity = Q8B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(32, 4);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ8_B32T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ8_B32T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ8_B32T2(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q8_B32T2) {
        LogError("The data type of A should be Q8_B32T2");
        return false;
    }

    const int quant_block_capacity = Q8B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(32, 4);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ8_B32T2_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ8_B32T2_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ6_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q6_B64T1) {
        LogError("The data type (%d) of A should be Q6_B64T1", A.data_type);
        return false;
    }

    const int quant_block_capacity = Q6_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(16, 8);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ6_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ6_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ5_B32T1(DeviceTensor &B, const DeviceTensor &A,
    bool be_transpose, int alg_id)
{
    bool ret = true;
    switch (alg_id)
    {
    case 1:
        ret = DequantizeQ5_Alg1(B, A);
        break;
    case 2:
        ret = DequantizeQ5_Alg2(B, A, be_transpose);
        break;
    default:
        ret = DequantizeQ5_Alg2(B, A, be_transpose);
        break;
    }

    return ret;
}

//static
bool TensorOpr::DequantizeQ5_Alg1(DeviceTensor &B, const DeviceTensor &A)
{
    if (A.data_type != ElementType::Q5_B32T1) {
        LogError("DequantizeQ5_Alg1: The data type of A should be Q5_B32T1");
        return false;
    }

    //const int quant_block_size = sizeof(BlockQ5);
    const int quant_block_capacity = Q5B32_CAPACITY;

    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(4, 32);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ5_Half_Kernel<<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        //float *b_data = B.data_f32();
        //Tensor_DequantizeQ5_Float_Kernel<<<grid, block>>>(a_data, b_data,
        //    M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ5_Alg2(DeviceTensor &B, const DeviceTensor &A, bool be_transpose)
{
    if (A.data_type != ElementType::Q5_B32T1) {
        LogError("DequantizeQ5_Alg2: The data type of A should be Q5_B32T1");
        return false;
    }

    if (B.data_type != ElementType::F32 && B.data_type != ElementType::F16) {
        LogError("The data type of B should be F32 or F16");
        return false;
    }

    bool ret = IsCompatible_AB(A, B, be_transpose);
    if (!ret) {
        LogKeyInfo("DequantizeQ5_Alg2: Tensor B is not compatible with A");
        return false;
    }

    //const int quant_block_size = sizeof(BlockQ5_B32T1);
    const int quant_block_capacity = Q5B32_CAPACITY;

    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(16, 8);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        if (be_transpose)
        {
            Tensor_DequantizeQ5_Alg2_Transpose_Kernel<half><<<grid, block>>>(
                a_data, b_data, M, N, (int)A.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_DequantizeQ5_Alg2_Kernel<half><<<grid, block>>>(a_data, b_data,
                M, N, (int)A.bytes_per_row, blocks_per_row);
        }
    }
    else
    {
        float *b_data = B.data_f32();
        if (be_transpose)
        {
            Tensor_DequantizeQ5_Alg2_Transpose_Kernel<float><<<grid, block>>>(
                a_data, b_data, M, N, (int)A.bytes_per_row, blocks_per_row);
        }
        else
        {
            Tensor_DequantizeQ5_Alg2_Kernel<float><<<grid, block>>>(a_data, b_data,
                M, N, (int)A.bytes_per_row, blocks_per_row);
        }
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ5_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q5_B64T1) {
        LogError("The data type (%d) of A should be Q5_B64T1", A.data_type);
        return false;
    }

    const int quant_block_capacity = Q5_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(16, 8);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ5_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ5_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ4B16(DeviceTensor &B, const DeviceTensor &A)
{
    bool be_transpose = false;
    if (A.data_type != ElementType::Q4_B16) {
        LogError("The data type of A should be Q4_B16");
        return false;
    }

    if (B.data_type != ElementType::F32 && B.data_type != ElementType::F16) {
        LogError("The data type of B should be F32 or F16");
        return false;
    }

    bool ret = IsCompatible_AB(A, B, be_transpose);
    if (!ret) {
        LogKeyInfo("DequantizeQ4B16: Tensor B is not compatible with A");
        return false;
    }

    const int quant_block_capacity = Q4B16_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ4B16_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ4B16_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ4_B32T1_Alg1(DeviceTensor &B, const DeviceTensor &A)
{
    if (A.data_type != ElementType::Q4_B32T1A && A.data_type != ElementType::Q4_B32T1B) {
        LogError("The data type of A should be Q4_B32T1A or Q4_B32T1B");
        return false;
    }

    const int quant_block_capacity = Q4B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(4, 32);
    dim3 grid((blocks_per_row + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ4_Half_Kernel<<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        //float *b_data = B.data_f32();
        //Tensor_DequantizeQ4_Float_Kernel<<<grid, block>>>(a_data, b_data,
        //    M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ4_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q4_B32T1A && A.data_type != ElementType::Q4_B32T1B) {
        LogError("The data type of A should be Q4_B32T1A or Q4_B32T1B");
        return false;
    }

    const int quant_block_capacity = Q4B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(16, 8);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ4_B32T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ4_B32T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ4_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q4_B64T1) {
        LogError("The data type (%d) of A should be Q4_B64T1", A.data_type);
        return false;
    }

    const int quant_block_capacity = Q4_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(16, 8);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ4_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ4_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ3H_B64T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q3H_B64T1) {
        LogError("The data type (%d) of A should be Q3H_B64T1", A.data_type);
        return false;
    }

    const int quant_block_capacity = Q3H_B64_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(16, 8);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ3H_B64T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ3H_B64T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ3_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q3_B32T1A && A.data_type != ElementType::Q3_B32T1B) {
        LogError("The data type (%d) of A should be Q3_B32T1A or Q3_B32T1B", A.data_type);
        return false;
    }

    const int quant_block_capacity = Q3B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ3_B32T1_Kernel<half><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ3_B32T1_Kernel<float><<<grid, block>>>(a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ2_B32T1(DeviceTensor &B, const DeviceTensor &A)
{
    //bool be_transpose = false;
    if (A.data_type != ElementType::Q2_B32T1A && A.data_type != ElementType::Q2_B32T1B) {
        LogError("The data type (%d) of A should be Q2_B32T1A or Q2_B32T1B", A.data_type);
        return false;
    }

    const int quant_block_capacity = Q2B32_CAPACITY;
    int M = A.Rows(), N = A.Columns();
    if (N % quant_block_capacity != 0) {
        LogError("The number of columns should be a multiple of %d", quant_block_capacity);
        return false;
    }

    int blocks_per_row = N / quant_block_capacity;
    dim3 block(8, 16);
    dim3 grid(blocks_per_row, (M + block.y - 1) / block.y);
    //cout << "grid: " << grid << "; block: " << block << endl;

    const uint8_t *a_data = (const uint8_t*)A.data;
    if (B.data_type == ElementType::F16)
    {
        half *b_data = B.data_f16();
        Tensor_DequantizeQ2_B32T1_Kernel<half> << <grid, block >> > (a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }
    else
    {
        float *b_data = B.data_f32();
        Tensor_DequantizeQ2_B32T1_Kernel<float> << <grid, block >> > (a_data, b_data,
            M, N, (int)A.bytes_per_row, blocks_per_row);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ8_GlobalLinear(DeviceTensor &B, const DeviceTensorEx &Ax)
{
    const auto &A = *Ax.tensor;
    if (A.data_type != ElementType::Q8_GL) {
        LogError("The data type of A should be Q8_GL");
        return false;
    }

    if (B.data_type != ElementType::F32 && B.data_type != ElementType::F16) {
        LogError("The data type of B should be F32 or F16");
        return false;
    }

    bool be_transpose = false;
    bool ret = IsCompatible_AB(A, B, be_transpose);
    if (!ret) {
        LogKeyInfo("DequantizeQ8: Tensor B is not compatible with A");
        return false;
    }

    int cx = A.ne[0], cy = A.Rows();
    //int stride_x = cx % 16 == 0 ? 16 : (cx % 8 == 0 ? 8 : 4);
    int stride_x = cx % 8 == 0 ? 8 : 4;
    cx /= stride_x;
    dim3 block(8, 8);
    dim3 grid((cx + block.x - 1) / block.x, (cy + block.y - 1) / block.y);

    const auto &quant = Ax.linear_quant_params;
    if (B.data_type == ElementType::F16)
    {
        const half *quant_map = nullptr; /*Ax.quant_map == nullptr
            ? nullptr : Ax.quant_map->data_f16();*/
        if (stride_x == 4)
        {
            const uint32_t *a_data = (const uint32_t*)A.data;
            uint64_t *b_data = (uint64_t*)B.data_f16();
            Tensor_DequantizeQ8_S4_Kernel<<<grid, block>>>(a_data, b_data,
                cx, cy, quant.z, quant.scale1, quant.scale2, quant_map);
        }
        else
        {
            const uint64_t *a_data = (const uint64_t*)A.data;
            uint64_t *b_data = (uint64_t*)B.data_f16();
            if (stride_x == 8)
            {
                Tensor_DequantizeQ8_S8_Kernel<<<grid, block>>>(a_data, b_data,
                    cx, cy, quant.z, quant.scale1, quant.scale2, quant_map);
            }
            else
            {
                Tensor_DequantizeQ8_S16_Kernel<<<grid, block>>>(a_data, b_data,
                    cx, cy, quant.z, quant.scale1, quant.scale2);
            }
        }
    }
    else
    {
        const uint8_t *a_data = (const uint8_t*)A.data;
        float *b_data = B.data_f32();
        Tensor_DequantizeQ8_Kernel<float><<<grid, block>>>(a_data, b_data,
            cx, cy, stride_x, quant.z, quant.scale1, quant.scale2);
    }

    return true;
}

//static
bool TensorOpr::DequantizeQ8_Log(DeviceTensor &B, const DeviceTensorEx &Ax)
{
    const auto &A = *Ax.tensor;
    if (A.data_type != ElementType::Q8_LOG) {
        LogError("The data type of A should be Q8_LOG");
        return false;
    }

    if (B.data_type != ElementType::F32 && B.data_type != ElementType::F16) {
        LogError("The data type of B should be F32 or F16");
        return false;
    }

    bool be_transpose = false;
    bool ret = IsCompatible_AB(A, B, be_transpose);
    if (!ret) {
        LogKeyInfo("DequantizeQ8: Tensor B is not compatible with A");
        return false;
    }

    int cx = A.ne[0], cy = A.Rows();
    int stride_x = cx % 8 == 0 ? 8 : 4;
    cx /= stride_x;
    dim3 block(16, 8);
    dim3 grid((cx + block.x - 1) / block.x, (cy + block.y - 1) / block.y);

    const auto &params = Ax.log_quant_params;
    if (B.data_type == ElementType::F16)
    {
        const half *quant_map = nullptr; /*Ax.quant_map == nullptr
            ? nullptr : Ax.quant_map->data_f16();*/
        if (stride_x == 4)
        {
            LogWarning("Not implemented yet.");
            //const uint32_t *a_data = (const uint32_t*)A.data;
            //uint64_t *b_data = (uint64_t*)B.data_f16();
            //Tensor_DequantizeQ8_Log_S4_Kernel<<<grid, block>>>(a_data, b_data,
            //    cx, cy, params.base, params.scale, params.start, quant_map);
        }
        else
        {
            const uint64_t *a_data = (const uint64_t*)A.data;
            uint64_t *b_data = (uint64_t*)B.data_f16();
            Tensor_DequantizeQ8_Log_S8_Kernel<<<grid, block>>>(a_data, b_data,
                cx, cy, params.base, params.scale, params.start, quant_map);
        }
    }
    else
    {
        //const uint8_t *a_data = (const uint8_t*)A.data;
        //float *b_data = B.data_f32();
        //Tensor_DequantizeQ8_Log_Kernel<float><<<grid, block>>>(a_data, b_data,
        //    cx, cy, stride_x, params.base, params.scale, params.start);
    }

    return true;
}

INFER_FLOW_END
