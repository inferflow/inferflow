#include "host_tensor.h"
#include <algorithm>
#include "sslib/log.h"
#include "common/quantization.h"

INFER_FLOW_BEGIN

using namespace std;
using namespace sslib;

void HostTensor::Clear()
{
    if (this->data != nullptr)
    {
        if (is_auto_free_) {
            delete[] (uint8_t*)this->data;
        }
        this->data = nullptr;
    }

    this->size = 0;

    this->id = 0;
    this->dim = 0;
    this->ne[0] = 0;
    this->ne[1] = 0;
    this->ne[2] = 0;

    bytes_per_row = 0;
    is_zy_data = false;
}

bool HostTensor::SetStructure(int n0, int n1, int n2)
{
    if (n0 <= 0 || n1 < 0 || n2 < 0) {
        return false;
    }

    dim = n1 == 0 ? 1 : (n2 == 0 ? 2 : 3);
    this->ne[0] = n0;
    this->ne[1] = max(1, n1);
    this->ne[2] = max(1, n2);
    this->bytes_per_row = TensorCommon::ByteCount(this->data_type, n0);

    this->size = ne[0] * ne[1] * ne[2];
    return true;
}

bool HostTensor::SetStructure(int dim_count, const int dims[MaxDimCount])
{
    if (dim_count <= 0 || dim_count > MaxDimCount) {
        return false;
    }

    int new_size = 1;
    for (int idx = 0; idx < MaxDimCount; idx++)
    {
        this->ne[idx] = max(1, dims[idx]);
        new_size *= this->ne[idx];
    }

    this->bytes_per_row = TensorCommon::ByteCount(this->data_type, this->ne[0]);
    this->dim = dim_count;
    this->size = new_size;
    return true;
}

const float* HostTensor::data_f32() const
{
    return data_type == ElementType::F32 ? (const float*)this->data : nullptr;
}

float* HostTensor::data_f32()
{
    return data_type == ElementType::F32 ? (float*)this->data : nullptr;
}

const inferflow_fp16* HostTensor::data_f16() const
{
    return data_type == ElementType::F16 ? (const inferflow_fp16*)this->data : nullptr;
}

inferflow_fp16* HostTensor::data_f16()
{
    return data_type == ElementType::F16 ? (inferflow_fp16*)this->data : nullptr;
}

const void* HostTensor::RowData(int row) const
{
    return (const void*)((const uint8_t*)this->data + row * this->bytes_per_row);
}

void* HostTensor::RowData(int row)
{
    return (void*)((uint8_t*)this->data + row * this->bytes_per_row);
}

const void* HostTensor::RowData(int y, int z) const
{
    int row = is_zy_data ? (y * ne[2] + z) : (z * ne[1] + y);
    return (const void*)((const uint8_t*)this->data + row * this->bytes_per_row);
}

void* HostTensor::RowData(int y, int z)
{
    int row = is_zy_data ? (y * ne[2] + z) : (z * ne[1] + y);
    return (void*)((uint8_t*)this->data + row * this->bytes_per_row);
}

void HostTensor::Set(float *src_data, int cx, int cy, int cz, bool arg_is_auto_free)
{
    this->data_type = ElementType::F32;
    this->data = src_data;
    this->is_auto_free_ = arg_is_auto_free;

    SetStructure(cx, cy, cz);
}

void HostTensor::Set(inferflow_fp16 *src_data, int cx, int cy, int cz, bool arg_is_auto_free)
{
    this->data_type = ElementType::F16;
    this->data = src_data;
    this->is_auto_free_ = arg_is_auto_free;

    SetStructure(cx, cy, cz);
}

void HostTensor::Set(ElementType etype, void *src_data,
    int cx, int cy, int cz, bool arg_is_auto_free)
{
    this->data_type = etype;
    this->data = src_data;
    this->is_auto_free_ = arg_is_auto_free;

    SetStructure(cx, cy, cz);
}

bool HostTensor::New(ElementType etype, int n0, int n1, int n2)
{
    Clear();

    this->data_type = etype;
    bool ret = SetStructure(n0, n1, n2);
    if (!ret) {
        return false;
    }

    uint64_t bytes = this->bytes_per_row * this->ne[1] * this->ne[2];
    this->data = new uint8_t[bytes];
    return this->data != nullptr;
}

bool HostTensor::New(ElementType etype, int input_dim, const int *ne_per_dim)
{
    Clear();

    dim = min(input_dim, MaxDimCount);
    int n0 = dim > 0 && ne_per_dim != nullptr ? ne_per_dim[0] : 0;
    int n1 = dim > 1 && ne_per_dim != nullptr ? ne_per_dim[1] : 0;
    int n2 = dim > 2 && ne_per_dim != nullptr ? ne_per_dim[2] : 0;
    bool ret = New(etype, n0, n1, n2);
    return ret;
}

bool HostTensor::NewF32(int n0, int n1, int n2)
{
    return New(ElementType::F32, n0, n1, n2);
}

bool HostTensor::NewF16(int n0, int n1, int n2)
{
    return New(ElementType::F16, n0, n1, n2);
}

bool HostTensor::AssignZero()
{
    uint64_t byte_count = TensorCommon::ByteCount(this->data_type, this->size);
    memset(this->data, 0, byte_count);
    return true;
}

bool HostTensor::HasCompatibleStructureWith(const HostTensor &rhs, bool be_transpose) const
{
    if (be_transpose)
    {
        return this->ne[0] == rhs.ne[1] && this->ne[1] == rhs.ne[0]
            && this->ne[2] == rhs.ne[2] && this->ne[2] == 1;
    }
    else
    {
        return this->ne[0] == rhs.ne[0] && this->ne[1] == rhs.ne[1]
            && this->ne[2] == rhs.ne[2];
    }
}

bool HostTensor::CopyTo(HostTensor &target, int start_row, int rows) const
{
    bool ret = true;
    target.Clear();

    if (start_row == 0 && rows < 0)
    {
        ret = target.New(this->data_type, this->dim, this->ne);
        Macro_RetxFalseIf(!ret, LogError("Failed to create the host tensor"));

        uint64_t bytes = TensorCommon::ByteCount(this->data_type, this->size);
        ret = memcpy(target.data, this->data, bytes);
    }
    else
    {
        int src_rows = Rows();
        start_row = min(max(start_row, 0), rows);
        int target_rows = rows >= 0 ? (rows - start_row) : (src_rows - start_row);
        target.New(this->data_type, this->ne[0], target_rows);

        const void *src_data = this->RowData(start_row);
        uint8_t *target_data = (uint8_t*)target.data;
        uint64_t bytes = TensorCommon::ByteCount(this->data_type, target.size);
        ret = memcpy(target_data, src_data, bytes);
    }

    return ret;
}

bool HostTensor::CopyRow(std::vector<float> &vec, int row_idx) const
{
    int y = row_idx % ne[1], z = row_idx / ne[1];
    bool ret = CopyRow(vec, y, z);
    return ret;
}

bool HostTensor::CopyRow(std::vector<float> &vec, int y, int z) const
{
    bool ret = true;
    uint64_t byte_num = this->bytes_per_row;
    bool is_fp32 = data_type == ElementType::F32;
    const uint8_t *row_data = (const uint8_t*)RowData(y, z);
    uint64_t block_num = 0;

    vec.resize(this->ne[0], 0);

    if (is_fp32)
    {
        if (this->ne[0] * sizeof(float) != byte_num) {
            LogError("Inconsistent numbers of bytes");
            return false;
        }

        memcpy(vec.data(), row_data, byte_num);
    }
    else
    {
        switch (this->data_type)
        {
        case ElementType::F16:
            for (int idx = 0; idx < this->ne[0]; idx++)
            {
                vec[idx] = (float)(*(const inferflow_fp16*)(row_data + idx * sizeof(inferflow_fp16)));
            }
            break;
        case ElementType::Q8_B32T1:
            block_num = this->bytes_per_row / sizeof(BlockQ8_B32T1);
            Quantization::DequantizeRow_Q8_B32T1(vec.data(),
                (const BlockQ8_B32T1*)row_data, block_num);
            break;
        case ElementType::Q8_B32T2:
            block_num = this->bytes_per_row / sizeof(BlockQ8_B32T2);
            Quantization::DequantizeRow_Q8_B32T2(vec.data(),
                (const BlockQ8_B32T2*)row_data, block_num);
            break;
        case ElementType::Q6_B64T1:
            block_num = this->bytes_per_row / sizeof(BlockQ6_B64T1);
            Quantization::DequantizeRow_Q6_B64T1(vec.data(),
                (const BlockQ6_B64T1*)row_data, block_num);
            break;
        case ElementType::Q5_B32T1:
            block_num = this->bytes_per_row / sizeof(BlockQ5_B32T1);
            Quantization::DequantizeRow_Q5(vec.data(),
                (const BlockQ5_B32T1*)row_data, block_num);
            break;
        case ElementType::Q5_B64T1:
            block_num = this->bytes_per_row / sizeof(BlockQ5_B64T1);
            Quantization::DequantizeRow_Q5_B64T1(vec.data(),
                (const BlockQ5_B64T1*)row_data, block_num);
            break;
        case ElementType::Q4_B16:
            block_num = this->bytes_per_row / sizeof(BlockQ4_B16);
            Quantization::DequantizeRow_Q4_B16(vec.data(),
                (const BlockQ4_B16*)row_data, block_num);
            break;
        case ElementType::Q4_B32T1A:
        case ElementType::Q4_B32T1B:
            block_num = this->bytes_per_row / sizeof(BlockQ4_B32T1);
            Quantization::DequantizeRow_Q4_B32T1(vec.data(),
                (const BlockQ4_B32T1*)row_data, block_num);
            break;
        case ElementType::Q4_B64T1:
            block_num = this->bytes_per_row / sizeof(BlockQ4_B64T1);
            Quantization::DequantizeRow_Q4_B64T1(vec.data(),
                (const BlockQ4_B64T1*)row_data, block_num);
            break;
        case ElementType::Q3H_B64T1:
            block_num = this->bytes_per_row / sizeof(BlockQ3H_B64T1);
            Quantization::DequantizeRow_Q3H_B64T1(vec.data(),
                (const BlockQ3H_B64T1*)row_data, block_num);
            break;
        case ElementType::Q3_B32T1A:
        case ElementType::Q3_B32T1B:
            block_num = this->bytes_per_row / sizeof(BlockQ3_B32T1);
            Quantization::DequantizeRow_Q3_B32T1(vec.data(),
                (const BlockQ3_B32T1*)row_data, block_num);
            break;
        case ElementType::Q2_B32T1A:
        case ElementType::Q2_B32T1B:
            block_num = this->bytes_per_row / sizeof(BlockQ2_B32T1);
            Quantization::DequantizeRow_Q2_B32T1(vec.data(),
                (const BlockQ2_B32T1*)row_data, block_num);
            break;
        default:
            LogError("Element type %d has not been handled yet.", this->data_type);
            ret = false;
            break;
        }
    }

    return ret;
}

std::ostream& HostTensor::Print(std::ostream &strm, int max_ne0,
    int max_ne1, int max_ne2, const char *title) const
{
    if (max_ne0 <= 0) {
        max_ne0 = INT32_MAX;
    }
    if (max_ne1 <= 0) {
        max_ne1 = INT32_MAX;
    }
    if (max_ne2 <= 0) {
        max_ne2 = INT32_MAX;
    }

    if (title != nullptr && strlen(title) != 0)
    {
        strm << title;
    }

    std::vector<float> row_data;
    for (int i2 = 0; i2 < this->ne[2]; i2++)
    {
        strm << (i2 > 0 ? ",\n " : (this->dim >= 3 ? "[" : ""));
        if (i2 >= max_ne2) {
            strm << "...";
            break;
        }

        for (int i1 = 0; i1 < this->ne[1]; i1++)
        {
            strm << (i1 > 0 ? (this->dim >= 3 ? ",\n  " : ",\n ") : (this->dim >= 2 ? "[" : ""));
            if (i1 >= max_ne1) {
                strm << "...";
                break;
            }

            CopyRow(row_data, i1, i2);

            strm << "[";
            for (int i0 = 0; i0 < this->ne[0]; i0++)
            {
                if (i0 >= max_ne0) {
                    strm << "...";
                    break;
		}

                if (i0 > 0) {
                    strm << ", ";
                }

                strm << row_data[i0];
            } //i0
            strm << "]";
        } //i1

        if (this->dim >= 2) {
            strm << "]";
        }
    } //i2

    if (this->dim >= 3) {
        strm << "]";
    }

    return strm;
}

std::ostream& operator << (std::ostream &strm, const HostTensor &tensor)
{
    tensor.Print(strm);
    return strm;
}

////////////////////////////////////////////////////////////////////////////////
// class HostSparseMatrix

HostSparseMatrix::HostSparseMatrix()
{
}

HostSparseMatrix::~HostSparseMatrix()
{
    Clear();
}

void HostSparseMatrix::Clear()
{
    if (this->data_ != nullptr)
    {
        if (is_auto_free_) {
            delete[] this->data_;
        }
        this->data_ = nullptr;
    }

    this->id_ = 0;
    this->rows_ = 0;
    this->cols_ = 0;
    this->size_ = 0;
}

bool HostSparseMatrix::New(int non_zero_cell_count, int cols, int rows)
{
    Clear();

    int bytes = non_zero_cell_count * sizeof(SparseMatrixCell);
    this->data_ = new SparseMatrixCell[bytes];
    this->size_ = non_zero_cell_count;

    cols_ = cols;
    rows_ = rows;
    return this->data_ != nullptr;
}

bool HostSparseMatrix::Set(const vector<SparseMatrixCell> &cells, int cols, int rows)
{
    int cell_count = (int)cells.size();
    bool ret = New(cell_count, cols, rows);
    Macro_RetFalseIf(!ret);

    int bytes = cell_count * sizeof(SparseMatrixCell);
    memcpy(this->data_, cells.data(), bytes);
    return ret;
}

void HostSparseMatrix::SetCell(int cell_idx, const SparseMatrixCell &cell)
{
    if (cell_idx >= 0 && cell_idx < size_) {
        data_[cell_idx] = cell;
    }
}

const SparseMatrixCell* HostSparseMatrix::GetCell(int cell_idx) const
{
    return cell_idx >= 0 && cell_idx < size_ ? &data_[cell_idx] : nullptr;
}

INFER_FLOW_END
