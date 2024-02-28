#pragma once

#include <vector>
#include <iostream>
#include "common/cuda_util.h"
#include "tensor_common.h"
#include "host_tensor.h"

INFER_FLOW_BEGIN

using std::vector;

class RawDeviceArray
{
public:
    ElementType data_type = ElementType::F32;
    int size = 0;
    int max_bytes = 0;
    void *data = nullptr;

protected:
    bool is_auto_free_ = true;

public:
    explicit RawDeviceArray(bool is_auto_free = true)
    {
        this->is_auto_free_ = is_auto_free;
    }

    explicit RawDeviceArray(float *src_data, int element_count, bool is_auto_free = true)
    {
        this->data = src_data;
        this->size = element_count;
        this->is_auto_free_ = is_auto_free;
    }

    virtual ~RawDeviceArray()
    {
        Clear();
    }

    void Clear();

    uint64_t ByteCount() const;
    float MemoryCost_MB() const;
    float MemoryCost_GB() const;

    const float* data_f32() const;
    float* data_f32();
    const half* data_f16() const;
    half* data_f16();

    bool IsAutoFree() const {
        return is_auto_free_;
    }
    void SetAutoFree(bool is_auto_free) {
        is_auto_free_ = is_auto_free;
    }

    void Set(float *src_data, int src_size, bool is_auto_free = true);
    void Set(half *src_data, int src_size, bool is_auto_free = true);
    void Set(ElementType etype, void *src_data, int src_size, bool is_auto_free = true);

    bool New(ElementType etype, int element_count);
    bool NewF32(int element_count);
    bool NewF16(int element_count);

    bool AssignZero();

    bool FromHost(const vector<float> &host_vec);
    bool FromHost(const float *host_data, int element_count);
    bool FromHost(const vector<half> &host_vec);
    bool FromHost(const half *host_data, int element_count);

    bool CopyFromDevice(const float *source_vec, int element_count);
    bool CopyFromDevice(const half *source_vec, int element_count);

protected:
    //disable the copy constructor and the assignment function
    RawDeviceArray(const RawDeviceArray &rhs) = delete;
    RawDeviceArray& operator = (const RawDeviceArray &rhs) = delete;
};

// A tensor on the GPU device
class DeviceTensor : public RawDeviceArray
{
public:
    //three dimensions: x, y, z
    static const int MaxDimCount = 3;

public:
    int id = 0;
    int dim = 0;
    int ne[MaxDimCount]; //number of elements (i.e., size of each dimension)
    uint64_t bytes_per_row = 0;
    bool is_zy_data = false; //data are organized in x-z-y order (but not x-y-z)

public:
    explicit DeviceTensor(bool is_auto_free = true);
    virtual ~DeviceTensor();
    void Clear();

    //get the number of rows
    int Rows() const {
        return ne[1] * ne[2];
    }

    int Columns() const {
        return ne[0];
    }

    bool IsQuantized() const {
        return data_type > ElementType::BF16;
    }

    //static int ElementSize(ElementType etype);

    bool HasCompatibleStructureWith(const DeviceTensor &rhs,
        bool be_transpose = false) const;

    bool New(ElementType etype, int n0, int n1 = 0, int n2 = 0);
    bool New(ElementType etype, int input_dim, const int *ne_per_dim);

    bool FromHost(const float *host_vec, int n0, int n1 = 0, int n2 = 0);
    bool FromHost(const half *host_vec, int n0, int n1 = 0, int n2 = 0);

    bool CopyFromHost(const void *host_data, int bytes);
    bool CopyRowFromHost(const void *host_data, int y, int z);

    bool CopyFromDevice(const void *source, int bytes);

    bool SetStructure(int n0, int n1 = 0, int n2 = 0);
    bool SetStructure(int dim_count, const int dims[MaxDimCount]);

    const void* RowData(int row) const;
    void* RowData(int row);
    const void* RowData(int y, int z) const;
    void* RowData(int y, int z);

    bool CopyToHost(std::vector<float> &host_vec) const;
    bool CopyToHost(std::vector<half> &host_vec) const;
    bool CopyToHost(HostTensor &host_tensor, int start_row = 0, int rows = -1) const;
    bool CopyRowToHost(std::vector<float> &host_vec, int y, int z) const;
    bool CopyRowToHost(std::vector<inferflow_fp16> &host_vec, int y, int z) const;

    std::ostream& Print(std::ostream &strm, int max_ne0 = 0, int max_ne1 = 0,
        int max_ne2 = 0, const char *title = nullptr) const;

protected:
    //disable the copy constructor and the assignment function
    DeviceTensor(const DeviceTensor &rhs) = delete;
    DeviceTensor& operator = (const DeviceTensor &rhs) = delete;
};

std::ostream& operator << (std::ostream &strm, const DeviceTensor &tensor);

class DeviceSparseMatrix
{
public:
    DeviceSparseMatrix();
    virtual ~DeviceSparseMatrix();
    void Clear();

    bool New(int non_zero_cell_count, int cols, int rows = 1);

    bool Set(const vector<SparseMatrixCell> &cells, const vector<int> &row_offset_array,
        int cols, int rows = 1);

    //cells should be sorted by row
    bool SetSortedCells(const vector<SparseMatrixCell> &cells, int cols, int rows = 1);

    //void SetCell(int cell_idx, const SparseMatrixCell &cell);

    bool IsEmpty() const {
        return size_ <= 0;
    }

    int Size() const {
        return size_;
    }

    int Rows() const {
        return rows_;
    }

    int Columns() const {
        return cols_;
    }

    const SparseMatrixCell* Cells() const {
        return data_;
    }

    const int* RowOffsetArray() const {
        return row_offset_array_;
    }

protected:
    int id_ = 0;
    int rows_ = 0; //number of rows
    int cols_ = 0; //number of columns
    int size_ = 0; //number of non-zero cells
    SparseMatrixCell *data_ = nullptr;
    int *row_offset_array_ = nullptr;
    bool is_auto_free_ = true;

protected:
    //disable the copy constructor and the assignment function
    DeviceSparseMatrix(const DeviceSparseMatrix &rhs) = delete;
    DeviceSparseMatrix& operator = (const DeviceSparseMatrix &rhs) = delete;
};

struct DeviceTensorEx
{
    DeviceTensor *tensor = nullptr;
    LinearQuantParams linear_quant_params;
    LogQuantParams log_quant_params;
    DeviceTensor *quant_map = nullptr;
    DeviceSparseMatrix *delta = nullptr;
};

INFER_FLOW_END
