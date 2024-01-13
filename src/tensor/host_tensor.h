#pragma once

#include <iostream>
#include <vector>
#include "tensor_common.h"
#include "namespace.inc"

INFER_FLOW_BEGIN

using std::vector;

class HostTensor
{
public:
    static const int MaxDimCount = 3;

public:
    int id = 0;
    uint64_t size = 0;
    void *data = nullptr;
    ElementType data_type = ElementType::F32;

    int dim = 0;
    int ne[MaxDimCount]; //number of elements in each dimension

    uint64_t bytes_per_row = 0;
    bool is_zy_data = false; //data are organized in x-z-y order (but not x-y-z)

protected:
    bool is_auto_free_ = true;

public:
    explicit HostTensor(bool is_auto_free = true)
    {
        this->is_auto_free_ = is_auto_free;
    }

    explicit HostTensor(float *src_data, int element_count, bool is_auto_free = true)
    {
        this->data = src_data;
        this->is_auto_free_ = is_auto_free;
        this->SetStructure(element_count, 0, 0);
    }

    virtual ~HostTensor()
    {
        Clear();
    }

    void Clear();

    bool SetStructure(int n0, int n1 = 0, int n2 = 0);
    bool SetStructure(int dim_count, const int dims[MaxDimCount]);

    const float* data_f32() const;
    float* data_f32();
    const inferflow_fp16* data_f16() const;
    inferflow_fp16* data_f16();

    const void* RowData(int row) const;
    void* RowData(int row);
    const void* RowData(int y, int z) const;
    void* RowData(int y, int z);

    bool New(ElementType etype, int n0, int n1 = 0, int n2 = 0);
    bool New(ElementType etype, int input_dim, const int *ne_per_dim);
    bool NewF32(int n0, int n1 = 0, int n2 = 0);
    bool NewF16(int n0, int n1 = 0, int n2 = 0);

    void Set(float *src_data, int cx, int cy = 0, int cz = 0,
        bool is_auto_free = true);
    void Set(inferflow_fp16 *src_data, int cx, int cy = 0, int cz = 0,
        bool is_auto_free = true);
    void Set(ElementType etype, void *src_data, int cx, int cy = 0, int cz = 0,
        bool is_auto_free = true);

    bool AssignZero();

    //get the number of rows
    int Rows() const {
        return ne[1] * ne[2];
    }

    int Columns() const {
        return ne[0];
    }

    bool CopyTo(HostTensor &target, int start_row = 0, int rows = -1) const;

    bool CopyRow(std::vector<float> &vec, int row_idx) const;
    bool CopyRow(std::vector<float> &vec, int y, int z) const;

    bool HasCompatibleStructureWith(const HostTensor &rhs, bool be_transpose = false) const;

    std::ostream& Print(std::ostream &strm, int max_ne0 = 0, int max_ne1 = 0,
        int max_ne2 = 0, const char *title = nullptr) const;

protected:
    //disable the copy constructor and the assignment function
    HostTensor(const HostTensor &rhs) = delete;
    HostTensor& operator = (const HostTensor &rhs) = delete;
};

//std::ostream& operator << (std::ostream &strm, const HostTensor &tensor);

class HostSparseMatrix
{
public:
    HostSparseMatrix();
    virtual ~HostSparseMatrix();
    void Clear();

    bool New(int non_zero_cell_count, int cols, int rows = 1);

    bool Set(const vector<SparseMatrixCell> &cells, int cols, int rows = 1);

    void SetCell(int cell_idx, const SparseMatrixCell &cell);
    const SparseMatrixCell* GetCell(int cell_idx) const;

    uint64_t Size() const {
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

protected:
    int id_ = 0;
    int rows_ = 0; //number of rows
    int cols_ = 0; //number of columns
    int size_ = 0; //number of non-zero cells
    SparseMatrixCell *data_ = nullptr;
    bool is_auto_free_ = true;

protected:
    //disable the copy constructor and the assignment function
    HostSparseMatrix(const HostSparseMatrix &rhs) = delete;
    HostSparseMatrix& operator = (const HostSparseMatrix &rhs) = delete;
};

INFER_FLOW_END
