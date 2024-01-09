#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include "hashtable.h"
#include "prime_types.h"
#include "item_dict.h"
#include "string_dict.h"
#include "binary_stream.h"
#include "binary_file_stream.h"
#include "stream_helper.h"
#include "log.h"

namespace sslib
{

#pragma pack(push, 1)
template <class WEIGHT_TYPE>
struct MatrixElement
{
    uint32_t row, col;
    WEIGHT_TYPE weight;

    MatrixElement(uint32_t r = 0, uint32_t c = 0, WEIGHT_TYPE w = 0) {
        row = r; col = c; weight = w;
    }

    static bool LessPtrRowWeightCol(const MatrixElement<WEIGHT_TYPE> *lhs,
        const MatrixElement<WEIGHT_TYPE> *rhs)
    {
        if(lhs->row != rhs->row) {
            return lhs->row < rhs->row;
        }
        if(lhs->weight != rhs->weight) {
            return rhs->weight < lhs->weight;
        }
        return lhs->col < rhs->col;
    }

    static bool LessPtrRowColWeight(const MatrixElement<WEIGHT_TYPE> *lhs,
        const MatrixElement<WEIGHT_TYPE> *rhs)
    {
        if (lhs->row != rhs->row) {
            return lhs->row < rhs->row;
        }
        if (lhs->col != rhs->col) {
            return lhs->col < rhs->col;
        }
        return lhs->weight > rhs->weight;
    }

    static bool LessPtrColWeightRow(const MatrixElement<WEIGHT_TYPE> *lhs,
        const MatrixElement<WEIGHT_TYPE> *rhs)
    {
        if(lhs->col != rhs->col) {
            return lhs->col < rhs->col;
        }
        if(lhs->weight != rhs->weight) {
            return rhs->weight < lhs->weight;
        }
        return lhs->row < rhs->row;
    }

    static bool LessPtrColRowWeight(const MatrixElement<WEIGHT_TYPE> *lhs,
        const MatrixElement<WEIGHT_TYPE> *rhs)
    {
        if (lhs->col != rhs->col) {
            return lhs->col < rhs->col;
        }
        if (lhs->row != rhs->row) {
            return lhs->row < rhs->row;
        }
        return lhs->weight > rhs->weight;
    }

    static bool LessRowColWeight(const MatrixElement<WEIGHT_TYPE> &lhs,
        const MatrixElement<WEIGHT_TYPE> &rhs)
    {
        if (lhs.row != rhs.row) {
            return lhs.row < rhs.row;
        }
        if (lhs.col != rhs.col) {
            return lhs.col < rhs.col;
        }
        return lhs.weight > rhs.weight;
    }

    static bool LessColRowWeight(const MatrixElement<WEIGHT_TYPE> &lhs,
        const MatrixElement<WEIGHT_TYPE> &rhs)
    {
        if (lhs.col != rhs.col) {
            return lhs.col < rhs.col;
        }
        if (lhs.row != rhs.row) {
            return lhs.row < rhs.row;
        }
        return lhs.weight > rhs.weight;
    }
};
#pragma pack(pop)

template <class WEIGHT_TYPE>
class MatrixElementHashTraits
{
public:
    uint32_t operator()(const MatrixElement<WEIGHT_TYPE> &key) const {
       return key.row * Large_Prime + key.col;
    }

    uint32_t operator()(const MatrixElement<WEIGHT_TYPE> *key) const {
       return key->row * Large_Prime + key->col;
    }

    int operator()(const MatrixElement<WEIGHT_TYPE> &lhs, const MatrixElement<WEIGHT_TYPE> &rhs) const
    {
        if(lhs.row != rhs.row) {
            return lhs.row < rhs.row ? -1 : 1;
        }
        else {
            return lhs.col < rhs.col ? -1 : (lhs.col > rhs.col ? 1 : 0);
        }
    }

    int operator()(const MatrixElement<WEIGHT_TYPE> *lhs, const MatrixElement<WEIGHT_TYPE> *rhs) const
    {
        if(lhs->row != rhs->row) {
            return lhs->row < rhs->row ? -1 : 1;
        }
        else {
            return lhs->col < rhs->col ? -1 : (lhs->col > rhs->col ? 1 : 0);
        }
    }
};

template <class WEIGHT_TYPE>
class NaiveMatrix
{
public:
    typedef MatrixElement<WEIGHT_TYPE> Cell;
    typedef Hashtable<Cell, MatrixElementHashTraits<WEIGHT_TYPE>> CellTable;

public:
    NaiveMatrix(uint32_t estimated_non_zero_element_num = 0, bool use_location32 = false);
    virtual ~NaiveMatrix();

    uint32_t Size() const {
        return (uint32_t)cells_.Size();
    }

    void Clear()
    {
        cells_.Clear();
        cells_.Reinit(estimated_non_zero_element_num_);
    }

    bool AddCell(uint32_t row, uint32_t col, WEIGHT_TYPE weight)
    {
        Cell cell(row, col, weight);
        auto iter = cells_.Insert(cell);
        return !iter.IsEnd();
    }

    const Cell* GetCell(uint32_t row, uint32_t col) const
    {
        Cell cell_key(row, col);
        auto iter = cells_.Find(cell_key);
        return !iter.IsEnd() ? &(*iter) : nullptr;
    }

    WEIGHT_TYPE GetCellValue(uint32_t row, uint32_t col) const
    {
        Cell cell_key(row, col);
        auto iter = cells_.Find(cell_key);
        return !iter.IsEnd() ? (*iter).weight : 0;
    }
    WEIGHT_TYPE GetCellValue(const Cell &cell) const
    {
        auto iter = cells_.Find(cell);
        return !iter.IsEnd() ? (*iter).weight : 0;
    }

    void AddWeight(const Cell &cell)
    {
        auto iter = cells_.Find(cell);
        if(!iter.IsEnd()) {
            (*iter).weight += cell.weight;
        }
        else {
            cells_.Insert(cell);
        }
    }
    void AddWeight(uint32_t row, uint32_t col, const WEIGHT_TYPE &weight)
    {
        Cell cell(row, col, weight);
        auto iter = cells_.Find(cell);
        if(!iter.IsEnd()) {
            (*iter).weight += cell.weight;
        }
        else {
            cells_.Insert(cell);
        }
    }

    void AddCellMax(const Cell &cell)
    {
        auto iter = cells_.Find(cell);
        if(!iter.IsEnd()) {
            (*iter).weight = max((*iter).weight, cell.weight);
        }
        else {
            cells_.Insert(cell);
        }
    }
    void AddCellMax(uint32_t row, uint32_t col, const WEIGHT_TYPE &weight)
    {
        Cell cell(row, col, weight);
        auto iter = cells_.Find(cell);
        if(!iter.IsEnd()) {
            (*iter).weight = max((*iter).weight, cell.weight);
        }
        else {
            cells_.Insert(cell);
        }
    }

    void SetWeight(uint32_t row, uint32_t col, const WEIGHT_TYPE &weight)
    {
        Cell cell(row, col, weight);
        auto iter = cells_.Find(cell);
        if(!iter.IsEnd()) {
            (*iter).weight = cell.weight;
        }
        else {
            cells_.Insert(cell);
        }
    }

    const CellTable& GetCellTable() const {
        return cells_;
    }

    bool Load(const std::string &path);
    bool Load(std::istream &strm);
    bool Save(const std::string &path) const;
    bool Save(std::ostream &strm) const;

    bool SaveFmtCompactBin(const std::string &path, uint32_t max_nbr_num = UINT32_MAX,
        bool is_reverse = false, bool is_order_by_weight = true) const;
    bool SaveFmtCompactBin(IBinStream &strm, uint32_t max_nbr_num = UINT32_MAX,
        bool is_reverse = false, bool is_order_by_weight = true) const;
    bool LoadFmtCompactBin(const std::string &path);
    bool LoadFmtCompactBin(IBinStream &strm);

    bool Print(const std::string &path, const ItemDict *dict = nullptr) const;
    bool Print(std::ostream &strm, const ItemDict *dict = nullptr) const;

    template <class T1, class T2>
    bool Print(const std::string &path, const StringDict<T1> *row_dict,
        const StringDict<T2> *col_dict) const
    {
        ofstream strm(path.c_str(), ios::binary);
        if(!strm) {
            return false;
        }

        bool ret = Print(strm, row_dict, col_dict);
        strm.flush();
        if(!strm.good()) {
            ret = false;
        }

        strm.close();
        return ret;
    }

    template <class T1, class T2>
    bool Print(std::ostream &strm, const StringDict<T1> *row_dict,
        const StringDict<T2> *col_dict) const
    {
        bool ret = true;
        uint32_t item_num = Size();
        strm << "; items: " << item_num << "\r\n";

        auto iter = cells_.Begin();
        for(; ret && strm.good() && !iter.IsEnd(); iter.Next())
        {
            const Cell &cell = *iter;
            strm << cell.row << "\t" << cell.col << "\t" << cell.weight;
            if(row_dict != nullptr)
            {
                const char *row_str = row_dict->ItemStr(cell.row);
                strm << "\t" << (row_str != nullptr ? row_str : "<null>");
            }
            if(col_dict != nullptr)
            {
                const char *col_str = col_dict->ItemStr(cell.col);
                strm << "\t" << (col_str != nullptr ? col_str : "<null>");
            }
            strm << "\r\n";
        }

        return ret && strm.good();
    }

    template <class T1, class T2>
    bool Print(const std::string &path, const WStringDict<T1> *row_dict,
        const WStringDict<T2> *col_dict) const
    {
        ofstream strm(path.c_str(), ios::binary);
        if (!strm) {
            return false;
        }

        bool ret = Print(strm, row_dict, col_dict);
        strm.flush();
        if (!strm.good()) {
            ret = false;
        }

        strm.close();
        return ret;
    }

    template <class T1, class T2>
    bool Print(std::ostream &strm, const WStringDict<T1> *row_dict,
        const WStringDict<T2> *col_dict) const
    {
        bool ret = true;
        uint32_t item_num = Size();
        strm << "; items: " << item_num << "\r\n";

        auto iter = cells_.Begin();
        for (; ret && strm.good() && !iter.IsEnd(); iter.Next())
        {
            const Cell &cell = *iter;
            strm << cell.row << "\t" << cell.col << "\t" << cell.weight;
            if (row_dict != nullptr)
            {
                const wchar_t *row_str = row_dict->ItemStr(cell.row);
                strm << "\t" << (row_str != nullptr ? StringUtil::ToUtf8(row_str) : "<null>");
            }
            if (col_dict != nullptr)
            {
                const wchar_t *col_str = col_dict->ItemStr(cell.col);
                strm << "\t" << (col_str != nullptr ? StringUtil::ToUtf8(col_str) : "<null>");
            }
            strm << "\r\n";
        }

        return ret && strm.good();
    }

protected:
    CellTable cells_;
    uint32_t estimated_non_zero_element_num_;

protected:
    bool WriteStdHeader(IBinaryStream &strm, const std::string &format_str,
        uint32_t vertex_num, uint32_t stored_vertex_num, uint64_t edge_num) const;
    bool ReadStdHeader(IBinaryStream &strm, std::string &format_str,
        uint32_t &stored_vertex_num, uint64_t &edge_num);
};

typedef MatrixElement<float> FloatNaiveMatrixCell;
typedef NaiveMatrix<float> FloatNaiveMatrix;
typedef MatrixElement<uint32_t> UInt32NaiveMatrixCell;
typedef NaiveMatrix<uint32_t> UInt32NaiveMatrix;

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////////////////////

template <class WEIGHT_TYPE>
NaiveMatrix<WEIGHT_TYPE>::NaiveMatrix(
    uint32_t estimated_non_zero_element_num, bool use_location32)
    : cells_(estimated_non_zero_element_num, 1.5f, use_location32)
{
    estimated_non_zero_element_num_ = estimated_non_zero_element_num;
}

template <class WEIGHT_TYPE>
NaiveMatrix<WEIGHT_TYPE>::~NaiveMatrix()
{
    Clear();
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::Load(const std::string &path)
{
    ifstream strm(path.c_str(), ios::binary);
    if(!strm) {
        return false;
    }

    bool ret = Load(strm);
    strm.close();
    return ret;
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::Load(std::istream &strm)
{
    bool ret = true;
    Clear();

    uint32_t item_num = 0;
    strm.read((char*)&item_num, sizeof(item_num));
    Cell cell;
    for(uint32_t item_idx = 0; ret && strm.good() && item_idx < item_num; item_idx++)
    {
        strm.read((char*)&cell.row, sizeof(cell.row));
        strm.read((char*)&cell.col, sizeof(cell.col));
        strm.read((char*)&cell.weight, sizeof(cell.weight));
        ret = AddCell(cell.row, cell.col, cell.weight);
    }

    return ret && strm.good();
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::Save(const std::string &path) const
{
    ofstream strm(path.c_str(), ios::binary);
    if(!strm) {
        return false;
    }

    bool ret = Save(strm);
    strm.flush();
    if(!strm.good()) {
        ret = false;
    }

    strm.close();
    return ret;
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::Save(std::ostream &strm) const
{
    bool ret = true;
    uint32_t item_num = Size();
    strm.write((const char*)&item_num, sizeof(item_num));

    auto iter = cells_.Begin();
    for(; ret && strm.good() && !iter.IsEnd(); iter.Next())
    {
        const Cell &cell = *iter;
        strm.write((const char*)&cell.row, sizeof(cell.row));
        strm.write((const char*)&cell.col, sizeof(cell.col));
        strm.write((const char*)&cell.weight, sizeof(cell.weight));
    }

    return ret && strm.good();
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::SaveFmtCompactBin(const std::string &path,
    uint32_t max_nbr_num, bool is_reverse, bool is_order_by_weight) const
{
    BinFileStream strm;
    uint32_t mode = BinFileStream::MODE_WRITE | BinFileStream::MODE_CREATE
        | BinFileStream::MODE_TRUNC;
    bool ret = strm.Open(path.c_str(), mode);
    if(!ret) {
        LogError("Fail to open file %s", path.c_str());
        return false;
    }

    return SaveFmtCompactBin(strm, max_nbr_num, is_reverse, is_order_by_weight);
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::SaveFmtCompactBin(IBinStream &strm,
    uint32_t max_nbr_num, bool is_reverse, bool is_order_by_weight) const
{
    bool ret = true;
    uint64_t edge_num = cells_.Size();
    std::vector<const Cell*> cells;
    cells.reserve((size_t)edge_num);
    auto iter = cells_.Begin();
    uint32_t row_num = 0, col_num = 0;
    for(; !iter.IsEnd(); iter.Next())
    {
        const Cell *cell_ptr = &(*iter);
        if(row_num <= cell_ptr->row) {
            row_num = cell_ptr->row + 1;
        }
        if(col_num <= cell_ptr->col) {
            col_num = cell_ptr->col + 1;
        }
        cells.push_back(cell_ptr);
    }

    uint32_t stored_vertex_num = 0, vertex_num = max(row_num, col_num);
    if(is_reverse)
    {
        stored_vertex_num = col_num;
        if (is_order_by_weight) {
            std::sort(cells.begin(), cells.end(), MatrixElement<WEIGHT_TYPE>::LessPtrColWeightRow);
        }
        else {
            std::sort(cells.begin(), cells.end(), MatrixElement<WEIGHT_TYPE>::LessPtrColRowWeight);
        }
    }
    else
    {
        stored_vertex_num = row_num;
        if (is_order_by_weight) {
            std::sort(cells.begin(), cells.end(), MatrixElement<WEIGHT_TYPE>::LessPtrRowWeightCol);
        }
        else {
            std::sort(cells.begin(), cells.end(), MatrixElement<WEIGHT_TYPE>::LessPtrRowColWeight);
        }
    }

    if(edge_num != (uint64_t)cells.size()) {
        LogError("Corrupted cell-table");
        return false;
    }

    std::vector<uint32_t> vertex_nbr_count_list;
    vertex_nbr_count_list.reserve(stored_vertex_num);

	uint32_t vertex_id = 0, nbr_count = 0;
	uint64_t total_nbr_count = 0, edge_num_delta = 0;
    for(uint64_t edge_idx = 0; edge_idx < edge_num; edge_idx++)
    {
        const Cell *cell_ptr = cells[(size_t)edge_idx];
        uint32_t cur_vertex_id = is_reverse ? cell_ptr->col : cell_ptr->row;
        if(cur_vertex_id > vertex_id)
        {
            if(nbr_count > max_nbr_num) {
                edge_num_delta += (nbr_count - max_nbr_num);
                nbr_count = max_nbr_num;
            }
            for(uint32_t vtx_idx = vertex_id; vtx_idx < cur_vertex_id; vtx_idx++)
            {
                vertex_nbr_count_list.push_back(nbr_count);
                total_nbr_count += nbr_count;
                nbr_count = 0;
            }

            vertex_id = cur_vertex_id;
            nbr_count = 1;
        }
        else //cur_vertex_id == vertex_id
        {
            nbr_count++;
        }
    }

    if(nbr_count > max_nbr_num)
    {
        edge_num_delta += (nbr_count - max_nbr_num);
        nbr_count = max_nbr_num;
    }

    for(uint32_t vtx_idx = vertex_id; ret && vtx_idx < stored_vertex_num; vtx_idx++)
    {
        vertex_nbr_count_list.push_back(nbr_count);
        total_nbr_count += nbr_count; 
        nbr_count = 0;
    }

    if(total_nbr_count + edge_num_delta != edge_num)
    {
        LogError("Inconsistent edge count: %I64u + %I64u vs. %I64u",
            total_nbr_count, edge_num_delta, edge_num);
        return false;
    }

    std::string format_str = "CompactBinaryGraph";
    ret = WriteStdHeader(strm, format_str, vertex_num, stored_vertex_num, total_nbr_count);
    for(size_t vertex_idx = 0; vertex_idx < vertex_nbr_count_list.size(); vertex_idx++)
    {
        nbr_count = vertex_nbr_count_list[vertex_idx];
        ret = strm.Write(nbr_count);
    }

    vertex_id = 0;
    nbr_count = 0;
    for(uint64_t edge_idx = 0; ret && edge_idx < edge_num; edge_idx++)
    {
        const Cell *cell_ptr = cells[(size_t)edge_idx];
        uint32_t cur_vertex_id = is_reverse ? cell_ptr->col : cell_ptr->row;
        uint32_t iNeighborId = is_reverse ? cell_ptr->row : cell_ptr->col;
        if(cur_vertex_id > vertex_id)
        {
            vertex_id = cur_vertex_id;
            nbr_count = 1;
        }
        else
        {
            nbr_count++;
        }

        if(nbr_count <= max_nbr_num)
        {
            ret = ret && strm.Write((const char*)&iNeighborId, sizeof(iNeighborId));
            ret = ret && strm.Write((const char*)&cell_ptr->weight, sizeof(cell_ptr->weight));
        }
    }

    return ret;
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::LoadFmtCompactBin(const std::string &path)
{
    Clear();

    BinFileStream strm;
    uint32_t mode = BinFileStream::MODE_READ;
    bool ret = strm.Open(path.c_str(), mode);
    if(!ret) {
        LogError("Fail to open file %s", path.c_str());
        return false;
    }

    return LoadFmtCompactBin(strm);
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::LoadFmtCompactBin(IBinaryStream &strm)
{
    bool ret = true;
    std::string format_str;
	uint32_t vertex_num = 0;
	uint64_t edge_num = 0;
    ret = ReadStdHeader(strm, format_str, vertex_num, edge_num);
    if(!ret) {
        LogError("Error occurred in reading the header");
        return false;
    }
    if(String::CaseCmp(format_str.c_str(), "CompactBinaryGraph") != 0) {
        LogError("Invalid format string: %s", format_str.c_str());
        return false;
    }

    //
    std::vector<uint32_t> nbr_num_list(vertex_num, 0);
    uint32_t vertex_nbr_num = 0;
    for (uint32_t vertex_idx = 0; ret && vertex_idx < vertex_num; vertex_idx++)
    {
        ret = strm.Read((char*)&vertex_nbr_num, sizeof(vertex_nbr_num));
        if (!ret) {
            LogError("Failed to read the neighbor count of vertex %u/%u", vertex_idx, vertex_num);
            return false;
        }
        nbr_num_list[vertex_idx] = vertex_nbr_num;
    }

    Cell curCell;
    for(uint32_t vertex_idx = 0; ret && vertex_idx < vertex_num; vertex_idx++)
    {
        uint32_t nbr_num = nbr_num_list[vertex_idx];
        curCell.row = vertex_idx;
        for(uint32_t nbr_idx = 0; ret && nbr_idx < nbr_num; nbr_idx++)
        {
            ret = ret && strm.Read((char*)&curCell.col, sizeof(curCell.col));
            ret = ret && strm.Read((char*)&curCell.weight, sizeof(curCell.weight));
            if(!ret) {
                LogError("Error occurred in reading the neighbor %u/%u of vertex %u/%u",
                    nbr_idx, nbr_num, vertex_idx, vertex_num);
                return false;
            }
            cells_.Insert(curCell);
        }
    }

    return ret;
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::Print(const std::string &path, const ItemDict *dict) const
{
    ofstream strm(path.c_str(), ios::binary);
    if(!strm) {
        return false;
    }

    bool ret = Print(strm, dict);
    strm.flush();
    if(!strm.good()) {
        ret = false;
    }

    strm.close();
    return ret;
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::Print(std::ostream &strm, const ItemDict *dict) const
{
    bool ret = true;
    uint32_t item_num = Size();
    strm << "; items: " << item_num << "\r\n";

    auto iter = cells_.Begin();
    for(; ret && strm.good() && !iter.IsEnd(); iter.Next())
    {
        const Cell &cell = *iter;
        strm << cell.row << "\t" << cell.col << "\t" << cell.weight;
        if(dict != nullptr)
        {
            const auto *item_ptr = dict->Find(cell.row);
            strm << "\t" << (item_ptr != nullptr ? item_ptr->str : "<null>");
            item_ptr = dict->Find(cell.col);
            strm << "\t" << (item_ptr != nullptr ? item_ptr->str : "<null>");
        }
        strm << "\r\n";
    }

    return ret && strm.good();
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::WriteStdHeader(IBinaryStream &strm, const std::string &format_str,
    uint32_t vertex_num, uint32_t stored_vertex_num, uint64_t edge_num) const
{
    bool ret = BinStreamHelper::WriteString16(strm, format_str);
    uint32_t version = 300; //edge_num is 64 bit when version >= 300
    ret = ret && strm.Write((char*)&version, sizeof(version));

    ret = ret && strm.Write(vertex_num);
    ret = ret && strm.Write(stored_vertex_num);
    ret = ret && strm.Write(edge_num);
    uint16_t edge_weight_value_size = sizeof(WEIGHT_TYPE);
    ret = ret && strm.Write((const char*)&edge_weight_value_size, sizeof(edge_weight_value_size));
    if(!ret) {
        return false;
    }
    return ret;
}

template <class WEIGHT_TYPE>
bool NaiveMatrix<WEIGHT_TYPE>::ReadStdHeader(IBinaryStream &strm,
    std::string &format_str, uint32_t &stored_vertex_num, uint64_t &edge_num)
{
    bool ret = true;
    BinStreamHelper::ReadString16(strm, format_str);
    uint32_t version = 0;
    ret = ret && strm.Read((char*)&version, sizeof(version));

    ret = ret && strm.Read((char*)&stored_vertex_num, sizeof(stored_vertex_num));
    if(version > 100) {
        ret = ret && strm.Read((char*)&stored_vertex_num, sizeof(stored_vertex_num));
    }

	uint32_t edge_num_32 = 0;
	if (version >= 300)
	{
		ret = ret && strm.Read(edge_num);
	}
	else
	{
		ret = ret && strm.Read(edge_num_32);
		edge_num = edge_num_32;
	}

    uint16_t edge_weight_value_size = 0;
    ret = ret && strm.Read((char*)&edge_weight_value_size, sizeof(edge_weight_value_size));
    if(!ret) {
        return false;
    }
    if(edge_weight_value_size != sizeof(WEIGHT_TYPE)) {
        LogError("Inconsistent edge weight size: %d vs. %d",
            edge_weight_value_size, sizeof(float));
        return false;
    }

    return ret;
}

} //end of namespace
