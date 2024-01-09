#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include "hashtable.h"
#include "prime_types.h"
#include "string_dict.h"
#include "binary_stream.h"
#include "binary_file_stream.h"
#include "stream_helper.h"
#include "log.h"

namespace sslib
{

using namespace std;

template <class WeightType>
class StdStaticMatrix
{
public:
    typedef IdWeight<WeightType> Column;

    struct Row
    {
        uint32_t len;
        Column *columns;

        Row(uint32_t p_len = 0, IdWeight<WeightType> *p_cols = nullptr)
        {
            len = p_len; columns = p_cols;
        }
    };

public:
    StdStaticMatrix(uint32_t estimated_row_num = 0, bool use_location32 = true);
    virtual ~StdStaticMatrix();

    uint32_t GetRowCount() const {
        return (uint32_t)rows_.Size();
    }

    void Clear();

    const Row* AddRow(uint32_t row_id, const vector<Column> &data)
    {
        Row row((uint32_t)data.size());
        row.columns = element_heap_.New(row.len);
        for(uint32_t col_idx = 0; col_idx < row.len; col_idx++) {
            row.columns[col_idx] = data[col_idx];
        }

        auto iterRet = rows_.Insert(row_id, row);
        return !iterRet.IsEnd() ? &iterRet.Value() : nullptr;
    }

    const Row* GetRow(uint32_t row_id) const
    {
        auto iter = rows_.Find(row_id);
        return !iter.IsEnd() ? &iter.Value() : nullptr;
    }

    Row* GetRow(uint32_t row_id)
    {
        auto iter = rows_.Find(row_id);
        return !iter.IsEnd() ? &iter.Value() : nullptr;
    }

    const Row* SetRow(UInt32 row_id, const vector<Column> &data)
    {
        auto iter = rows_.Find(row_id);
        if (iter.IsEnd()) {
            return AddRow(row_id, data);
        }

        Row &row = iter.Value();
        if (row.len != (UInt32)data.size())
        {
            row.len = (UInt32)data.size();
            row.columns = element_heap_.New(row.len);
        }

        for (UInt32 idx = 0; idx < row.len; idx++) {
            row.columns[idx] = data[idx];
        }
        return &row;
    }

    bool Load(const string &file_path);
    bool Load(IBinStream &stream);
    bool Save(const string &file_path) const;
    bool Save(IBinStream &stream) const;

    bool SaveFmtCompactBin(const string &file_path, uint32_t max_row_len = UINT32_MAX) const;
    bool SaveFmtCompactBin(IBinStream &stream, uint32_t max_row_len = UINT32_MAX) const;
    bool LoadFmtCompactBin(const string &file_path, uint32_t max_row_len = UINT32_MAX);
    bool LoadFmtCompactBin(IBinStream &stream, uint32_t max_row_len = UINT32_MAX);

    bool Print(const string &file_path, const StrDict *row_dict = nullptr, const StrDict *col_dict = nullptr) const;
    bool Print(ostream &ps, const StrDict *row_dict = nullptr, const StrDict *col_dict = nullptr) const;
    bool Print(const string &file_path, const WStrDict *row_dict, const WStrDict *col_dict) const;
    bool Print(ostream &ps, const WStrDict *row_dict, const WStrDict *col_dict) const;

protected:
    HashMap<uint32_t, Row> rows_;
    BlockedAllocator<IdWeight<WeightType>> element_heap_;

    uint32_t estimated_row_num_;

protected:
    bool WriteStdHeader(IBinStream &stream, const string &format_str,
        uint32_t vertex_num, uint32_t stored_vertex_num, uint64_t edge_num) const;
    bool ReadStdHeader(IBinStream &stream, string &format_str,
        uint32_t &stored_vertex_num, uint64_t &edge_num);
    void CalculateStat(uint32_t &max_row_id, uint64_t &cell_num) const;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////////////////////

template <class WeightType>
StdStaticMatrix<WeightType>::StdStaticMatrix(uint32_t estimated_row_num, bool use_location32)
    : rows_(estimated_row_num, 1.5f, use_location32),
    element_heap_(max((uint32_t)10240, 100 * estimated_row_num))
{
    estimated_row_num_ = estimated_row_num;
}

template <class WeightType>
StdStaticMatrix<WeightType>::~StdStaticMatrix()
{
    Clear();
}

template <class WeightType>
void StdStaticMatrix<WeightType>::Clear()
{
    rows_.Clear();
    element_heap_.Clear();
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Load(const string &file_path)
{
    BinaryFileStream stream;
    bool ret = stream.OpenForRead(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    ret = Load(stream);
    stream.Close();
    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Load(IBinStream &stream)
{
    bool ret = true;
    Clear();

    uint32_t item_num = 0;
    ret = stream.Read(item_num);
    LogSevere("Not implemented yet");
    ret = false;
    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Save(const string &file_path) const
{
    BinaryFileStream stream;
    bool ret = stream.OpenForWrite(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    ret = Save(stream);
    ret = ret && stream.Flush();
    stream.Close();

    return ret && stream.IsGood();
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Save(IBinStream &stream) const
{
	uint32_t max_row_id = 0;
	uint64_t cell_count = 0;
    CalculateStat(max_row_id, cell_count);
    bool ret = stream.Write(cell_count);

    LogSevere("Not implemented yet");
    ret = false;
    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::SaveFmtCompactBin(
    const string &file_path, uint32_t max_row_len) const
{
    BinaryFileStream stream;
    bool ret = stream.OpenForWrite(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    return SaveFmtCompactBin(stream, max_row_len);
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::SaveFmtCompactBin(
    IBinStream &stream, uint32_t max_row_len) const
{
    bool ret = true;
	uint32_t row_count = 0;
	uint64_t cell_count = 0;
    for(auto iter = rows_.Begin(); !iter.IsEnd(); iter.Next())
    {
        if(row_count <= iter.Key()) {
            row_count = iter.Key() + 1;
        }
        cell_count += min(max_row_len, iter.Value().len);
    }

    std::string format_str = "CompactBinaryGraph";
    ret = WriteStdHeader(stream, format_str, row_count, row_count, cell_count);

    uint32_t row_len = 0;
    for(uint32_t row_idx = 0; ret && row_idx < row_count; row_idx++)
    {
        auto iter = rows_.Find(row_idx);
        row_len = !iter.IsEnd() ? min(max_row_len, iter.Value().len) : 0;
        ret = ret && stream.Write(row_len);
    }

    uint64_t written_cell_count = 0;
    for(uint32_t row_idx = 0; ret && row_idx < row_count; row_idx++)
    {
        auto iter = rows_.Find(row_idx);
        if(!iter.IsEnd())
        {
            const Row &cur_row = iter.Value();
            row_len = min(max_row_len, cur_row.len);
            for(uint32_t col_idx = 0; col_idx < row_len; col_idx++)
            {
                const auto &col = cur_row.columns[col_idx];
                stream.Write(col.id);
                stream.Write((const char*)&col.weight, sizeof(col.weight));
            }

            written_cell_count += row_len;
        }
    }

    if(cell_count != written_cell_count) {
        LogError("Inconsistent cell counts: %I64u vs. %I64u", cell_count, written_cell_count);
        return false;
    }

    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::LoadFmtCompactBin(
    const string &file_path, uint32_t max_row_len)
{
    BinaryFileStream stream;
    bool ret = stream.OpenForRead(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    return LoadFmtCompactBin(stream, max_row_len);
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::LoadFmtCompactBin(IBinStream &stream, uint32_t max_row_len)
{
    bool ret = true;
    Clear();

    std::string format_str;
	uint32_t row_num = 0;
	uint64_t cell_num = 0;
    ret = ReadStdHeader(stream, format_str, row_num, cell_num);
    if(!ret) {
        LogError("Error occurred in reading the header");
        return false;
    }
    if(String::CaseCmp(format_str.c_str(), "CompactBinaryGraph") != 0) {
        LogError("Invalid format string: %s", format_str.c_str());
        return false;
    }

    //
    vector<uint32_t> row_len_list(row_num, 0);
    uint32_t row_len = 0;
    for(uint32_t row_idx = 0; ret && row_idx < row_num; row_idx++)
    {
        ret = stream.Read(row_len);
        if(!ret) {
            LogError("Failed to read the length of row %u/%u", row_idx, row_num);
            return false;
        }

        row_len_list[row_idx] = row_len;
        Row cur_row(row_len);
        cur_row.columns = row_len > 0 ? element_heap_.New(min(row_len, max_row_len)) : nullptr;
        rows_.Insert(row_idx, cur_row);
    }

    Column col;
    for(uint32_t row_idx = 0; ret && row_idx < row_num; row_idx++)
    {
        auto row_iter = rows_.Find(row_idx);
        if(row_iter.IsEnd()) {
            LogError("Failed to read the length of row %u/%u", row_idx, row_num);
            return false;
        }

        auto &cur_row = row_iter.Value();
        row_len = row_len_list[row_idx];

        for(uint32_t col_idx = 0; ret && col_idx < row_len; col_idx++)
        {
            ret = ret && stream.Read(col.id);
            ret = ret && stream.Read((char*)&col.weight, sizeof(col.weight));
            if(!ret) {
                LogError("Failed to read column %u/%u of row %u/%u", col_idx, row_len, row_idx, row_num);
                return false;
            }

            if(col_idx < cur_row.len) {
                cur_row.columns[col_idx] = col;
            }
        }
    }

    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Print(const string &file_path, const StrDict *row_dict, const StrDict *col_dict) const
{
    ofstream stream(file_path.c_str(), ios::binary);
    if(!stream) {
        return false;
    }

    bool ret = Print(stream, row_dict, col_dict);
    stream.flush();
    if(!stream.good()) {
        ret = false;
    }

    stream.close();
    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Print(ostream &ps, const StrDict *row_dict, const StrDict *col_dict) const
{
    bool ret = true;
	uint32_t max_row_id = 0;
	uint64_t cell_count = 0;
    CalculateStat(max_row_id, cell_count);
    uint32_t row_num = GetRowCount();

    ps << "Rows: " << row_num << "; MaxRowId: " << max_row_id << "; Cells: " << cell_count << endl;

    uint32_t row_idx = 0;
    for(uint32_t row_id = 0; row_id <= max_row_id; row_id++)
    {
        auto row_iter = rows_.Find(row_id);
        if(row_iter.IsEnd()) {
            continue;
        }

        const auto &cur_row = row_iter.Value();
        row_idx++;
        const char *row_str = row_dict != nullptr ? row_dict->ItemStr(row_id) : nullptr;
        if(row_str == nullptr) {
            row_str = "";
        }

        ps << row_idx << " (" << row_id << ", " << row_str << "): ";
        for(uint32_t col_idx = 0; col_idx < cur_row.len; col_idx++)
        {
            const Column &col = cur_row.columns[col_idx];
            const char *col_str = col_dict != nullptr ? col_dict->ItemStr(col.id) : nullptr;
            if(col_str == nullptr) {
                col_str = "";
            }

            ps << (col_idx > 0 ? "; " : " ") << col_str << " (" << col.id << ", " << col.weight << ")";
        }
        ps << endl;
    }

    return ret && ps.good();
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Print(const string &file_path, const WStrDict *row_dict, const WStrDict *col_dict) const
{
    ofstream stream(file_path.c_str(), ios::binary);
    if (!stream) {
        return false;
    }

    bool ret = Print(stream, row_dict, col_dict);
    stream.flush();
    if (!stream.good()) {
        ret = false;
    }

    stream.close();
    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::Print(ostream &ps, const WStrDict *row_dict, const WStrDict *col_dict) const
{
    bool ret = true;
    uint32_t max_row_id = 0;
    uint64_t cell_count = 0;
    CalculateStat(max_row_id, cell_count);
    uint32_t row_num = GetRowCount();

    ps << "Rows: " << row_num << "; MaxRowId: " << max_row_id << "; Cells: " << cell_count << endl;

    uint32_t row_idx = 0;
    for (uint32_t row_id = 0; row_id <= max_row_id; row_id++)
    {
        auto row_iter = rows_.Find(row_id);
        if (row_iter.IsEnd()) {
            continue;
        }

        const auto &cur_row = row_iter.Value();
        row_idx++;
        const wchar_t *row_str = row_dict != nullptr ? row_dict->ItemStr(row_id) : nullptr;
        if (row_str == nullptr) {
            row_str = L"";
        }

        ps << row_idx << " (" << row_id << ", " << StringUtil::ToUtf8(row_str) << "): ";
        for (uint32_t col_idx = 0; col_idx < cur_row.len; col_idx++)
        {
            const Column &col = cur_row.columns[col_idx];
            const wchar_t *col_str = col_dict != nullptr ? col_dict->ItemStr(col.id) : nullptr;
            if (col_str == nullptr) {
                col_str = L"";
            }

            ps << (col_idx > 0 ? "; " : " ") << StringUtil::ToUtf8(col_str) << " (" << col.id << ", " << col.weight << ")";
        }
        ps << endl;
    }

    return ret && ps.good();
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::WriteStdHeader(IBinStream &stream, const string &format_str,
    uint32_t vertex_num, uint32_t stored_vertex_num, uint64_t edge_num) const
{
    bool ret = BinStreamHelper::WriteString16(stream, format_str);
    uint32_t version = 300;
    ret = ret && stream.Write(version);

    ret = ret && stream.Write(vertex_num);
    ret = ret && stream.Write(stored_vertex_num);
    ret = ret && stream.Write(edge_num);
    uint16_t edge_weight_value_size = sizeof(WeightType);
    ret = ret && stream.Write(edge_weight_value_size);
    if(!ret) {
        return false;
    }
    return ret;
}

template <class WeightType>
bool StdStaticMatrix<WeightType>::ReadStdHeader(IBinStream &stream,
    std::string &format_str, uint32_t &stored_vertex_num, uint64_t &edge_num)
{
    bool ret = true;
    BinStreamHelper::ReadString16(stream, format_str);
    uint32_t version = 0;
    ret = ret && stream.Read(version);

    ret = ret && stream.Read(stored_vertex_num);
    if(version > 100) {
        ret = ret && stream.Read(stored_vertex_num);
    }

	if (version >= 300)
	{
		ret = ret && stream.Read(edge_num);
	}
	else
	{
		uint32_t edge_num_32 = 0;
		ret = ret && stream.Read(edge_num_32);
		edge_num = edge_num_32;
	}

    uint16_t edge_weight_value_size = 0;
    ret = ret && stream.Read(edge_weight_value_size);
    if(!ret) {
        return false;
    }
    if(edge_weight_value_size != sizeof(WeightType)) {
        LogError("Inconsistent edge weight size: %d vs. %d", edge_weight_value_size, sizeof(WeightType));
        return false;
    }

    return ret;
}

template <class WeightType>
void StdStaticMatrix<WeightType>::CalculateStat(uint32_t &max_row_id, uint64_t &cell_num) const
{
    max_row_id = 0;
    cell_num = 0;
    for(auto iter = rows_.Begin(); !iter.IsEnd(); iter.Next())
    {
        if(max_row_id < iter.Key()) {
            max_row_id = iter.Key();
        }
        cell_num += iter.Value().len;
    }
}

} //end of namespace
