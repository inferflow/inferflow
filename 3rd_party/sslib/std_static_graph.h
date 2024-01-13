#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "string.h"
#include "prime_types.h"
#include "binary_file_stream.h"
#include "stream_helper.h"
#include "log.h"

namespace sslib
{

struct GraphConst
{
    enum EnumSortNeighborsBy
    {
        SortByIdx = 0,
        SortByWeight
    };
};

template <class WeightType>
struct VertexNeighborArray
{
    IdWeight<WeightType> *items = nullptr;
    uint32_t size = 0;
};

template <class EdgeWeightType>
class StdStaticGraph
{
public:
    typedef IdWeight<EdgeWeightType> Neighbor;
    typedef VertexNeighborArray<EdgeWeightType> NeighborArray;

    enum EnumSortNeighborsBy {
        SORT_BY_IDX = GraphConst::SortByIdx,
        SORT_BY_WEIGHT = GraphConst::SortByWeight
    };

public:
    StdStaticGraph()
    {
        vertex_count_ = stored_vertex_count_ = 0;
        vertex_offsets_ = nullptr;
        edge_num_ = 0;
        neighbors_ = nullptr;
    }

    virtual ~StdStaticGraph()
    {
        Clear();
    }

    virtual void Clear();

    virtual uint32_t GetVertexNum() const {
        return vertex_count_;
    }
    virtual uint32_t GetSNodeNum() const {
        return stored_vertex_count_;
    }
    virtual uint64_t GetEdgeNum() const {
        return edge_num_;
    }

    virtual Neighbor* GetNeighbor(uint32_t node_id, uint32_t nbr_id, bool is_binary_search = true);
    virtual const Neighbor* GetNeighbor(uint32_t node_id, uint32_t nbr_id, bool is_binary_search = true) const;

    EdgeWeightType GetEdgeWeight(uint32_t node_id, uint32_t nbr_id, bool is_binary_search = true) const
    {
        const Neighbor *nbr_ptr = GetNeighbor(node_id, nbr_id, is_binary_search);
        return nbr_ptr != nullptr ? nbr_ptr->weight : 0;
    }

    virtual bool SetEdgeWeight(uint32_t node_id, uint32_t nbr_id, EdgeWeightType val);
    virtual bool GetNeighbors(uint32_t node_id, NeighborArray &nbr_array) const;
    virtual NeighborArray GetNeighbors(uint32_t node_id) const
    {
        NeighborArray nbr_array;
        GetNeighbors(node_id, nbr_array);
        return nbr_array;
    }

    virtual bool SortVertexNeighbors(uint32_t vertex_index, uint32_t sort_by, bool is_asc);
    virtual bool Sort(uint32_t sort_by, bool is_asc); //sort the neighbors of every vertex in the graph
    static void SortNeighbors(Neighbor *nbr_start, Neighbor *nbr_end, uint32_t sort_by, bool is_asc);

    virtual bool Load(const std::string &uri, const std::string &format);

    bool LoadFmtCompactBin(const std::string &file_path);
    bool LoadFmtCompactBin(const std::string &file_path, uint32_t max_loaded_nbr_num);
    bool LoadFmtCompactBin(IBinaryStream &stream, uint32_t max_loaded_nbr_num = UINT32_MAX);
    bool StoreFmtCompactBin(const std::string &file_path) const;
    bool StoreFmtCompactBin(IBinaryStream &stream) const;

    bool LoadFmtEdgeListBin(const std::string &file_path);
    bool StoreFmtEdgeListBin(const std::string &file_path) const;

    bool LoadFmtVertexListBin(const std::string &file_path);
    bool StoreFmtVertexListBin(const std::string &file_path) const;

    bool LoadFmtSimpleBin(const std::string &file_path);
    bool StoreFmtSimpleBin(const std::string &file_path) const;

    bool LoadFmtYaleSparseMatrix(const std::string &file_path);
    bool StoreFmtYaleSparseMatrix(const std::string &file_path) const;

protected:
    uint32_t vertex_count_ = 0, stored_vertex_count_ = 0;
    uint64_t *vertex_offsets_ = nullptr;
    uint64_t edge_num_ = 0;
    Neighbor *neighbors_ = nullptr;

protected:
    bool ReadStdHeader(IBinaryStream &stream, std::string &format);
    bool WriteStdHeader(IBinaryStream &stream, const std::string &format) const;

protected:
    struct EdgeForLoad
    {
        uint32_t node;
        uint32_t nbr;
        EdgeWeightType weight;

        EdgeForLoad() {
            node = 0; nbr = 0;
        }

        bool operator < (const EdgeForLoad &rhs) const {
            if(node == rhs.node) {
                return nbr < rhs.nbr;
            }
            return node < rhs.node;
        }

        bool operator > (const EdgeForLoad &rhs) const {
            if (node == rhs.node) {
                return nbr > rhs.nbr;
            }
            return node > rhs.node;
        }
    };
};

typedef StdStaticGraph<uint32_t>::NeighborArray GraphNeighborArray_UInt32;
typedef StdStaticGraph<float>::NeighborArray GraphNeighborArray_Float;

//////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////////////////////////

template <class EdgeWeightType>
const IdWeight<EdgeWeightType>*
StdStaticGraph<EdgeWeightType>::GetNeighbor(uint32_t node_id, uint32_t nbr_id, bool is_binary_search) const
{
    if(node_id >= stored_vertex_count_) {
        return nullptr;
    }
    uint64_t node_start = vertex_offsets_[node_id];
    uint64_t node_end = vertex_offsets_[node_id+1];
    if(node_end <= node_start) {
        return nullptr;
    }

    Neighbor nbr(nbr_id);
    const Neighbor *nbr_ptr = nullptr;
    if(is_binary_search)
    {
        nbr_ptr = std::lower_bound(neighbors_+node_start, neighbors_+node_end, nbr, Neighbor::LessId);
        if(nbr_ptr->id != nbr.id) {
            nbr_ptr = nullptr;
        }
    }
    else
    {
        nbr_ptr = std::find(neighbors_+node_start, neighbors_+node_end, nbr);
    }

    if(nbr_ptr == nullptr || nbr_ptr == neighbors_ + node_end) {
        return nullptr;
    }
    return nbr_ptr;
}

template <class EdgeWeightType>
IdWeight<EdgeWeightType>*
StdStaticGraph<EdgeWeightType>::GetNeighbor(uint32_t node_id, uint32_t nbr_id, bool is_binary_search)
{
    if(node_id >= stored_vertex_count_) {
        return nullptr;
    }
    uint64_t node_start = vertex_offsets_[node_id];
    uint64_t node_end = vertex_offsets_[node_id+1];
    if(node_end <= node_start) {
        return nullptr;
    }

    Neighbor nbr(nbr_id);
    Neighbor *nbr_ptr = nullptr;
    if(is_binary_search)
    {
        nbr_ptr = std::lower_bound(neighbors_ + node_start, neighbors_ + node_end, nbr, Neighbor::LessId);
        if(nbr_ptr->id != nbr.id) {
            nbr_ptr = nullptr;
        }
    }
    else
    {
        nbr_ptr = std::find(neighbors_ + node_start, neighbors_ + node_end, nbr);
    }

    if(nbr_ptr == nullptr || nbr_ptr == neighbors_ + node_end) {
        return nullptr;
    }
    return nbr_ptr;
}

template <class EdgeWeightType>
void StdStaticGraph<EdgeWeightType>::Clear()
{
    vertex_count_ = stored_vertex_count_ = 0;
    if(vertex_offsets_ != nullptr) {
        delete[] vertex_offsets_;
        vertex_offsets_ = nullptr;
    }
    edge_num_ = 0;
    if(neighbors_ != nullptr) {
        delete[] neighbors_;
        neighbors_ = nullptr;
    }
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::SetEdgeWeight(uint32_t node_id, uint32_t nbr_id, EdgeWeightType val)
{
    if(node_id >= stored_vertex_count_) {
        return false;
    }
    uint64_t node_start = vertex_offsets_[node_id];
    uint64_t node_end = vertex_offsets_[node_id+1];
    if(node_end <= node_start) {
        return false;
    }

    Neighbor nbr(nbr_id);
    Neighbor *nbr_ptr = std::find(neighbors_+node_start, neighbors_+node_end, nbr);
    if(nbr_ptr == nullptr || nbr_ptr == neighbors_ + node_end) {
        return false;
    }

    nbr_ptr->weight = val;
    return true;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::GetNeighbors(uint32_t node_id, NeighborArray &nbr_array) const
{
    if(node_id >= stored_vertex_count_) {
        nbr_array.size = 0;
        return false;
    }
    uint64_t node_start = vertex_offsets_[node_id];
    uint64_t node_end = vertex_offsets_[node_id+1];
    if(node_end <= node_start) {
        nbr_array.items = nullptr;
        nbr_array.size = 0;
    }
    nbr_array.items = neighbors_ + node_start;
    nbr_array.size = (uint32_t)(node_end - node_start);
    return true;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::SortVertexNeighbors(uint32_t vertex_index, uint32_t sort_by, bool is_asc)
{
    if(vertex_index >= vertex_count_) {
        return false;
    }
    if(vertex_index >= stored_vertex_count_) {
        return true;
    }

    uint64_t node_start = vertex_offsets_[vertex_index];
    uint64_t node_end = vertex_offsets_[vertex_index+1];
    if(node_start < node_end) {
        SortNeighbors(neighbors_+node_start, neighbors_+node_end, sort_by, is_asc);
    }
    return true;
}

//static
template <class EdgeWeightType>
void StdStaticGraph<EdgeWeightType>::SortNeighbors(
    Neighbor *nbr_start, Neighbor *nbr_end, uint32_t sort_by, bool is_asc)
{
    if(is_asc)
    {
        switch(sort_by)
        {
        case SORT_BY_WEIGHT:
            std::sort(nbr_start, nbr_end, Neighbor::LessWeight);
            break;
        case SORT_BY_IDX:
        default:
            std::sort(nbr_start, nbr_end, Neighbor::LessId);
            break;
        }
    }
    else
    {
        switch(sort_by)
        {
        case SORT_BY_WEIGHT:
            std::sort(nbr_start, nbr_end, Neighbor::GreaterWeight);
            break;
        case SORT_BY_IDX:
        default:
            std::sort(nbr_start, nbr_end, Neighbor::GreaterId);
            break;
        }
    }
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::Sort(uint32_t sort_by, bool is_asc)
{
    bool ret = true;
    for(uint32_t vertex_idx = 0; ret && vertex_idx < stored_vertex_count_; ++vertex_idx)
    {
        ret = SortVertexNeighbors(vertex_idx, sort_by, is_asc);
    }
    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::Load(const std::string &uri, const std::string &format_str)
{
    (void)format_str;
    return LoadFmtCompactBin(uri);
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::LoadFmtCompactBin(const std::string &file_path)
{
    Clear();
    BinFileStream stream;
    bool ret = stream.OpenForRead(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    std::string format_str;
    ret = ReadStdHeader(stream, format_str);
    if(!ret) {
        return false;
    }
    if(String::CaseCmp(format_str.c_str(), "CompactBinaryGraph") != 0) {
        LogError("Invalid format string");
        return false;
    }

    uint64_t offset_base = 0;
    uint32_t vertex_nbr_num = 0;
    vertex_offsets_ = new uint64_t[stored_vertex_count_ + 1];
    vertex_offsets_[0] = offset_base;
    for(uint32_t vertex_idx = 0; ret && vertex_idx < stored_vertex_count_; vertex_idx++)
    {
        ret = stream.Read((char*)&vertex_nbr_num, sizeof(vertex_nbr_num));
        offset_base += vertex_nbr_num;
        vertex_offsets_[vertex_idx+1] = offset_base;
    }

    if(offset_base != edge_num_) {
        LogError("Inconsistent data, edge num %I64u vs. %I64u", offset_base, edge_num_);
        return false;
    }

    neighbors_ = new Neighbor[(size_t)edge_num_];
    for(uint64_t edge_idx = 0; ret && edge_idx < edge_num_; edge_idx++)
	{
        Neighbor &nbr = neighbors_[edge_idx];
        ret = ret && stream.Read((char*)&nbr.id, sizeof(nbr.id));
        ret = ret && stream.Read((char*)&nbr.weight, sizeof(nbr.weight));
    }
    return ret;
}

struct VertexNbrNumPair
{
    uint32_t nbr_count, loaded_nbr_count;

    VertexNbrNumPair()
    {
        nbr_count = 0;
        loaded_nbr_count = 0;
    }

    void Set(uint32_t num, uint32_t loaded_num)
    {
        nbr_count = num;
        loaded_nbr_count = loaded_num;
    }
};

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::LoadFmtCompactBin(const std::string &file_path, uint32_t max_loaded_nbr_num)
{
    Clear();
    BinFileStream stream;
    bool ret = stream.Open(file_path.c_str());
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    return LoadFmtCompactBin(stream, max_loaded_nbr_num);
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::LoadFmtCompactBin(IBinaryStream &stream, uint32_t max_loaded_nbr_num)
{
    Clear();
    std::string format_str;
    bool ret = ReadStdHeader(stream, format_str);
    if(!ret) {
        return false;
    }
    if(String::CaseCmp(format_str.c_str(), "CompactBinaryGraph") != 0) {
        LogError("Invalid format string");
        return false;
    }

    uint64_t loaded_edge_num = 0, total_edge_num = 0;
    std::vector<VertexNbrNumPair> vertex_nbr_num_list(stored_vertex_count_);
    uint32_t offset_base = 0;
    uint32_t vertex_nbr_num = 0, loaded_vertex_nbr_num = 0, max_nbr_num = 0;
    vertex_offsets_ = new uint64_t[stored_vertex_count_ + 1];
    vertex_offsets_[0] = offset_base;
    for(uint32_t vertex_idx = 0; ret && vertex_idx < stored_vertex_count_; vertex_idx++)
    {
        ret = stream.Read((char*)&vertex_nbr_num, sizeof(vertex_nbr_num));
        loaded_vertex_nbr_num = min(vertex_nbr_num, max_loaded_nbr_num);
        vertex_nbr_num_list[vertex_idx].Set(vertex_nbr_num, loaded_vertex_nbr_num);

        if(max_nbr_num < vertex_nbr_num) {
            max_nbr_num = vertex_nbr_num;
        }

        total_edge_num += vertex_nbr_num;
        offset_base += loaded_vertex_nbr_num;
        vertex_offsets_[vertex_idx+1] = offset_base;
    }
    if(total_edge_num != edge_num_) {
        LogError("Inconsistent data, edge num %I64u vs. %I64u", total_edge_num, edge_num_);
        return false;
    }
    loaded_edge_num = offset_base;
    edge_num_ = loaded_edge_num;

    Neighbor *nbr_arr = new Neighbor[max((uint32_t)1, max_nbr_num)];
    neighbors_ = loaded_edge_num > 0 ? new Neighbor[(size_t)loaded_edge_num] : nullptr;
    for(uint32_t vertex_idx = 0; ret && vertex_idx < stored_vertex_count_; vertex_idx++)
    {
        const VertexNbrNumPair &vtx_nbr_nbr = vertex_nbr_num_list[vertex_idx];
        for(uint32_t nbr_idx = 0; nbr_idx < vtx_nbr_nbr.nbr_count; nbr_idx++)
        {
            Neighbor &nbr = nbr_arr[nbr_idx];
            ret = ret && stream.Read((char*)&nbr.id, sizeof(nbr.id));
            ret = ret && stream.Read((char*)&nbr.weight, sizeof(nbr.weight));
        }

        if(vtx_nbr_nbr.loaded_nbr_count > 0)
        {
            std::sort(nbr_arr, nbr_arr + vtx_nbr_nbr.nbr_count, Neighbor::GreaterWeight);
            memcpy(neighbors_ + vertex_offsets_[vertex_idx], nbr_arr, vtx_nbr_nbr.loaded_nbr_count * sizeof(Neighbor));
        }
    }
    delete []nbr_arr;

    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::StoreFmtCompactBin(const std::string &file_path) const
{
    BinFileStream stream;
    bool ret = stream.OpenForWrite(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    return StoreFmtCompactBin(stream);
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::StoreFmtCompactBin(IBinaryStream &stream) const
{
    std::string format_str = "CompactBinaryGraph";
    bool ret = WriteStdHeader(stream, format_str);

    for(uint32_t vertex_idx = 0; ret && vertex_idx < stored_vertex_count_; vertex_idx++)
	{
        uint32_t vertex_nbr_num = (uint32_t)(vertex_offsets_[vertex_idx + 1] - vertex_offsets_[vertex_idx]);
        ret = stream.Write((const char*)&vertex_nbr_num, sizeof(vertex_nbr_num));
    }

    for(uint64_t edge_idx = 0; ret && edge_idx < edge_num_; edge_idx++)
	{
        const Neighbor &nbr = neighbors_[edge_idx];
        ret = ret && stream.Write((const char*)&nbr.id, sizeof(nbr.id));
        ret = ret && stream.Write((const char*)&nbr.weight, sizeof(nbr.weight));
    }
    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::LoadFmtEdgeListBin(const std::string &file_path)
{
    Clear();
    BinFileStream stream;
    bool ret = stream.OpenForRead(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    string format_str;
    ret = ReadStdHeader(stream, format_str);
    if(!ret) {
        return false;
    }
    if(String::CaseCmp(format_str.c_str(), "EdgeListGraph") != 0) {
        LogError("Invalid format string");
        return false;
    }

    vertex_offsets_ = new uint64_t[stored_vertex_count_ + 1];
    neighbors_ = new Neighbor[(size_t)edge_num_];
    uint32_t node_id = 0, last_node = 0;
    vertex_offsets_[0] = 0;
    for(uint64_t edge_idx = 0; ret && edge_idx < edge_num_; edge_idx++)
	{
        Neighbor &nbr = neighbors_[edge_idx];
        ret = ret && stream.Read((char*)&node_id, sizeof(node_id));
        ret = ret && stream.Read((char*)&nbr.id, sizeof(nbr.id));
        ret = ret && stream.Read((char*)&nbr.weight, sizeof(nbr.weight));
        if(node_id != last_node)
        {
            for(uint32_t node_idx = last_node + 1; node_idx <= node_id; node_idx++) {
                vertex_offsets_[node_idx] = edge_idx;
            }
            last_node = node_id;
        }
    }
    for(uint32_t node_idx = last_node + 1; node_idx <= stored_vertex_count_; node_idx++) {
        vertex_offsets_[node_idx] = edge_num_;
    }

    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::StoreFmtEdgeListBin(const std::string &file_path) const
{
    BinFileStream stream;
    bool ret = stream.OpenForWrite(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    string format_str = "EdgeListGraph";
    ret = WriteStdHeader(stream, format_str);

    for(uint32_t node_idx = 0; node_idx < stored_vertex_count_; node_idx++)
	{
        uint64_t offset_start = vertex_offsets_[node_idx];
        uint64_t offset_end = vertex_offsets_[node_idx+1];
        for(uint64_t edge_idx = offset_start; edge_idx < offset_end; edge_idx++)
		{
            const Neighbor &nbr = neighbors_[edge_idx];
            ret = ret && stream.Write((const char*)&node_idx, sizeof(node_idx));
            ret = ret && stream.Write((const char*)&nbr.id, sizeof(nbr.id));
            ret = ret && stream.Write((const char*)&nbr.weight, sizeof(nbr.weight));
        }
    }

    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::LoadFmtVertexListBin(const std::string &file_path)
{
    Clear();
    BinFileStream stream;
    bool ret = stream.OpenForRead(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    string format_str;
    ret = ReadStdHeader(stream, format_str);
    if(!ret) {
        return false;
    }
    if(String::CaseCmp(format_str.c_str(), "VertexListGraph") != 0) {
        LogError("Invalid format string");
        return false;
    }

    vertex_offsets_ = new uint64_t[stored_vertex_count_ + 1];
    neighbors_ = new Neighbor[(size_t)edge_num_];
    uint32_t node_id = 0, nbr_count = 0;
    uint64_t edge_count_now = 0;
    while(ret)
	{
        ret = ret && stream.Read((char*)&node_id, sizeof(node_id));
        ret = ret && stream.Read((char*)&nbr_count, sizeof(nbr_count));
        for(uint32_t nbr_idx = 0; ret && nbr_idx < nbr_count; nbr_idx++, edge_count_now++)
		{
            if(edge_count_now >= edge_num_) {
                LogError("More than %d edges detected", edge_num_);
                return false;
            }
            Neighbor &nbr = neighbors_[edge_count_now];
            ret = ret && stream.Read((char*)&nbr.id, sizeof(nbr.id));
            ret = ret && stream.Read((char*)&nbr.weight, sizeof(nbr.weight));
        }

        if(edge_count_now >= edge_num_) {
            break;
        }
    }

    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::StoreFmtVertexListBin(const std::string &file_path) const
{
    BinFileStream stream;
    bool ret = stream.OpenForWrite(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    string format_str = "VertexListGraph";
    ret = WriteStdHeader(stream, format_str);

    for(uint32_t node_idx = 0; ret && node_idx < stored_vertex_count_; node_idx++)
    {
        uint64_t offset_start = vertex_offsets_[node_idx];
		uint64_t offset_end = vertex_offsets_[node_idx+1];
        if(offset_end > offset_start)
		{
            ret = ret && stream.Write((const char*)&node_idx, sizeof(node_idx));
            uint32_t nbr_count = (uint32_t)(offset_end - offset_start);
            ret = ret && stream.Write((const char*)&nbr_count, sizeof(nbr_count));
            for(uint64_t edge_idx = offset_start; ret && edge_idx < offset_end; edge_idx++)
			{
                const Neighbor &nbr = neighbors_[edge_idx];
                ret = ret && stream.Write((const char*)&nbr.id, sizeof(nbr.id));
                ret = ret && stream.Write((const char*)&nbr.weight, sizeof(nbr.weight));
            }
        }
    }
    uint32_t node_id_end = UINT32_MAX;
    ret = ret && stream.Write((const char*)&node_id_end, sizeof(node_id_end));

    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::LoadFmtSimpleBin(const std::string &file_path)
{
    Clear();
    BinFileStream stream;
    bool ret = stream.OpenForRead(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    uint64_t edge_count = 0;
    ret = stream.Read((char*)&edge_count, sizeof(edge_count));
    if(!ret) {
        return false;
    }

    std::vector<EdgeForLoad> edge_list(2 * (size_t)edge_count);
    for(uint32_t edge_idx = 0; ret && edge_idx < edge_count; edge_idx++)
    {
        EdgeForLoad &edge = edge_list[2 * edge_idx];
        ret = ret && stream.Read((char*)&edge.node, sizeof(edge.node));
        ret = ret && stream.Read((char*)&edge.nbr, sizeof(edge.nbr));
        ret = ret && stream.Read((char*)&edge.weight, sizeof(edge.weight));
        EdgeForLoad &edge2 = edge_list[2 * edge_idx + 1];
        edge2.node = edge.nbr;
        edge2.nbr = edge.node;
        edge2.weight = edge.weight;
    }
    std::sort(edge_list.begin(), edge_list.end());

    edge_num_ = 2 * edge_count;
    neighbors_ = new Neighbor[(size_t)edge_num_];
    std::vector<uint64_t> vertex_list;
    uint32_t last_node = 0;
    vertex_list.push_back(0);
    for(uint64_t edge_idx = 0; edge_idx < edge_num_; edge_idx++)
	{
        Neighbor &nbr = neighbors_[edge_idx];
        const EdgeForLoad &edge = edge_list[(size_t)edge_idx];
        nbr.id = edge.nbr;
        nbr.weight = edge.weight;
        if(edge.node != last_node) {
            for(uint32_t node_idx = last_node + 1; node_idx <= edge.node; node_idx++) {
                vertex_list.push_back(edge_idx);
            }
            last_node = edge.node;
        }
    }

    vertex_count_ = stored_vertex_count_ = (uint32_t)vertex_list.size();
    vertex_offsets_ = new uint64_t[stored_vertex_count_];
    for(uint32_t vertex_idx = 0; vertex_idx < (uint32_t)vertex_list.size(); vertex_idx++) {
        vertex_offsets_[vertex_idx] = vertex_list[vertex_idx];
    }

    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::StoreFmtSimpleBin(const std::string &file_path) const
{
    BinFileStream stream;
    bool ret = stream.OpenForWrite(file_path);
    if(!ret) {
        LogError("Fail to open file %s", file_path.c_str());
        return false;
    }

    ret = ret && stream.Write((const char*)&edge_num_, sizeof(edge_num_));
    for(uint32_t node_idx = 0; ret && node_idx < stored_vertex_count_; node_idx++)
	{
        uint64_t offset_start = vertex_offsets_[node_idx];
		uint64_t offset_end = vertex_offsets_[node_idx+1];
        for(uint64_t edge_idx = offset_start; ret && edge_idx < offset_end; edge_idx++)
		{
            const Neighbor &nbr = neighbors_[edge_idx];
            ret = ret && stream.Write((const char*)&node_idx, sizeof(node_idx));
            ret = ret && stream.Write((const char*)&nbr.id, sizeof(nbr.id));
            ret = ret && stream.Write((const char*)&nbr.weight, sizeof(nbr.weight));
        }
    }

    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::LoadFmtYaleSparseMatrix(const std::string &file_path)
{
    Clear();
    std::string matrix_file = file_path + "_dim";
    std::ifstream fi(matrix_file.c_str());
    if(!fi) {
        LogError("Fail to open the dim file");
        return false;
    }

    fi >> vertex_count_ >> edge_num_;
    stored_vertex_count_ = vertex_count_;
    fi.close();
    fi.clear();

    vertex_offsets_ = new uint64_t[stored_vertex_count_ + 1];
    if(vertex_offsets_ == nullptr) {
        LogError("Failed to alloc mem for vertex offsets");
        return false;
    }

    neighbors_ = new Neighbor[(size_t)edge_num_];
    if(neighbors_ == nullptr)
	{
        delete[] vertex_offsets_;
        vertex_offsets_ = nullptr;
        LogError("Failed to alloc mem for edges");
        return false;
    }

    matrix_file = file_path + "_row";
    fi.open(matrix_file.c_str());
    if(!fi) {
        LogError("Cannot open the row file");
        return false;
    }

    for(uint32_t row_idx = 0; row_idx <= stored_vertex_count_; row_idx++) {
        fi >> vertex_offsets_[row_idx];
    }
    fi.close();
    fi.clear();

    matrix_file = file_path + "_col";
    fi.open(matrix_file.c_str());
    if(!fi) {
        LogError("Cannot open the col file");
        return false;
    }
    for(uint64_t edge_idx = 0; edge_idx < edge_num_; edge_idx++) {
        fi >> neighbors_[edge_idx].id;
    }
    fi.close();
    fi.clear();

    matrix_file = file_path + "_nz";
    fi.open(matrix_file.c_str());
    if(!fi) {
        LogError("Cannot open the nz file");
        return false;
    }

    for(uint64_t edge_idx = 0; edge_idx < edge_num_; edge_idx++) {
        fi >> neighbors_[edge_idx].weight;
    }
    fi.close();
    return true;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::StoreFmtYaleSparseMatrix(const std::string &file_path) const
{
    ofstream out;
    string matrix_file = file_path + "_dim";
    out.open(matrix_file.c_str(), ios::binary);
    if(!out) {
        LogError("Cannot open the dim file");
        return false;
    }
    out << stored_vertex_count_ << "\n" << edge_num_;
    out.close();

    matrix_file = file_path + "_row";
    out.open(matrix_file.c_str(), ios::binary);
    if(!out) {
        LogError("Cannot open the row file");
        return false;
    }

    for(uint32_t row_idx = 0; row_idx <= stored_vertex_count_; row_idx++)
    {
        if(row_idx > 0) {
            out << "\n";
        }
        out << vertex_offsets_[row_idx];
    }
    out.close();

    matrix_file = file_path + "_col";
    out.open(matrix_file.c_str());
    if(!out) {
        LogError("Cannot open the col file");
        return false;
    }
    for(uint64_t edge_idx = 0; edge_idx < edge_num_; edge_idx++)
    {
        if(edge_idx > 0) {
            out << "\n";
        }
        out << neighbors_[edge_idx].id;
    }
    out.close();

    matrix_file = file_path + "_nz";
    out.open(matrix_file.c_str());
    if(!out) {
        LogError("Cannot open the nz file");
        return false;
    }
    for(uint64_t edge_idx = 0; edge_idx < edge_num_; edge_idx++)
    {
        if(edge_idx > 0) {
            out << "\n";
        }
        out << neighbors_[edge_idx].weight;
    }
    out.close();
    return out.good();
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::ReadStdHeader(IBinaryStream &stream, std::string &format_str)
{
    bool ret = true;
    BinStreamHelper::ReadString16(stream, format_str);
    uint32_t version = 0;
    ret = ret && stream.Read((char*)&version, sizeof(version));

    ret = ret && stream.Read((char*)&vertex_count_, sizeof(vertex_count_));
    stored_vertex_count_ = vertex_count_;
    if(version > 100) {
        ret = ret && stream.Read((char*)&stored_vertex_count_, sizeof(stored_vertex_count_));
    }

	if (version >= 300)
	{
		ret = ret && stream.Read(edge_num_);
	}
	else
	{
		uint32_t edge_num_32 = 0;
		ret = ret && stream.Read(edge_num_32);
		edge_num_ = edge_num_32;
	}

    uint16_t nEdgeWeightValueSize = 0;
    ret = ret && stream.Read((char*)&nEdgeWeightValueSize, sizeof(nEdgeWeightValueSize));
    if(!ret) {
        return false;
    }
    if(nEdgeWeightValueSize != sizeof(EdgeWeightType)) {
        LogError("Inconsistent edge weight size: %d vs. %d",
            nEdgeWeightValueSize, sizeof(EdgeWeightType));
        return false;
    }
    return ret;
}

template <class EdgeWeightType>
bool StdStaticGraph<EdgeWeightType>::WriteStdHeader(IBinaryStream &stream, const std::string &format_str) const
{
    bool ret = BinStreamHelper::WriteString16(stream, format_str);
    uint32_t version = 300;
    ret = ret && stream.Write((char*)&version, sizeof(version));

    ret = ret && stream.Write((const char*)&vertex_count_, sizeof(vertex_count_));
    ret = ret && stream.Write((const char*)&stored_vertex_count_, sizeof(stored_vertex_count_));
    ret = ret && stream.Write(edge_num_);
    uint16_t nEdgeWeightValueSize = sizeof(EdgeWeightType);
    ret = ret && stream.Write((const char*)&nEdgeWeightValueSize, sizeof(nEdgeWeightValueSize));
    if(!ret) {
        return false;
    }
    return ret;
}

} //end of namespace
