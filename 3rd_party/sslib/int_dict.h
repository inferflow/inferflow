#pragma once

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include "prime_types.h"
#include "blocked_array.h"
#include "string_blocked_heap.h"
#include "hashtable.h"
#include "hash.h"
#include "binary_stream.h"

namespace sslib
{

typedef BlockedArray<uint32_t> UInt32Array;

class IntDict
{
public:
    enum PrintOptionsEnum
    {
        PrintOpt_MaskSortBy = 0x00FF,
        PrintOpt_SortById   = 0x0001,
        PrintOpt_SortByIdx  = 0x0002,
        PrintOpt_Desc       = 0x0100
    };

public:
    IntDict();
    virtual ~IntDict();

    uint32_t Size() const {
        return (uint32_t)items_.Size();
    }
    uint32_t GetItemNum() const {
        return (uint32_t)items_.Size();
    }
    uint32_t Id2Index(uint32_t item_id) const;
    uint32_t Index2Id(uint32_t item_idx) const;

    void Clear();
    uint32_t AddItem(uint32_t id, uint32_t idx = UINT32_MAX);

    bool Store(const std::string &path) const;
    bool Load(const std::string &path);
    bool Print(const std::string &path, uint32_t options) const;

    bool Write(IBinaryStream &stream) const;
    bool Read(IBinaryStream &stream);

protected:
    BlockedArray<uint32_t> items_;
    HashMapUInt32 id2idx_map_;
};

} //end of namespace
