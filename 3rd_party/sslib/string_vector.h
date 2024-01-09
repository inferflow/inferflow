#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include "string_blocked_heap.h"

namespace sslib
{

using std::string;
using std::basic_string;
using std::vector;

template <typename CharType>
class BasicStringVector
{
public:
    BasicStringVector();
    virtual ~BasicStringVector();

    uint32_t Size() const;
    const CharType* Get(uint32_t idx) const;

    void Clear();
    const CharType* Add(const basic_string<CharType> &str);

    bool Load(const string &path);
    bool Load(IBinaryStream &reader);
    bool Save(const string &path) const;
    bool Save(IBinaryStream &writer) const;

protected:
#pragma pack(push, 1)
    struct Location
    {
        uint16_t block_id = 0;
        uint32_t offset_in_block = 0;
    };
#pragma pack(pop)

protected:
    std::vector<Location> location_list_;
    BasicStringBlockedHeap<CharType> str_heap_;
};

typedef BasicStringVector<char> StringVector;
typedef BasicStringVector<wchar_t> WStringVector;
typedef BasicStringVector<char16_t> U16StringVector;
typedef BasicStringVector<char32_t> U32StringVector;

////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename CharType>
BasicStringVector<CharType>::BasicStringVector()
{
}

template <typename CharType>
BasicStringVector<CharType>::~BasicStringVector()
{
    Clear();
}

template <typename CharType>
uint32_t BasicStringVector<CharType>::Size() const
{
    return (uint32_t)location_list_.size();
}

template <typename CharType>
const CharType* BasicStringVector<CharType>::Get(uint32_t idx) const
{
    if (idx >= (uint32_t)location_list_.size()) {
        return nullptr;
    }

    const Location &loc = location_list_[idx];
    const auto *block = str_heap_.GetBlock(loc.block_id);
    if (block == nullptr) {
        return nullptr;
    }

    return block->data + loc.offset_in_block;
}

template <typename CharType>
void BasicStringVector<CharType>::Clear()
{
    location_list_.clear();
    str_heap_.Clear();
}

template <typename CharType>
const CharType* BasicStringVector<CharType>::Add(const basic_string<CharType> &str)
{
    StringLocation loc;
    CharType *buf = str_heap_.AddStringEx(loc, str);
    if (buf == nullptr) {
        return nullptr;
    }

    Location new_loc;
    new_loc.block_id = (uint16_t)loc.block_id;
    new_loc.offset_in_block = loc.offset_in_block;
    location_list_.push_back(new_loc);
    return buf;
}

template <typename CharType>
bool BasicStringVector<CharType>::Load(const string &path)
{
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(path);
    if (!ret) {
        return false;
    }

    ret = Load(reader);
    reader.Close();
    return ret;
}

template <typename CharType>
bool BasicStringVector<CharType>::Load(IBinaryStream &reader)
{
    bool ret = str_heap_.Load(reader);
    if (!ret) {
        return false;
    }

    uint32_t size = 0;
    ret = reader.Read(size);
    if (!ret) {
        return false;
    }

    location_list_.resize(size);
    Location *location_array = location_list_.data();
    ret = reader.Read((char*)location_array, size * sizeof(Location));
    return ret;
}

template <typename CharType>
bool BasicStringVector<CharType>::Save(const string &path) const
{
    BinaryFileStream writer;
    bool ret = writer.OpenForWrite(path);
    if (!ret) {
        return false;
    }

    ret = Save(writer);
    ret = ret && writer.Close();

    return ret && writer.IsGood();
}

template <typename CharType>
bool BasicStringVector<CharType>::Save(IBinaryStream &writer) const
{
    bool ret = str_heap_.Save(writer);
    if (!ret) {
        return false;
    }

    uint32_t size = (uint32_t)location_list_.size();
    ret = writer.Write(size);
    if (!ret) {
        return false;
    }

    const Location *location_array = location_list_.data();
    ret = writer.Write((const char*)location_array, size * sizeof(Location));
    return ret;
}

}  // namespace nlu
