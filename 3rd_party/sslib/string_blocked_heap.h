#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include "prime_types.h"
#include "log.h"
#include "binary_file_stream.h"

namespace sslib
{

class StringBlockedHeap
{
public:
    static const int MinBlockSize = 128;
    static const int DefaultBlockSize = 40960;

public:
    StringBlockedHeap(int block_size= DefaultBlockSize);
    virtual ~StringBlockedHeap();

    uint32_t GetBlockNum() const  {
        return (uint32_t)blocks_.size();
    }

    //Set block size (in byte). Old blocks are not affected.
    //block_size_: New block size
    void SetBlockSize(int block_size);
    //Get block size (in byte)
    int GetBlockSize(){ return block_size_; }

    char* AddWord(const std::string &word);
    char* AddWord(const char *word, int len = -1);
    char* AddString(const std::string &str);
    char* AddString(const char *str, int len = -1);
    char* New(int size);

    wchar_t* AddWord(const std::wstring &word);
    wchar_t* AddWord(const wchar_t *word, int len = -1);
    wchar_t* AddString(const std::wstring &str);
    wchar_t* AddString(const wchar_t *str, int len = -1);
    wchar_t* NewWchar(int size);

    void Reset();
    void Clear() {
        Reset();
    }

public:
    StringBlockedHeap(const StringBlockedHeap&) = delete;
    StringBlockedHeap& operator = (const StringBlockedHeap&) = delete;

protected:
    char* NewInner(int size);

private:
    int block_size_ = 0;

    std::vector<char*> blocks_; //all blocks
    int cur_block_idx_ = 0;     //index of the current memory block
    char *cur_block_ = nullptr; //current memory block (for containing new characters)
    int cur_position_ = 0;      //position of new characters
};

struct StringLocation
{
    uint32_t block_id = 0;
    uint32_t offset_in_block = 0;
    uint32_t length = 0;
};

template <typename CharType>
class BasicStringBlockedHeap
{
public:
    static const uint32_t MinBlockCapacity = 128;
    static const uint32_t MaxBlockCapacity = 1024 * 1024 * 1024;
    static const uint32_t DefaultBlockCapacity = 40960;

    struct Block
    {
        CharType *data = nullptr;
        uint32_t size = 0;
    };

public:
    BasicStringBlockedHeap(uint32_t block_size = DefaultBlockCapacity);
    virtual ~BasicStringBlockedHeap();

    void Clear();

    uint32_t BlockCount() const {
        return (uint32_t)blocks_.size();
    }
    const Block* GetBlock(uint32_t idx) const {
        return idx < (uint32_t)blocks_.size() ? &blocks_[idx] : nullptr;
    }

    //Set block capacity. Existing blocks are not affected.
    void SetBlockCapacity(uint32_t block_capacity);
    uint32_t BlockCapacity() const { return block_capacity_; }

    CharType* AddString(const std::basic_string<CharType> &str);
    CharType* AddString(const CharType *str, uint32_t len = UINT32_MAX);
    CharType* AddStringEx(StringLocation &loc, const std::basic_string<CharType> &str);
    CharType* AddStringEx(StringLocation &loc, const CharType *str, uint32_t len = UINT32_MAX);

    CharType* NewString(uint32_t str_len);
    CharType* NewStringEx(StringLocation &loc, uint32_t str_len);

    bool Load(IBinaryStream &reader);
    bool Load(const std::string &path);
    bool Save(IBinaryStream &writer) const;
    bool Save(const std::string &path) const;

protected:
    uint32_t block_capacity_ = 0;

    std::vector<Block> blocks_; //all blocks
    int32_t cur_block_idx_ = 0; //index of the current memory block
    uint32_t cur_position_ = 0; //position of new characters (in the current block)

protected:
    BasicStringBlockedHeap(const BasicStringBlockedHeap&) = delete;
    BasicStringBlockedHeap& operator = (const BasicStringBlockedHeap &) = delete;

    Block* BlockAt(uint32_t idx) {
        return idx < (uint32_t)blocks_.size() ? &blocks_[idx] : nullptr;
    }
};

typedef BasicStringBlockedHeap<char16_t> U16StringHeap;
typedef BasicStringBlockedHeap<char32_t> U32StringHeap;
typedef BasicStringBlockedHeap<wchar_t> WStringHeap;

////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename CharType>
BasicStringBlockedHeap<CharType>::BasicStringBlockedHeap(uint32_t block_capacity)
{
    SetBlockCapacity(block_capacity);
    cur_block_idx_ = -1;
    cur_position_ = 0;
}

template <typename CharType>
BasicStringBlockedHeap<CharType>::~BasicStringBlockedHeap()
{
    Clear();
}

template <typename CharType>
void BasicStringBlockedHeap<CharType>::SetBlockCapacity(uint32_t block_capacity)
{
    block_capacity_ = block_capacity < MinBlockCapacity ? MinBlockCapacity
        : (block_capacity > MaxBlockCapacity ? MaxBlockCapacity : block_capacity);
}

template <typename CharType>
CharType* BasicStringBlockedHeap<CharType>::AddString(const std::basic_string<CharType> &str)
{
    return AddString(str.c_str(), (uint32_t)str.size());
}

template <typename CharType>
CharType* BasicStringBlockedHeap<CharType>::AddString(const CharType *str, uint32_t len)
{
    if (str == nullptr) {
        return nullptr;
    }

    uint32_t size = len == UINT32_MAX ? (uint32_t)std::char_traits<CharType>::length(str) : len;
    StringLocation loc;
    CharType *buf = NewStringEx(loc, size);
    if (buf != nullptr)
    {
        memcpy(buf, str, size * sizeof(CharType));
        buf[size] = (CharType)0;
    }

    return buf;
}

template <typename CharType>
CharType* BasicStringBlockedHeap<CharType>::AddStringEx(StringLocation &loc,
    const std::basic_string<CharType> &str)
{
    return AddStringEx(loc, str.c_str(), (uint32_t)str.size());
}

template <typename CharType>
CharType* BasicStringBlockedHeap<CharType>::AddStringEx(StringLocation &loc,
    const CharType *str, uint32_t len)
{
    if (str == nullptr) {
        return nullptr;
    }

    uint32_t size = len == UINT32_MAX ? (uint32_t)std::char_traits<CharType>::length(str) : len;
    CharType *buf = NewStringEx(loc, size);
    if (buf != nullptr)
    {
        memcpy(buf, str, size * sizeof(CharType));
        buf[size] = (CharType)0;
    }

    return buf;
}

template <typename CharType>
CharType* BasicStringBlockedHeap<CharType>::NewString(uint32_t str_len)
{
    StringLocation loc;
    return NewStringEx(loc, str_len);
}

template <typename CharType>
CharType* BasicStringBlockedHeap<CharType>::NewStringEx(StringLocation &loc, uint32_t str_len)
{
    uint32_t data_len = str_len + 1;
    if (data_len >= MaxBlockCapacity)
    {
        loc.block_id = UINT32_MAX;
        loc.length = 0;
        loc.offset_in_block = 0;
        return nullptr;
    }

    Block *cur_block = BlockAt(cur_block_idx_);
    if (cur_block == nullptr || cur_position_ + data_len > cur_block->size)
    {
        Block new_block;
        new_block.size = data_len > block_capacity_ ? data_len : block_capacity_;
        new_block.data = new CharType[new_block.size];
        blocks_.push_back(new_block);
        cur_block_idx_ = (int)blocks_.size() - 1;
        cur_position_ = 0;
        cur_block = BlockAt(cur_block_idx_);

        LogDebugInfoD("A new block is added");
    }

    loc.block_id = cur_block_idx_;
    loc.offset_in_block = cur_position_;
    loc.length = str_len;

    CharType *ret = cur_block->data + cur_position_;
    cur_position_ += data_len;
    return ret;
}

template <typename CharType>
void BasicStringBlockedHeap<CharType>::Clear()
{
    int block_num = (int)blocks_.size();
    for (int idx = 0; idx < block_num; idx++)
    {
        if (blocks_[idx].data != nullptr) {
            delete[] blocks_[idx].data;
        }
    }
    blocks_.clear();

    //Reinitilaize
    cur_block_idx_ = -1;
    cur_position_ = 0;
}

template <typename CharType>
bool BasicStringBlockedHeap<CharType>::Load(IBinaryStream &reader)
{
    Clear();
    bool ret = reader.Read(block_capacity_);
    uint32_t block_count = 0;
    ret = ret && reader.Read(block_count);
    if (!ret) {
        return false;
    }

    for (uint32_t block_id = 0; ret && block_id < block_count; block_id++)
    {
        Block new_block;
        ret = ret && reader.Read(new_block.size);
        if (new_block.size > MaxBlockCapacity) {
            break;
        }

        new_block.data = new CharType[new_block.size];
        ret = ret && reader.Read((char*)new_block.data, new_block.size * sizeof(CharType));
        blocks_.push_back(new_block);
    }

    if (!ret) {
        Clear();
    }

    return ret;
}

template <typename CharType>
bool BasicStringBlockedHeap<CharType>::Load(const std::string &path)
{
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(path);
    if (!ret) {
        return false;
    }

    return Load(reader);
}

template <typename CharType>
bool BasicStringBlockedHeap<CharType>::Save(IBinaryStream &writer) const
{
    bool ret = writer.Write(block_capacity_);
    uint32_t block_count = (uint32_t)blocks_.size();
    ret = ret && writer.Write(block_count);
    for (uint32_t block_id = 0; ret && block_id < block_count; block_id++)
    {
        const auto &block = blocks_[block_id];
        ret = ret && writer.Write(block.size);
        ret = ret && writer.Write((const char*)block.data, block.size * sizeof(CharType));
    }

    return ret && writer.IsGood();
}

template <typename CharType>
bool BasicStringBlockedHeap<CharType>::Save(const std::string &path) const
{
    BinaryFileStream writer;
    bool ret = writer.OpenForWrite(path);
    if (!ret) {
        return false;
    }

    return Save(writer);
}

} //end of namespace
