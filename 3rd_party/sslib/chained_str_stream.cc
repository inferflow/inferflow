#include "chained_str_stream.h"
#include <cassert>
#include <algorithm>
#include "macro.h"
#include "binary_file_stream.h"
#include "string_util.h"

using namespace std;

namespace sslib
{

ChainedStrStream::ChainedStrStream()
{
    min_capacity_ = 100 * 1024;
    max_capacity_ = 1024 * 1024;
    Clear();
}

ChainedStrStream::~ChainedStrStream()
{
    Clear();
}

bool ChainedStrStream::Clear()
{
    size_t block_num = blocks_.size();
    for(size_t block_idx = 0; block_idx < block_num; block_idx++)
    {
        ChainedStrStream::Block &block = blocks_[block_idx];
        assert((block_idx+1 < block_num && block.size == block.capacity)
            || (block_idx+1 == block_num && block.size <= block.capacity));

        if(block.content != nullptr)
        {
            delete[] block.content;
            block.capacity = 0;
            block.size = 0;
            block.content = nullptr;
        }
    }
    blocks_.clear();

    pos_r_.Set(0, 0);
    pos_w_.Set(0, 0);

    return true;
}

size_t ChainedStrStream::Size() const
{
    size_t size = 0;
    size_t block_num = blocks_.size();
    for(size_t block_idx = 0; block_idx < block_num; block_idx++)
    {
        const ChainedStrStream::Block &block = blocks_[block_idx];
        assert((block_idx+1 < block_num && block.size == block.capacity)
            || (block_idx+1 == block_num && block.size <= block.capacity));
        size += (size_t)block.size;
    }

    return size;
}

size_t ChainedStrStream::TellRPos() const
{
    size_t pos = 0;
    size_t block_num = blocks_.size();
    for(size_t block_idx = 0; block_idx < min<uint64_t>(pos_r_.block_idx, block_num); block_idx++)
    {
        const ChainedStrStream::Block &block = blocks_[block_idx];
        assert((block_idx+1 < block_num && block.size == block.capacity)
            || (block_idx+1 == block_num && block.size <= block.capacity));
        pos += (size_t)block.capacity;
    }

    pos += (size_t)pos_r_.pos_in_block;
    return pos;
}

size_t ChainedStrStream::TellWPos() const
{
    size_t pos = 0;
    size_t block_num = blocks_.size();
    for(size_t block_idx = 0; block_idx < min<uint64_t>(pos_w_.block_idx, block_num); block_idx++)
    {
        const ChainedStrStream::Block &block = blocks_[block_idx];
        assert((block_idx+1 < block_num && block.size == block.capacity)
            || (block_idx+1 == block_num && block.size <= block.capacity));
        pos += (size_t)block.capacity;
    }

    pos += (size_t)pos_w_.pos_in_block;
    return pos;
}

bool ChainedStrStream::SeekR(size_t pos)
{
    uint64_t block_num = (uint64_t)blocks_.size();
    uint64_t pos_remained = pos;
    size_t block_idx = 0;
    for (; block_idx < block_num; block_idx++)
    {
        ChainedStrStream::Block &block = blocks_[block_idx];
        assert((block_idx + 1 < block_num && block.size == block.capacity)
            || (block_idx + 1 == block_num && block.size <= block.capacity));

        if (block.content != nullptr)
        {
            if (pos_remained >= block.capacity)
            {
                pos_remained -= block.size;
            }
            else if (pos_remained > block.size)
            {
                pos_r_.block_idx = block_idx;
                pos_r_.pos_in_block = block.size;
                return false;
            }
            else
            {
                pos_r_.block_idx = block_idx;
                pos_r_.pos_in_block = pos_remained;
                return true;
            }
        }
    }

    //!!!
    if (pos_remained == 0)
    {
        pos_r_.block_idx = block_idx;
        pos_r_.pos_in_block = 0;
        return true;
    }

    return false;
}

bool ChainedStrStream::SeekW(size_t pos)
{
    size_t block_num = blocks_.size();
    size_t pos_remained = pos;
    size_t block_idx = 0;
    for (; block_idx < block_num; block_idx++)
    {
        ChainedStrStream::Block &block = blocks_[block_idx];
        assert((block_idx + 1 < block_num && block.size == block.capacity)
            || (block_idx + 1 == block_num && block.size <= block.capacity));

        if (block.content != nullptr)
        {
            if (pos_remained >= block.capacity) {
                pos_remained -= (size_t)block.size;
            }
            else if (pos_remained > block.size) {
                pos_w_.block_idx = block_idx;
                pos_w_.pos_in_block = block.size;
                return false;
            }
            else {
                pos_w_.block_idx = block_idx;
                pos_w_.pos_in_block = pos_remained;
                return true;
            }
        }
    }

    //!!!
    if (pos_remained == 0)
    {
        pos_w_.block_idx = block_idx;
        pos_w_.pos_in_block = 0;
        return true;
    }

    return false;
}

bool ChainedStrStream::Read(char *szBuf, size_t size, void *params)
{
    (void)params;
    if (size == 0) {
        return true;
    }

    size_t block_num = GetBlockNum();
    size_t size_remained = size;
    while (pos_r_.block_idx < block_num)
    {
        ChainedStrStream::Block &block = blocks_[(size_t)pos_r_.block_idx];
        size_t block_idx = (size_t)pos_r_.block_idx;
        assert((block_idx + 1 < block_num && block.size == block.capacity)
            || (block_idx + 1 == block_num && block.size <= block.capacity));
        (void)block_idx;

        if (block.size < pos_r_.pos_in_block + size_remained)
        {
            if (block.capacity > block.size) {
                return false;
            }

            memcpy(szBuf + (size - size_remained), block.content + (size_t)pos_r_.pos_in_block,
                (size_t)(block.size - pos_r_.pos_in_block));
            size_remained -= (size_t)(block.size - pos_r_.pos_in_block);
            pos_r_.block_idx++;
            pos_r_.pos_in_block = 0;
        }
        else
        {
            memcpy(szBuf + (size - size_remained), block.content + pos_r_.pos_in_block, size_remained);
            pos_r_.pos_in_block += size_remained;
            size_remained = 0;
            break;
        }
    }

    if (size_remained == 0) {
        return true;
    }

    return false;
}

bool ChainedStrStream::Write(const char *szBuf, size_t size, void *params)
{
    (void)params;
    if (size == 0) {
        return true;
    }

    size_t block_num = GetBlockNum();
    size_t size_remained = size;
    while (pos_w_.block_idx < block_num)
    {
        size_t block_idx = (size_t)pos_w_.block_idx;
        ChainedStrStream::Block &block = blocks_[block_idx];
        assert((block_idx + 1 < block_num && block.size == block.capacity)
            || (block_idx + 1 == block_num && block.size <= block.capacity));

        if (block.capacity < pos_w_.pos_in_block + size_remained)
        {
            block.size = block.capacity;
            memcpy(block.content + pos_w_.pos_in_block, szBuf + (size - size_remained),
                (size_t)(block.size - pos_w_.pos_in_block));
            size_remained -= (size_t)(block.size - pos_w_.pos_in_block);
            pos_w_.block_idx++;
            pos_w_.pos_in_block = 0;
        }
        else
        {
            memcpy(block.content + pos_w_.pos_in_block, szBuf + (size - size_remained), size_remained);
            pos_w_.pos_in_block += size_remained;
            size_remained = 0;

            if (block.size < pos_w_.pos_in_block) {
                block.size = pos_w_.pos_in_block;
            }
            break;
        }
    }

    while (size_remained > 0)
    {
        Block block;
        block.size = 0;
        block.capacity = min_capacity_;
        if (size_remained > min_capacity_) {
            block.capacity = max_capacity_;
        }

        block.content = new char[(size_t)block.capacity];
        block.size = min((size_t)block.capacity, size_remained);

        if (block.content != nullptr)
        {
            memcpy((char*)block.content, szBuf + (size - size_remained), (size_t)block.size);
            blocks_.push_back(block);

            pos_w_.block_idx = blocks_.size() - 1;
            pos_w_.pos_in_block = block.size;

            size_remained -= (size_t)block.size;
        }
        else
        { // fail to alloc memory
            return false;
        }
    }

    return true;
}

size_t ChainedStrStream::GetBlockNum() const
{
    return blocks_.size();
}

const ChainedStrStream::Block* ChainedStrStream::GetBlock(size_t block_idx) const
{
    size_t block_num = GetBlockNum();
    return block_idx < block_num ? &blocks_[block_idx] : nullptr;
}

bool ChainedStrStream::GetBlock(size_t block_idx, ChainedStrStream::Block &block) const
{
    size_t block_num = GetBlockNum();
    if(block_idx < block_num)
    {
        const ChainedStrStream::Block &blk = blocks_[block_idx];
        block.size = blk.size;
        block.capacity = blk.capacity;
        block.content = blk.content;
        return true;
    }

    return false;
}

bool ChainedStrStream::AddBlock(const Block &block)
{
    if(block.content == nullptr || block.size <= 0 || block.size > block.capacity) {
        return false;
    }

    size_t block_num = blocks_.size();
    if(block_num > 0)
    {
        ChainedStrStream::Block &last_block = blocks_[block_num-1];
        if(last_block.size != last_block.capacity) {
            return false;
        }
    }

    blocks_.push_back(block);
    return true;
}

void ChainedStrStream::SetBlockCapacity(uint32_t min_capacity, uint32_t max_capacity)
{
    min_capacity_ = min_capacity;
    max_capacity_ = max_capacity;
}

bool ChainedStrStream::Read(IBinStream &reader, void *params)
{
    (void)params;
    Clear();
    uint32_t block_num = 0;
    bool ret = reader.Read(block_num);
    Macro_RetIf(false, !ret || block_num > 0xFFFFFFF);

    for (uint32_t block_idx = 0; ret && block_idx < block_num; block_idx++)
    {
        Block cur_block;
        ret = ret && reader.Read(cur_block.capacity);
        ret = ret && reader.Read(cur_block.size);
        Macro_RetIf(false, cur_block.capacity >= 0xFFFFFFF || cur_block.size > cur_block.capacity);

        char *blockContent = new char[(size_t)cur_block.capacity];
        ret = ret && reader.Read(blockContent, (size_t)cur_block.size);
        cur_block.content = blockContent;
        blocks_.push_back(cur_block);
    }

    return ret && reader.IsGood();
}

bool ChainedStrStream::Write(IBinStream &writer, void *params) const
{
    (void)params;
    uint32_t block_num = (uint32_t)blocks_.size();
    bool ret = writer.Write(block_num);
    for(uint32_t block_idx = 0; ret && block_idx < block_num; block_idx++)
    {
        const auto &cur_block = blocks_[block_idx];
        ret = ret && writer.Write(cur_block.capacity);
        ret = ret && writer.Write(cur_block.size);
        ret = ret && writer.Write(cur_block.content, (size_t)cur_block.size);
    }

    return ret && writer.IsGood();
}

bool ChainedStrStream::GetLine(std::string &line_str, char delim)
{
    bool ret = true;
    line_str.clear();

    while (ret)
    {
        while (pos_r_.block_idx < (uint64_t)blocks_.size()
            && pos_r_.pos_in_block >= blocks_[(size_t)pos_r_.block_idx].size)
        {
            pos_r_.block_idx++;
            pos_r_.pos_in_block = 0;
        }

        if (pos_r_.block_idx >= blocks_.size()) {
            return line_str.size() > 0 ? ret : false;
        }

        const auto &cur_block = blocks_[(size_t)pos_r_.block_idx];
        bool has_delim = false;
        size_t cursor = (size_t)pos_r_.pos_in_block;
        for (; cursor < cur_block.size; cursor++)
        {
            if (cur_block.content[cursor] == delim) {
                has_delim = true;
                break;
            }
        }

        if (cursor > pos_r_.pos_in_block) {
            line_str.append(cur_block.content + pos_r_.pos_in_block, cursor - (size_t)pos_r_.pos_in_block);
        }

        if (has_delim) {
            pos_r_.pos_in_block = cursor + 1;
            break;
        }
        else {
            pos_r_.pos_in_block = cursor;
        }
    }

    if (delim == '\n' && line_str.length() > 0 && line_str[line_str.length() - 1] == '\r')
    {
        line_str.resize(line_str.length() - 1);
    }
    return ret;
}

bool ChainedStrStream::GetLine(std::wstring &line_str, wchar_t delim)
{
    char delim_char = (char)delim;
    string utf8_line_str;
    bool ret = GetLine(utf8_line_str, delim_char);
    StringUtil::Utf8ToWideStr(line_str, utf8_line_str);
    return ret;
}

bool ChainedStrStream::SetFromTextFile(const std::string &file_path)
{
    Clear();

    BinFileStream reader;
    bool ret = reader.OpenForRead(file_path);
    Macro_RetIf(false, !ret);

    string line_str;
    while (reader.GetLine(line_str))
    {
        bool bEndR = !line_str.empty() && line_str[line_str.size() - 1] == L'\r';
        Write(line_str.c_str(), bEndR ? line_str.size() - 1 : line_str.size());
        Write("\r\n", 2);
    }

    reader.Close();
    return true;
}

} //end of namespace
