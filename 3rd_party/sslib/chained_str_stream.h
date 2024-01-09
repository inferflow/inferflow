#pragma once

#include <vector>
#include "binary_stream.h"
#include "serializable.h"

namespace sslib
{

class ChainedStrStream : public IBinaryStream, public IBinSerializable
{
public:
    class Block
    {
    public:
        uint64_t capacity = 0;
        uint64_t size = 0;
        char *content = nullptr;
    };

    struct Position
    {
        uint64_t block_idx;
        uint64_t pos_in_block;

        void Set(uint32_t p_block_idx = 0, uint32_t p_pos_in_block = 0)
        {
            block_idx = p_block_idx;
            pos_in_block = p_pos_in_block;
        }
    };

public:
    ChainedStrStream();
    virtual ~ChainedStrStream();

    bool Clear();
    bool SetFromTextFile(const std::string &file_path);

    size_t Size() const;
    size_t TellRPos() const;
    size_t TellWPos() const;
    bool SeekR(size_t pos);
    bool SeekW(size_t pos);
    virtual bool Read(char *buf, size_t size, void *params = nullptr);
    virtual bool Write(const char *buf, size_t size, void *params = nullptr);

    size_t GetBlockNum() const;
    void SetBlockCapacity(uint32_t min_capacity, uint32_t max_capacity);
    bool GetBlock(size_t block_idx, Block &block) const;
    const Block* GetBlock(size_t block_idx) const;
    bool AddBlock(const Block &block);

    virtual bool Read(IBinStream &reader, void *params = nullptr) override;
    virtual bool Write(IBinStream &writer, void *params = nullptr) const override;

    virtual bool GetLine(std::string &line_str, char delim = '\n') override;
    virtual bool GetLine(std::wstring &line_str, wchar_t delim = L'\n') override;

protected:
    uint32_t min_capacity_, max_capacity_;
    std::vector<ChainedStrStream::Block> blocks_;
    ChainedStrStream::Position pos_r_, pos_w_;
};

} //end of namespace
