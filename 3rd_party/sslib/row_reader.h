#pragma once

#include "unicode_table.h"
#include "binary_file_stream.h"
#include "chained_str_stream.h"

namespace sslib
{

class IRowReader
{
public:
    struct Options
    {
        bool ignore_empty_lines = true;
        bool trim_column_str = true;
    };

public:
    const Options& options() const {
        return options_;
    }

    Options& options() {
        return options_;
    }

    virtual int NextRow(vector<wstring> &row) = 0;
    virtual int NextRow(vector<string> &row) = 0;
    virtual uint64_t GetRowId() const = 0;
    virtual bool Close() = 0;

    virtual ~IRowReader() {};

protected:
    Options options_;
};

class FileRowReader : public IRowReader
{
public:
    FileRowReader();
    virtual ~FileRowReader();

    bool Init(const string &file_path);
    virtual int NextRow(vector<wstring> &row);
    virtual int NextRow(vector<string> &row);

    virtual uint64_t GetRowId() const {
        return current_row_;
    }
    virtual bool Close() {
        return stream_.Close();
    }

protected:
    UnicodeTable uct_;
    BinaryFileStream stream_;
    bool is_end_;
    string line_str_;
    wstring line_wstr_;
    uint64_t current_row_;
};

class BinStreamRowReader : public IRowReader
{
public:
    BinStreamRowReader(ChainedStrStream *stream_ptr = nullptr);
    virtual ~BinStreamRowReader();

    bool Init(ChainedStrStream *stream_ptr);
    virtual int NextRow(vector<wstring> &row);
    virtual int NextRow(vector<string> &row);

    virtual uint64_t GetRowId() const {
        return current_row_;
    }

    virtual bool Close()
    {
        stream_ = nullptr;
        return true;
    }

protected:
    UnicodeTable uct_;
    ChainedStrStream *stream_;
    bool is_end_;
    string line_str_;
    wstring line_wstr_;
    uint64_t current_row_;
};

} //end of namespace
