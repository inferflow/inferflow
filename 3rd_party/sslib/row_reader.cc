#include "row_reader.h"
#include "string_util.h"
#include "string.h"
#include "macro.h"

namespace sslib
{

////////////////////////////////////////////////////////////////////////////////
// class FileRowReader

FileRowReader::FileRowReader()
{
    options_.ignore_empty_lines = true;
    is_end_ = false;
    current_row_ = 0;
    uct_.Init();
}

FileRowReader::~FileRowReader()
{
    stream_.Close();
}

bool FileRowReader::Init(const string &file_path)
{
    is_end_ = false;
    current_row_ = 0;
    uct_.Init();
    bool ret = stream_.OpenForRead(file_path);
    return ret;
}

int FileRowReader::NextRow(vector<wstring> &row)
{
    bool ret = true;
    row.clear();
    while (ret)
    {
        ret = stream_.GetLine(line_str_);
        Macro_RetIf(stream_.IsGood() ? 0 : -1, !ret);

        current_row_++;
        line_wstr_ = StringUtil::Utf8ToWideStr(line_str_);

        bool be_skip = options_.ignore_empty_lines;
        if (options_.ignore_empty_lines)
        {
            for (int ch_idx = 0; ch_idx < (int)line_wstr_.size(); ch_idx++)
            {
                const auto &uci = uct_.Get(line_wstr_[ch_idx]);
                if (uci.prime_type != UctPrime::Separator) {
                    be_skip = false;
                    break;
                }
            }
        }

        if (!be_skip) {
            break;
        }
    }

    WString::Split(line_wstr_, row, L"\t");
    if (options_.trim_column_str)
    {
        for (auto &str : row) {
            WString::Trim(str);
        }
    }

    return 1;
}

int FileRowReader::NextRow(vector<string> &row)
{
    bool ret = true;
    row.clear();
    while (ret)
    {
        ret = stream_.GetLine(line_str_);
        Macro_RetIf(stream_.IsGood() ? 0 : -1, !ret);

        current_row_++;
        line_wstr_ = StringUtil::Utf8ToWideStr(line_str_);

        bool be_skip = options_.ignore_empty_lines;
        if (options_.ignore_empty_lines)
        {
            for (int ch_idx = 0; ch_idx < (int)line_wstr_.size(); ch_idx++)
            {
                const auto &uci = uct_.Get(line_wstr_[ch_idx]);
                if (uci.prime_type != UctPrime::Separator) {
                    be_skip = false;
                    break;
                }
            }
        }

        if (!be_skip) {
            break;
        }
    }

    String::Split(line_str_, row, "\t");
    if (options_.trim_column_str)
    {
        for (auto &str : row) {
            String::Trim(str);
        }
    }

    return 1;
}

////////////////////////////////////////////////////////////////////////////////
// class BinStreamRowReader

BinStreamRowReader::BinStreamRowReader(ChainedStrStream *stream_ptr)
{
    stream_ = stream_ptr;
    options_.ignore_empty_lines = true;
    is_end_ = false;
    current_row_ = 0;
    uct_.Init();
}

BinStreamRowReader::~BinStreamRowReader()
{
    Close();
}

bool BinStreamRowReader::Init(ChainedStrStream *stream_ptr)
{
    stream_ = stream_ptr;
    uct_.Init();
    current_row_ = 0;
    return stream_ != nullptr;
}

int BinStreamRowReader::NextRow(vector<wstring> &row)
{
    bool ret = true;
    Macro_RetIf(-1, stream_ == nullptr);

    row.clear();
    while (ret)
    {
        ret = stream_->GetLine(line_str_);
        Macro_RetIf(stream_->IsGood() ? 0 : -1, !ret);

        current_row_++;
        line_wstr_ = StringUtil::Utf8ToWideStr(line_str_);

        bool be_skip = options_.ignore_empty_lines;
        if (options_.ignore_empty_lines)
        {
            for (int ch_idx = 0; ch_idx < (int)line_wstr_.size(); ch_idx++)
            {
                const auto &uci = uct_.Get(line_wstr_[ch_idx]);
                if (uci.prime_type != UctPrime::Separator) {
                    be_skip = false;
                    break;
                }
            }
        }

        if (!be_skip) {
            break;
        }
    }

    WString::Split(line_wstr_, row, L"\t");
    if (options_.trim_column_str)
    {
        for (auto &str : row) {
            WString::Trim(str);
        }
    }

    return 1;
}

int BinStreamRowReader::NextRow(vector<string> &row)
{
    bool ret = true;
    Macro_RetIf(-1, stream_ == nullptr);

    row.clear();
    while (ret)
    {
        ret = stream_->GetLine(line_str_);
        Macro_RetIf(stream_->IsGood() ? 0 : -1, !ret);

        current_row_++;
        line_wstr_ = StringUtil::Utf8ToWideStr(line_str_);

        bool be_skip = options_.ignore_empty_lines;
        if (options_.ignore_empty_lines)
        {
            for (int ch_idx = 0; ch_idx < (int)line_wstr_.size(); ch_idx++)
            {
                const auto &uci = uct_.Get(line_wstr_[ch_idx]);
                if (uci.prime_type != UctPrime::Separator) {
                    be_skip = false;
                    break;
                }
            }
        }

        if (!be_skip) {
            break;
        }
    }

    String::Split(line_str_, row, "\t");
    if (options_.trim_column_str)
    {
        for (auto &str : row) {
            String::Trim(str);
        }
    }

    return 1;
}

} //end of namespace
