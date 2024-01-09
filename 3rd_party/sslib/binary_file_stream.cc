#include "binary_file_stream.h"
#include "log.h"
#include "string_util.h"
#include "path.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef _WIN32
#   include <Windows.h>
#	define InvalidFileHandle INVALID_HANDLE_VALUE
#else
#endif

namespace sslib
{

using namespace std;

BinaryFileStream::BufInfo::BufInfo()
{
    data = nullptr;
    max_len = 0;
    buf_len = 0;
    cursor_in_buf = 0;
    buf_pos = 0;
}

void BinaryFileStream::BufInfo::Clear()
{
    buf_len = 0;
    cursor_in_buf = 0;
    buf_pos = 0;
}

bool BinaryFileStream::BufInfo::ReinitBuffer(uint32_t max_buf_len)
{
    if(data != nullptr)
    {
        delete[] data;
        data = nullptr;
    }
    Clear();

    max_len = max_buf_len;
    if(max_len > 0 && max_len <= 1024 * 1024 * 1024)
    {
        data = new char[max_len];
        if(data == nullptr) {
            return false;
        }
    }

    return true;
}

BinaryFileStream::BinaryFileStream()
{
    is_opened_ = false;
    has_error_ = false;
    rd_buf_.ReinitBuffer(DEFAULT_BUF_SIZE);
    wr_buf_.ReinitBuffer(DEFAULT_BUF_SIZE);

    offset_rd_ = 0;
    rd_count_ = 0;

    file_handle_ = InvalidFileHandle;
    mode_ = 0;
    file_size_ = 0;
}

BinaryFileStream::~BinaryFileStream()
{
    Close();
    Clear();
    rd_buf_.ReinitBuffer(0);
    wr_buf_.ReinitBuffer(0);
}

bool BinaryFileStream::Open(const char *file_path, uint32_t mode)
{
    Close();
    Clear();

    mode_ = mode;
#ifdef _WIN32
    unsigned long desired_access = 0, share_mode = 0, creation_disposition = 0;
    bool ret = ParseMode(mode, desired_access, share_mode, creation_disposition);
    if (!ret) {
        LogError("Failed to parse the mode parameter");
        return false;
    }

    file_handle_ = ::CreateFileA(file_path, desired_access, share_mode, nullptr,
        creation_disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == InvalidFileHandle)
    {
        has_error_ = true;
        LogError("Failed to call CreateFileA (Last error = %u)", GetLastError());
        return false;
    }
#else
    int flags = O_RDONLY;
    int mask = mode & (MODE_READ | MODE_WRITE);
    switch (mask)
    {
    case MODE_READ: flags = O_RDONLY; break;
    case MODE_WRITE: flags = O_WRONLY; break;
    default: flags = O_RDWR; break;
    }

    //flags |= O_LARGEFILE;
    if ((mode & MODE_TRUNC) != 0) {
        flags |= O_TRUNC;
    }
    if ((mode & MODE_CREATE) != 0) {
        flags |= O_CREAT;
    }
    if ((mode & MODE_APP) != 0) {
        flags |= O_APPEND;
    }
    file_handle_ = ::open(file_path, flags, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
#endif

    if(rd_buf_.data == nullptr && (mode_ & MODE_READ) != 0) {
        rd_buf_.ReinitBuffer(DEFAULT_BUF_SIZE);
    }
    if(wr_buf_.data == nullptr && (mode_ & MODE_WRITE) != 0) {
        wr_buf_.ReinitBuffer(DEFAULT_BUF_SIZE);
    }

    FileStat fs;
    File::GetFileStat(fs, file_path);
    file_size_ = fs.size;
    is_opened_ = true;
    return file_handle_ != InvalidFileHandle;
}

bool BinaryFileStream::OpenForRead(const char *file_path, bool skip_unicode_bom)
{
    bool ret = Open(file_path, MODE_READ);
    if(ret && skip_unicode_bom) {
        ret = SkipUnicodeBOM();
    }

    return ret;
}

bool BinaryFileStream::OpenForWrite(const char *file_path)
{
    bool ret = Open(file_path, MODE_WRITE|MODE_CREATE|MODE_TRUNC);
    return ret;
}

bool BinaryFileStream::Close()
{
    bool ret = true;
    if (is_opened_)
    {
        if (file_handle_ != InvalidFileHandle && (mode_ & MODE_WRITE) != 0) {
            ret = Flush();
        }
    }

    offset_rd_ = 0;
    rd_buf_.Clear();
    wr_buf_.Clear();

    if(file_handle_ != InvalidFileHandle)
    {
#ifdef _WIN32
        CloseHandle(file_handle_);
#else
        close(file_handle_);
#endif
        file_handle_ = InvalidFileHandle;
    }

    file_size_ = 0;
    is_opened_ = false;

    return ret;
}

void BinaryFileStream::Clear()
{
    has_error_ = false;
}

bool BinaryFileStream::IsOpen() const
{
    return file_handle_ != InvalidFileHandle;
}

void BinaryFileStream::SetEncryptionString(const string &str)
{
    encryption_string_ = str;
}

bool BinaryFileStream::SetRdBufferSize(uint32_t size)
{
    if(size <= 256 * 1024 * 1024 && rd_buf_.max_len != size)
    {
        bool ret = rd_buf_.ReinitBuffer(size);
        if(!ret) {
            has_error_ = true;
        }
        return ret;
    }
    return true;
}

bool BinaryFileStream::SetWrBufferSize(uint32_t size)
{
    if(size <= 256 * 1024 * 1024 && wr_buf_.max_len != size)
    {
        bool ret = wr_buf_.ReinitBuffer(size);
        if(!ret) {
            has_error_ = true;
        }
        return ret;
    }
    return true;
}

bool BinaryFileStream::Read(char *buf, size_t size, void *params)
{
    (void)params;
    if(!is_opened_ || buf == nullptr) {
        return false;
    }
    if(size == 0) {
        rd_count_ = 0;
        return true;
    }

    uint32_t encryption_len = (uint32_t)encryption_string_.length();
    const char *encryption_ptr = encryption_string_.c_str();

    bool ret = true;
    if(offset_rd_ >= rd_buf_.buf_pos && offset_rd_ < rd_buf_.buf_pos + rd_buf_.buf_len)
    {
        rd_buf_.cursor_in_buf = (uint32_t)(offset_rd_ - rd_buf_.buf_pos);
    }
    else
    {
        ret = UpdateRdBuffer(rd_buf_, offset_rd_);
        if(!ret || rd_buf_.buf_len == 0)
        {
            rd_count_ = 0;
            return false;
        }
    }

    uint32_t left = (uint32_t)size;
    while(ret && left > 0)
    {
        if(rd_buf_.cursor_in_buf + left <= rd_buf_.buf_len)
        {
            memcpy(buf, rd_buf_.data + rd_buf_.cursor_in_buf, left);
            if (encryption_len > 0)
            {
                for (uint32_t pos = 0; pos < left; pos++)
                {
                    buf[pos] -= encryption_ptr[(rd_buf_.buf_pos + rd_buf_.cursor_in_buf + pos) % encryption_len];
                }
            }
            rd_buf_.cursor_in_buf += left;
            offset_rd_ += left;
            rd_count_ = (uint32_t)size;
            break;
        }
        else
        {
            uint32_t read_size = rd_buf_.buf_len - rd_buf_.cursor_in_buf;
            memcpy(buf, rd_buf_.data + rd_buf_.cursor_in_buf, read_size);
            if (encryption_len > 0)
            {
                for (uint32_t pos = 0; pos < read_size; pos++)
                {
                    buf[pos] -= encryption_ptr[(rd_buf_.buf_pos + rd_buf_.cursor_in_buf + pos) % encryption_len];
                }
            }

            buf += read_size;
            left -= read_size;
            offset_rd_ += read_size;

            ret = UpdateRdBuffer(rd_buf_, rd_buf_.buf_pos + rd_buf_.buf_len);
            if(rd_buf_.buf_len == 0) {
                rd_count_ = (uint32_t)size - left;
                return false;
            }
        }
    }

    return ret;
}

bool BinaryFileStream::Write(const char *buf, size_t size, void *params)
{
    (void)params;
    if(!is_opened_ || buf == nullptr) {
        return false;
    }
    if(size == 0) {
        return true;
    }

    uint32_t encryption_len = (uint32_t)encryption_string_.length();
    const char *encryption_ptr = encryption_string_.c_str();

    bool ret = true;
    uint32_t left = (uint32_t)size;
    while(ret && left > 0)
    {
        if(wr_buf_.buf_len + left <= wr_buf_.max_len)
        {
            memcpy(wr_buf_.data + wr_buf_.buf_len, buf, left);
            if (encryption_len > 0)
            {
                for (uint32_t pos = wr_buf_.buf_len; pos < wr_buf_.buf_len + left; pos++)
                {
                    wr_buf_.data[pos] += encryption_ptr[(wr_buf_.buf_pos + pos) % encryption_len];
                }
            }
            wr_buf_.buf_len += left;
            break;
        }
        else
        {
            uint32_t write_size = wr_buf_.max_len - wr_buf_.buf_len;
            memcpy(wr_buf_.data + wr_buf_.buf_len, buf, write_size);
            buf += write_size;
            left -= write_size;
            if (encryption_len > 0)
            {
                for (uint32_t pos = wr_buf_.buf_len; pos < wr_buf_.buf_len + write_size; pos++)
                {
                    wr_buf_.data[pos] += encryption_ptr[(wr_buf_.buf_pos + pos) % encryption_len];
                }
            }
            wr_buf_.buf_len += write_size;

            ret = FlushWrBuffer(wr_buf_);
            if(!ret) {
                has_error_ = true;
                return false;
            }
        }
    }

    if(file_size_ < wr_buf_.buf_pos + wr_buf_.buf_len) {
        file_size_ = wr_buf_.buf_pos + wr_buf_.buf_len;
    }

    return ret;
}

bool BinaryFileStream::Flush()
{
    bool ret = FlushWrBuffer(wr_buf_);
    return ret;
}

bool BinaryFileStream::SeekRd(uint64_t offset, uint32_t seek_way)
{
    bool ret = true;
    switch(seek_way)
    {
    case SEEK_WAY_BEGIN:
        offset_rd_ = 0;
        break;
    case SEEK_WAY_END:
        offset_rd_ = file_size_;
        break;
    case SEEK_WAY_ABS:
        if(offset <= file_size_) {
            offset_rd_ = offset;
        }
        else {
            ret = false;
        }
        break;
    default:
        break;
    }

    return ret;
}

bool BinaryFileStream::SeekWr(uint64_t offset, uint32_t seek_way)
{
    uint64_t offset_wr = offset;
    switch(seek_way)
    {
    case SEEK_WAY_BEGIN:
        offset_wr = 0;
        break;
    case SEEK_WAY_END:
        offset_wr = file_size_;
        break;
    case SEEK_WAY_ABS:
        if(offset <= file_size_) {
            offset_wr = offset;
        }
        else {
            return false;
        }
        break;
    default:
        break;
    }

    if(offset_wr == wr_buf_.buf_pos && wr_buf_.buf_len == 0) {
        return true;
    }

    bool ret = FlushWrBuffer(wr_buf_);
    if(!ret)
    {
        has_error_ = true;
        return ret;
    }

    if(ret)
    {
        wr_buf_.buf_pos = offset_wr;
        wr_buf_.buf_len = 0;
        wr_buf_.cursor_in_buf = 0;
    }

    return ret;
}

uint64_t BinaryFileStream::TellRd() const
{
    return offset_rd_;
}

uint64_t BinaryFileStream::TellWr() const
{
    return wr_buf_.buf_pos + wr_buf_.buf_len;
}

bool BinaryFileStream::GetLine(std::string &line_str, char delim)
{
    bool ret = true;
    line_str.clear();

    if(offset_rd_ >= rd_buf_.buf_pos && offset_rd_ < rd_buf_.buf_pos + rd_buf_.buf_len)
    {
        rd_buf_.cursor_in_buf = (uint32_t)(offset_rd_ - rd_buf_.buf_pos);
    }
    else
    {
        ret = UpdateRdBuffer(rd_buf_, offset_rd_);
        if(!ret || rd_buf_.buf_len == 0)
        {
            rd_count_ = 0;
            return false;
        }
    }

    rd_count_ = 0;
    while(ret)
    {
        if(rd_buf_.cursor_in_buf >= rd_buf_.buf_len)
        {
            ret = UpdateRdBuffer(rd_buf_, rd_buf_.buf_pos + rd_buf_.buf_len);
            if(rd_buf_.buf_len == 0) {
                return line_str.size() > 0 ? ret : false;
            }
        }

        bool has_delim = false;
        uint32_t cursor = rd_buf_.cursor_in_buf;
        while(cursor < rd_buf_.buf_len)
        {
            if(rd_buf_.data[cursor] == delim)
            {
                has_delim = true;
                break;
            }
            cursor++;
        }

        rd_count_ += (cursor - rd_buf_.cursor_in_buf);
        if(cursor > rd_buf_.cursor_in_buf) {
            line_str.append(rd_buf_.data + rd_buf_.cursor_in_buf, cursor - rd_buf_.cursor_in_buf);
        }

        if(has_delim)
        {
            offset_rd_ += (cursor - rd_buf_.cursor_in_buf + 1);
            rd_buf_.cursor_in_buf = cursor + 1;
            break;
        }
        else
        {
            offset_rd_ += (cursor - rd_buf_.cursor_in_buf);
            rd_buf_.cursor_in_buf = cursor;
        }
    }

    if (delim == '\n' && line_str.length() > 0 && line_str[line_str.length() - 1] == '\r')
    {
        line_str.resize(line_str.length() - 1);
    }
    return ret;
}

//virtual
bool BinaryFileStream::GetLine(std::wstring &line_str, wchar_t delim)
{
    string line_text;
    bool ret = GetLine(line_text, (char)delim);
    StringUtil::Utf8ToWideStr(line_str, line_text);
    return ret;
}

#ifdef _WIN32
bool BinaryFileStream::ParseMode(uint32_t mode, unsigned long &desired_access,
    unsigned long &share_mode, unsigned long &creation_disposition)
{
    bool ret = true;
	desired_access = 0;
	share_mode = 0;
	creation_disposition = 0;

	//default value; use existed files
    if(mode & MODE_TRUNC) {
        creation_disposition = CREATE_ALWAYS;
    }
    else if(mode & MODE_CREATE) {
        creation_disposition = OPEN_ALWAYS;
    }
    else {
        creation_disposition = OPEN_EXISTING;
    }

    share_mode |= FILE_SHARE_READ;

    if((mode & MODE_APP) != 0) {
        desired_access |= FILE_APPEND_DATA;
    }

    if((mode & MODE_READ) != 0) {
        desired_access |= GENERIC_READ;
    }
    if((mode & MODE_WRITE) != 0) {
        desired_access |= GENERIC_WRITE;
    }

    return ret;
}
#endif //def _WIN32

bool BinaryFileStream::UpdateRdBuffer(BufInfo &rd_buf, uint64_t file_pointer)
{
    bool ret = true;
    rd_buf.buf_pos = file_pointer;
    rd_buf.buf_len = 0;
    rd_buf.cursor_in_buf = 0;

#ifdef _WIN32
    uint64_t dist_to_move_high = (uint64_t)(rd_buf.buf_pos >> 32);
    ret = SetFilePointer(file_handle_, (LONG)rd_buf.buf_pos, (PLONG)&dist_to_move_high, FILE_BEGIN)
        != INVALID_SET_FILE_POINTER;
    if(!ret)
    {
        has_error_ = true;
        return ret;
    }

    ret = ReadFile(file_handle_, rd_buf.data, (DWORD)rd_buf.max_len, (LPDWORD)&rd_buf.buf_len, nullptr) != FALSE;
#else
    ret = lseek(file_handle_, rd_buf.buf_pos, SEEK_SET) != off_t(-1);
    if (!ret)
    {
        has_error_ = true;
        return ret;
    }

    ssize_t nReadLen = read(file_handle_, rd_buf.data, rd_buf.max_len);
    if (nReadLen < 0)
    {
        has_error_ = true;
        return false;
    }
    rd_buf.buf_len = (uint32_t)nReadLen;
#endif

    return ret;
}

bool BinaryFileStream::FlushWrBuffer(BufInfo &wr_buf)
{
#ifdef _WIN32
    LONG dist_to_move_high = (LONG)(wr_buf.buf_pos >> 32);
    bool ret = SetFilePointer(file_handle_, (LONG)wr_buf.buf_pos, &dist_to_move_high, FILE_BEGIN)
        != INVALID_SET_FILE_POINTER;
    if(!ret)
    {
        has_error_ = true;
        return ret;
    }

    if(wr_buf.buf_len > 0)
    {
        DWORD bytes_written = 0;
        ret = WriteFile(file_handle_, wr_buf.data, (DWORD)wr_buf.buf_len, &bytes_written, nullptr) != FALSE;
        if(ret && bytes_written < wr_buf.buf_len)
        {
            has_error_ = true;
            ret = false;
        }
    }
#else
    bool ret = lseek(file_handle_, wr_buf.buf_pos, SEEK_SET) != off_t(-1);
    if (!ret)
    {
        has_error_ = true;
        return ret;
    }

    if (wr_buf.buf_len > 0) {
        ssize_t bytes_written = write(file_handle_, wr_buf.data, wr_buf.buf_len);
        if (ret && bytes_written < wr_buf.buf_len)
        {
            has_error_ = true;
            ret = false;
        }
    }

    auto new_offset = lseek(file_handle_, 0, SEEK_CUR);
    //LogKeyInfo("New offset: %I64u", new_offset);

    struct stat stat_buf;
    int rc = fstat(file_handle_, &stat_buf);
    if (rc == 0)
    {
        //LogKeyInfo("File size: %I64u", stat_buf.st_size);
        if (stat_buf.st_size != new_offset) {
            LogError("Inconsistent value: new offset vs. file size");
        }
    }
#endif

    wr_buf.buf_pos = wr_buf.buf_pos + wr_buf.buf_len;
    wr_buf.buf_len = 0;
    wr_buf.cursor_in_buf = 0;
    return ret;
}

//BOM: Byte Order Mark
bool BinaryFileStream::SkipUnicodeBOM()
{
    uint64_t file_len = GetFileLength();
    uint64_t offset = TellRd();
    if(offset != 0 || file_len <= 1) {
        return true;
    }

    char buf[4];
    bool ret = Read(buf, file_len >= 3 ? 3 : 2);
    if(ret)
    {
        if((buf[0] == '\xFE' && buf[1] == '\xFF') || (buf[0] == '\xFF' && buf[1] == '\xFE'))
        {
            SeekRd(2);
            return ret;
        }

        if(file_len >= 3 && buf[0] == '\xEF' && buf[1] == '\xBB' && buf[2] == '\xBF')
        {
            return ret;
        }
    }

    SeekRd(offset);
    return ret;
}

} //end of namespace
