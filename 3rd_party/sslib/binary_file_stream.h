#pragma once

#ifdef _WIN32
	typedef void* FileHandle;
#else
//#   define _LARGEFILE_SOURCE
#   define _FILE_OFFSET_BITS 64 //define this macro before including *any* header files
#   ifndef _LARGEFILE64_SOURCE
#       define _LARGEFILE64_SOURCE
#   endif //ifndef _LARGEFILE64_SOURCE
#	define InvalidFileHandle -1
#   include <sys/types.h>
#   include <unistd.h>
    typedef int FileHandle;
#endif //_WIN32
#include "binary_stream.h"

namespace sslib
{

using namespace std;

class BinaryFileStream : public IBinaryStream
{
public:
    enum OpenModeEnum
    {
        MODE_READ   = 0x01,
        MODE_WRITE  = 0x02,
        MODE_CREATE = 0x04,
        MODE_TRUNC  = 0x08,
        MODE_APP    = 0x10,
        MODE_TEXT   = 0x80  //text mode, not supported now
    };

    const static uint32_t DEFAULT_BUF_SIZE = 4096;

public:
    BinaryFileStream();
    virtual ~BinaryFileStream();

    bool Open(const char *file_path, uint32_t mode = MODE_READ);
    bool OpenForRead(const char *file_path, bool skip_unicode_bom = true);
    bool OpenForWrite(const char *file_path);

    bool Open(const string &path, uint32_t mode = MODE_READ)
    {
        return Open(path.c_str(), mode);
    }
    bool OpenForRead(const string &path, bool skip_unicode_bom = true)
    {
        return OpenForRead(path.c_str(), skip_unicode_bom);
    }
    bool OpenForWrite(const string &path)
    { 
        return OpenForWrite(path.c_str());
    }

    bool Close();
    void Clear();

    void SetEncryptionString(const string &str);

    uint64_t GetFileLength() const {
        return file_size_;
    }
    bool SetRdBufferSize(uint32_t size);
    bool SetWrBufferSize(uint32_t size);

    bool IsOpen() const;
    virtual bool IsGood() const {
        return !has_error_;
    }
    virtual uint32_t RdCount() const { //similar with istream::gcount
        return rd_count_;
    }
    virtual bool Read(char *buf, size_t size, void *params = nullptr);
    virtual bool Write(const char *buf, size_t size, void *params = nullptr);
    virtual bool Flush();

    virtual bool SeekRd(uint64_t offset, uint32_t seek_way = 0);
    virtual bool SeekWr(uint64_t offset, uint32_t seek_way = 0);
    virtual uint64_t TellRd() const;
    virtual uint64_t TellWr() const;

    virtual bool GetLine(std::string &line_str, char delim = '\n');
    virtual bool GetLine(std::wstring &line_str, wchar_t delim = L'\n');

protected:
    struct BufInfo
    {
        char *data;
        uint32_t max_len;
        uint32_t buf_len;
        uint32_t cursor_in_buf;
        uint64_t buf_pos;

        BufInfo();
        void Clear();
        bool ReinitBuffer(uint32_t max_buf_len);
    };

protected:
    bool is_opened_, has_error_;
    BufInfo rd_buf_, wr_buf_;
    uint64_t offset_rd_;
    uint32_t rd_count_;

	FileHandle file_handle_;
    uint32_t mode_;
    uint64_t file_size_;

    string encryption_string_;

protected:
#   ifdef _WIN32
    bool ParseMode(uint32_t mode, unsigned long &desired_access,
        unsigned long &share_mode, unsigned long &creation_disposition);
#   endif
    bool UpdateRdBuffer(BufInfo &rd_buf, uint64_t file_pointer);
    bool FlushWrBuffer(BufInfo &wr_buf);

    bool SkipUnicodeBOM();
};

typedef BinaryFileStream BinFileStream;

} //end of namespace
