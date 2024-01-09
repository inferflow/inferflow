#pragma once

#include <string>
#include <vector>
#include <map>
#include <sys/types.h>
#include <sys/stat.h>
#include "string.h"
#include "macro.h"
#include "prime_types.h"

namespace sslib
{

using std::string;
using std::wstring;
using std::vector;
using std::map;

enum class FileTypeEnum
{
    Regular = 0, Dir = 1, Other = 99
};

struct FileInfo
{
    string path;
    FileTypeEnum type;

    FileInfo(const string &src_path = "", FileTypeEnum src_type = FileTypeEnum::Regular)
    {
        path = src_path;
        type = src_type;
    }
};

struct FileStat : public FileInfo
{
    uint64_t size; //total size, in bytes
    time_t atime; //time of last access
    time_t mtime; //time of last modification
    time_t ctime; //time of last status change

public:
    FileStat()
    {
        size = 0;
        atime = 0;
        mtime = 0;
        ctime = 0;
    }

    void Clear()
    {
        size = 0;
        atime = mtime = ctime = 0;
    }
};

class Path
{
public:
    static const int MaxPathLen = 1024;
#ifdef _WIN32
    static const char SeparatorChar = '\\';
#else
    static const char SeparatorChar = '/';
#endif

public:
    struct ListDirOptions
    {
        string filter;
        bool is_recursive;
        bool including_dirs;
        bool including_regular_files;
        bool res_with_full_path;
        bool res_with_prefix_dir;
        bool enable_log = false;

        ListDirOptions()
        {
            filter = "*";
            is_recursive = false;
            including_dirs = true;
            including_regular_files = true;
            res_with_full_path = false;
            res_with_prefix_dir = true;
        }
    };

public:
    Path();
    Path(const char *path_str);
    virtual ~Path();

    void Clear();
    bool Set(const string &path_str, bool to_full_path = true, bool normalize_path = true);
    bool Set(const char *path_str, bool to_full_path = true, bool normalize_path = true);
    bool SetDir(const string &dir, bool to_full_path, bool normalize_path);
    bool SetName(const char *name_str);
    bool SetExt(const char *ext);
    bool SetNameAndExt(const char *name_and_ext);
    bool SetNameAndExt(const string &name_and_ext);

    string GetPath() const;
    string GetDrive() const;
    string GetDir() const;
    string GetDirNoEndSlash() const;
    string GetNameAndExt() const;
    string GetName() const;
    string GetExt() const;

    void GetModulePath(bool is_calling_app = true);

    //static members
    static bool GetFileStat(FileStat &fs, const string &file_path);
    static bool FileExists(const string &path);
    static bool FileExists(const char *file_name);
    static bool DirExists(const string &path);
    static bool DirExists(const char *dir);
    static bool PathExists(const string &path);
    static bool PathExists(const char *path_name);

    static bool Mkdir(const char *dir);
    static bool Mkdir(const string &dir);
    static bool Mkdirs(const char *dir);
    static bool Mkdirs(const string &dir);

    static bool Move(const char *old_path, const char *new_path);
    static bool Delete(const char *file_path);

    static uint64_t GetFileLength(const string &file_path);
    static bool GetFileLength(const string &file_path, uint64_t &file_len);
    static bool GetFileLength(const char *file_path, uint64_t &file_len);

#ifdef _WIN32
	static bool CopyDir(const char *src_dir, const char *target_dir, const char *filter, bool is_recursive);
#endif
	static bool GetFileNames(vector<string> &file_names, const char *dir,
        const char *filter = nullptr, bool is_full_path = true);
    static bool ListDir(vector<string> &file_names, const char *dir,
        const ListDirOptions *options = nullptr);
    static bool ListDir(vector<string> &file_names, const string &dir,
        const ListDirOptions *options = nullptr);

	static string GetLaunchedDir();
    static string GetModuleDir(bool is_calling_app = false);
    static string NormalizeDir(const string &dir);

    static bool IsAbsolute(const string &path);

    static bool GetFileContent_Text(string &content, const string &file_path);
    static bool GetFileContent_Text(wstring &content, const string &file_path);
    static bool GetFileContent_Text(vector<string> &lines, const string &file_path);
    static bool GetFileContent_Text(vector<wstring> &lines, const string &file_path);
    static bool GetFileContent_Text(map<string, int> &lines,
        const string &file_path, int score_column = -1);
    static bool GetFileContent_Text(map<wstring, int> &lines,
        const string &file_path, int score_column = -1);
    static bool GetFileContent_Text(map<string, int, StrLessNoCase> &lines,
        const string &file_path, int score_column = -1);
    static bool GetFileContent_Text(map<wstring, int, WStrLessNoCase> &lines,
        const string &file_path, int score_column);
    static bool GetFileContent_Binary(string &content, const string &file_path);

    static bool SetFileContent(const string &file_path, const string &content);
    static bool SetFileContent(const string &file_path,
        const vector<string> &lines, const char *line_end = nullptr);
    static bool SetFileContent(const string &file_path,
        const vector<wstring> &lines, const char *line_end = nullptr);

protected:
    string drive_; // Examples: "c:", "", etc
    string dir_;   // Examples: "/data/", "c:\\xxx\\", "\\\\server\\a\\b\\"
    string name_;  // "file"
    string ext_;   // ".txt"

protected:
	bool SetFullPath(const string &full_path, bool normalize_dir = true);

    static bool ListDirInner(vector<string> &file_names, const string &dir,
        const string &prefix_dir, const ListDirOptions *options);
    static bool ListDirInner(vector<FileInfo> &file_names, const string &dir,
        const string &prefix_dir, const ListDirOptions *options);
}; //class

typedef Path File;

} //end of namespace
