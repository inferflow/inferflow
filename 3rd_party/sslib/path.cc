#include "path.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "binary_file_stream.h"
#include "string_util.h"
#include "log.h"
#ifdef _WIN32
#   include <Windows.h>
#else
#   include <dirent.h>
#   include <limits.h>
#   include <stdlib.h>
#endif //_WIN32

#ifdef _WIN32
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#endif //_WIN32

using namespace std;

namespace sslib
{

#ifdef _WIN32
#else
    const int Path::MaxPathLen;
    const char Path::SeparatorChar;
#endif

Path::Path()
{
}

Path::Path(const char *path_str)
{
	Set(path_str, false, true);
}

Path::~Path()
{
}

void Path::Clear()
{
	drive_.clear();
	dir_.clear();
	name_.clear();
	ext_.clear();
}

bool Path::Set(const string &path_str, bool to_full_path, bool normalize_dir)
{
    return Set(path_str.c_str(), to_full_path, normalize_dir);
}

bool Path::Set(const char *path_str, bool is_to_full_path, bool is_normalize_dir)
{
    string path = path_str != nullptr ? path_str : "";
    bool is_abs = IsAbsolute(path);
	if(is_to_full_path && !is_abs)
    {
        string module_dir = GetModuleDir();
        path = module_dir + path;
        return SetFullPath(path, is_normalize_dir);

		/*char full_path[MaxPathLen];
#ifdef _WIN32
        const char *szRealPath = _fullpath(full_path, path_str, MaxPathLen);
#else
        const char *szRealPath = realpath(path_str, full_path);
#endif
		if(szRealPath != nullptr) {
			return SetFullPath(szRealPath, is_normalize_dir);
		}

		Clear();
		return false;*/
	}
	else
    {
		return SetFullPath(path, is_normalize_dir);
	}
}

bool Path::SetDir(const string &dir, bool to_full_path, bool normalize_path)
{
	string dir_path = dir;
    char first_char = dir.size() > 0 ? dir[dir.size() - 1] : '\0';
	if(first_char != '\\' && first_char != '/') {
		dir_path += SeparatorChar;
	}

    bool ret = Set(dir_path, to_full_path, normalize_path);
    return ret;
}

bool Path::SetName(const char *name)
{
	name_ = name;
	return true;
}

bool Path::SetExt(const char *ext)
{
	ext_ = ext;
	if(ext_.size() > 0 && ext_[0] != '.') {
		ext_ = "." + ext_;
	}
	return true;
}

bool Path::SetNameAndExt(const char *name_and_ext)
{
    string name_ext_str(name_and_ext);
    return SetNameAndExt(name_ext_str);
}

bool Path::SetNameAndExt(const string &name_and_ext)
{
    name_.clear();
    ext_.clear();

    string::size_type offset = name_and_ext.find_last_of("/\\");
    string pure_name_and_ext = offset == string::npos ? name_and_ext : name_and_ext.substr(offset + 1);

    offset = pure_name_and_ext.find_last_of('.');
    if (offset == string::npos)
    {
        name_ = pure_name_and_ext;
        ext_.clear();
    }
    else
    {
        name_ = pure_name_and_ext.substr(0, offset);
        ext_ = pure_name_and_ext.substr(offset);
    }

    return true;
}

string Path::GetPath() const
{
	return dir_ + name_ + ext_;
}

string Path::GetDrive() const
{
	return drive_;
}

string Path::GetDir() const
{
	return dir_;
}

string Path::GetDirNoEndSlash() const
{
    string dir = dir_;
    if (dir.size() > 0 && (dir[dir.size() - 1] == '\\' || dir[dir.size() - 1] == '/')) {
        dir.resize(dir.size() - 1);
    }

    return dir;
}

string Path::GetNameAndExt() const
{
	return name_ + ext_;
}

string Path::GetName() const
{
	return name_;
}

string Path::GetExt() const
{
	return ext_;
}

void Path::GetModulePath(bool is_calling_app)
{
    Clear();
#ifdef _WIN32
    char full_path[MaxPathLen];
    HMODULE module_handle = is_calling_app ? nullptr : (HINSTANCE)&__ImageBase;//_Module.GetModuleInstance();
    if (::GetModuleFileNameA(module_handle, full_path, MaxPathLen) != 0)
    {
        this->Set(full_path, true);
    }
#else
    (void)is_calling_app;
    char buf[MaxPathLen];
    const char *dir_name = getcwd(buf, MaxPathLen);
    if (dir_name != nullptr) {
        SetDir(dir_name, true, true);
    }
    //boost::filesystem::path the_path = boost::filesystem::initial_path();
    //Set(the_path.string().c_str(), true);
#endif
}

//static
bool  Path::GetFileStat(FileStat &fs, const string &file_path)
{
    fs.Clear();

#ifdef _WIN32
    struct __stat64 st;
    int ret = _stat64(file_path.c_str(), &st);
    if (ret != 0) {
        return false;
    }

    fs.path = file_path;
    fs.type = (st.st_mode & _S_IFREG) != 0 ? FileTypeEnum::Regular
        : ((st.st_mode & _S_IFDIR) != 0 ? FileTypeEnum::Dir : FileTypeEnum::Other);
#else
    struct stat st;
    int ret = stat(file_path.c_str(), &st);
    if (ret != 0) {
        return false;
    }

    fs.path = file_path;
    fs.type = S_ISREG(st.st_mode) ? FileTypeEnum::Regular
        : (S_ISDIR(st.st_mode) ? FileTypeEnum::Dir : FileTypeEnum::Other);
#endif

    fs.size = st.st_size;
    fs.atime = st.st_atime;
    fs.mtime = st.st_mtime;
    fs.ctime = st.st_ctime;

    return true;
}

//static
bool Path::FileExists(const string &path)
{
    return FileExists(path.c_str());
}

//static
bool Path::FileExists(const char *file_name)
{
    Path the_path;
    bool ret = the_path.Set(file_name, true, true);
    if (!ret) {
        return false;
    }

    string path_str = the_path.GetPath();
    FileStat fs;
    ret = Path::GetFileStat(fs, path_str);
    return ret && fs.type == FileTypeEnum::Regular;
}

//static
bool Path::PathExists(const string &path)
{
    return PathExists(path.c_str());
}

//static
bool Path::PathExists(const char *path_name)
{
    Path the_path;
    bool ret = the_path.Set(path_name, true, true);
    if (!ret) {
        return false;
    }

    string path_str = the_path.GetPath();
    FileStat fs;
    ret = GetFileStat(fs, path_str);
    return ret;
}

//static
bool Path::DirExists(const string &path)
{
    return DirExists(path.c_str());
}

//static
bool Path::DirExists(const char *dir_str)
{
    Path the_path;
    bool ret = the_path.Set(dir_str, true, true);
    if (!ret) {
        return false;
    }

    string path_str = the_path.GetPath();
    FileStat fs;
    ret = GetFileStat(fs, path_str);
    return ret && fs.type == FileTypeEnum::Dir;
}

//static
bool Path::Mkdir(const char *dir_str)
{
#ifdef _WIN32
	return ::CreateDirectoryA(dir_str, nullptr) ? true : false;
#else
    return mkdir(dir_str, 0777) == 0;
#endif
}

//static
bool Path::Mkdir(const string &dir)
{
    return Mkdir(dir.c_str());
}

//static
bool Path::Mkdirs(const char *dir_sz)
{
	bool ret = true;
	string dir_str = dir_sz;

	string::size_type offset = 2;
	string::size_type pos = 0;
    uint32_t sep_num = 0;

    bool is_continue = true;
	while(is_continue)
	{
		pos = dir_str.find_first_of("\\/", offset);

		if(pos == string::npos)
		{
			if(dir_str.size() > offset)
			{
				const char *csTemp = dir_str.c_str();
				if(!DirExists(csTemp))
					ret = Mkdir(csTemp);
			}
			break;
		}

		if(pos > offset && sep_num > 0)
		{
			string sub_str = dir_str.substr(0, pos);
			if(!DirExists(sub_str.c_str()))
				ret = Mkdir(sub_str.c_str());
		}
        sep_num++;
		offset = pos + 1;
	}

	return ret;
}

bool Path::Mkdirs(const string &dir)
{
    return Mkdirs(dir.c_str());
}

//static
bool Path::Move(const char *old_path, const char *new_path)
{
#ifdef _WIN32
    return !!MoveFileExA(old_path, new_path, MOVEFILE_REPLACE_EXISTING);
#else
    return rename(old_path, new_path) == 0;
#endif
}

//static
bool Path::Delete(const char *file_path)
{
#ifdef _WIN32
    return ::DeleteFileA(file_path) == TRUE;
#else
    return unlink(file_path) == 0;
#endif
}

//static
uint64_t Path::GetFileLength(const string &file_path)
{
    uint64_t file_len = 0;
    bool ret = GetFileLength(file_path.c_str(), file_len);
    return ret ? file_len : UINT64_MAX;
}

//static
bool Path::GetFileLength(const string &file_path, uint64_t &file_len)
{
    bool ret = GetFileLength(file_path.c_str(), file_len);
    return ret;
}

//static
bool Path::GetFileLength(const char *file_path, uint64_t &file_len)
{
    FileStat fs;
    bool ret = GetFileStat(fs, file_path);
    file_len = fs.size;
    return ret;
}

#ifdef _WIN32
//static
bool Path::CopyDir(const char *src_dir_str, const char *target_dir_str,
    const char *filter_str, bool is_recursive)
{
    bool ret = true;
    File src_dir, dst_dir;
    src_dir.SetDir(src_dir_str, true, true);
    dst_dir.SetDir(target_dir_str, true, true);

    WIN32_FIND_DATAA find_file_data;
    HANDLE find_handle;
    string src_file_str, dst_file_str;

    string src_filter_str = src_dir.GetDir() + (filter_str != nullptr ? filter_str : "*.*");
    find_handle = FindFirstFileA(src_filter_str.c_str(), &find_file_data);
    if (find_handle == INVALID_HANDLE_VALUE)
    {
        DWORD error_code = GetLastError();
        if (error_code == ERROR_FILE_NOT_FOUND/* || error_code == ERROR_PATH_NOT_FOUND*/) {
            return true;
        }
        return false;
    }
    else
    {
        while (ret)
        {
            bool is_dir = (find_file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
            if (strcmp(find_file_data.cFileName, ".") != 0 && strcmp(find_file_data.cFileName, "..") != 0)
            {
                src_file_str = src_dir.GetDir() + find_file_data.cFileName;
                dst_file_str = dst_dir.GetDir() + find_file_data.cFileName;

                if (is_dir && is_recursive)
                {
                    bool temp_ret = CopyDir(src_file_str.c_str(),
                        dst_file_str.c_str(), filter_str, is_recursive);
                    if (!temp_ret) {
                        ret = false;
                    }
                }
                else
                {
                    BOOL temp_ret = ::CopyFileA(src_file_str.c_str(),
                        dst_file_str.c_str(), FALSE);
                    if (!temp_ret) {
                        ret = false;
                    }
                }
            }

            BOOL have_more = FindNextFileA(find_handle, &find_file_data);
            if (!have_more)
            {
                DWORD error_code = GetLastError();
                if (error_code != ERROR_NO_MORE_FILES) {
                    ret = false;
                }
                break;
            }
        }

        FindClose(find_handle);
    }

    return ret;
}
#endif

//static
bool Path::GetFileNames(vector<string> &file_names, const char *dir_sz,
    const char *filter_str, bool is_full_path)
{
    ListDirOptions opt;
    opt.res_with_full_path = is_full_path;
    opt.filter = String::IsNullOrEmpty(filter_str) ? "" : filter_str;
    string dir = dir_sz, prefix_dir = "";
    return ListDirInner(file_names, dir, prefix_dir, &opt);
}

//static
bool Path::ListDir(vector<string> &file_names, const char *dir_sz,
    const ListDirOptions *options)
{
    file_names.clear();
    string dir = dir_sz, prefix_dir = "";
    return ListDirInner(file_names, dir, prefix_dir, options);
}

//static
bool Path::ListDir(vector<string> &file_names, const string &dir,
    const ListDirOptions *options)
{
    file_names.clear();
    return ListDirInner(file_names, dir.c_str(), "", options);
}

//static
string Path::GetLaunchedDir()
{
    return GetModuleDir(true);
}

//static
string Path::GetModuleDir(bool is_calling_app)
{
    File the_path;
    the_path.GetModulePath(is_calling_app);
    return the_path.GetDir();
}

bool Path::GetFileContent_Text(string &content, const string &file_str)
{
    content.clear();

    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

	string line_str;
	while(reader.GetLine(line_str))
	{
		content += line_str;
        content += "\n";
		//content += "\r\n";
	}

    reader.Close();
	return ret;
}

bool Path::GetFileContent_Text(wstring &content, const string &file_str)
{
    content.clear();

    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

    wstring line_str;
    while (reader.GetLine(line_str))
    {
        content += line_str;
        content += L"\n";
        //content += L"\r\n";
    }

    reader.Close();
    return ret;
}

//static
bool Path::GetFileContent_Text(vector<string> &lines, const string &file_str)
{
    lines.clear();
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

    uint32_t line_len = 0;
    string line_str;
    while (reader.GetLine(line_str))
    {
        line_len = (uint32_t)line_str.size();
        if (line_len > 0 && '\r' == line_str[line_len-1]) {
            line_str.resize(line_len - 1);
        }
        lines.push_back(line_str);
    }

    reader.Close();
    return ret;
}

//static
bool Path::GetFileContent_Text(vector<wstring> &lines, const string &file_str)
{
    lines.clear();
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

    uint32_t line_len = 0;
    string line_str;
    while (reader.GetLine(line_str))
    {
        line_len = (uint32_t)line_str.size();
        if (line_len > 0 && '\r' == line_str[line_len - 1]) {
            line_str.resize(line_len - 1);
        }
        lines.push_back(StringUtil::Utf8ToWideStr(line_str));
    }

    reader.Close();
    return ret;
}

//static
bool Path::GetFileContent_Text(map<string, int> &lines,
    const string &file_str, int score_column_idx)
{
    lines.clear();
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

    uint32_t line_len = 0;
    string line_str;
    string delims = "\t";
    vector<string> tokens;
    int score = 0;
    while (reader.GetLine(line_str))
    {
        line_len = (uint32_t)line_str.size();
        if (line_len > 0 && '\r' == line_str[line_len - 1]) {
            line_str.resize(line_len - 1);
        }

        score = 1;
        if (score_column_idx > 0)
        {
            String::Split(line_str, tokens, delims);
            if (score_column_idx < (int)tokens.size()) {
                line_str = tokens[0];
                score = String::ToInt32(tokens[score_column_idx]);
            }
        }

        auto iter = lines.find(line_str);
        if (iter != lines.end()) {
            iter->second += score;
        }
        else {
            lines[line_str] = score;
        }
    }

    reader.Close();
    return ret;
}

//static
bool Path::GetFileContent_Text(map<wstring, int> &lines,
    const string &file_str, int score_column_idx)
{
    lines.clear();
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

    uint32_t line_len = 0;
    wstring line_wstr;
    string line_str;
    wstring delims = L"\t";
    vector<wstring> tokens;
    int score = 0;
    while (reader.GetLine(line_str))
    {
        line_len = (uint32_t)line_str.size();
        if (line_len > 0 && '\r' == line_str[line_len - 1]) {
            line_str.resize(line_len - 1);
        }

        line_wstr = StringUtil::Utf8ToWideStr(line_str);
        score = 1;
        if (score_column_idx > 0)
        {
            WString::Split(line_wstr, tokens, delims);
            if (score_column_idx < (int)tokens.size())
            {
                line_wstr = tokens[0];
                score = String::ToInt32(tokens[score_column_idx]);
            }
        }

        auto iter = lines.find(line_wstr);
        if (iter != lines.end()) {
            iter->second += score;
        }
        else {
            lines[line_wstr] = score;
        }
    }

    reader.Close();
    return ret;
}

//static
bool Path::GetFileContent_Text(map<string, int, StrLessNoCase> &lines,
    const string &file_str, int score_column_idx)
{
    lines.clear();
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

    uint32_t line_len = 0;
    string line_str;
    string delims = "\t";
    vector<string> tokens;
    int score = 0;
    while (reader.GetLine(line_str))
    {
        line_len = (uint32_t)line_str.size();
        if (line_len > 0 && '\r' == line_str[line_len - 1]) {
            line_str.resize(line_len - 1);
        }

        score = 1;
        if (score_column_idx > 0)
        {
            String::Split(line_str, tokens, delims);
            if (score_column_idx < (int)tokens.size())
            {
                line_str = tokens[0];
                score = String::ToInt32(tokens[score_column_idx]);
            }
        }

        auto iter = lines.find(line_str);
        if (iter != lines.end()) {
            iter->second += score;
        }
        else {
            lines[line_str] = score;
        }
    }

    reader.Close();
    return ret;
}

//static
bool Path::GetFileContent_Text(map<wstring, int, WStrLessNoCase> &lines,
    const string &file_str, int score_column_idx)
{
    lines.clear();
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str);
    Macro_RetFalseIf(!ret);

    uint32_t line_len = 0;
    wstring line_wstr;
    string line_str;
    wstring delims = L"\t";
    vector<wstring> tokens;
    int score = 0;
    while (reader.GetLine(line_str))
    {
        line_len = (uint32_t)line_str.size();
        if (line_len > 0 && '\r' == line_str[line_len - 1]) {
            line_str.resize(line_len - 1);
        }

        line_wstr = StringUtil::Utf8ToWideStr(line_str);
        score = 1;
        if (score_column_idx > 0)
        {
            WString::Split(line_wstr, tokens, delims);
            if (score_column_idx < (int)tokens.size()) {
                line_wstr = tokens[0];
                score = String::ToInt32(tokens[score_column_idx]);
            }
        }

        auto iter = lines.find(line_wstr);
        if (iter != lines.end()) {
            iter->second += score;
        }
        else {
            lines[line_wstr] = score;
        }
    }

    reader.Close();
    return ret;
}

bool Path::GetFileContent_Binary(string &content, const string &file_str)
{
    content.clear();

    BinaryFileStream reader;
    bool ret = reader.OpenForRead(file_str, false);
    Macro_RetFalseIf(!ret);

    uint64_t len = reader.GetFileLength();
    uint64_t buf_len = min((uint64_t)256 * 1024, len);
    char *buf = buf_len > 0 ? new char[(size_t)buf_len] : nullptr;
    while (ret && len > 0)
    {
        int to_read_len = (int)min(buf_len, len);
        ret = reader.Read(buf, to_read_len);
        if (ret)
        {
            content.append(buf, to_read_len);
            len -= to_read_len;
        }
    }

    if (buf != nullptr) {
        delete[] buf;
    }
    reader.Close();
    return ret;
}

//static
bool Path::SetFileContent(const string &file_str, const string &content)
{
    BinaryFileStream writer;
    bool ret = writer.OpenForWrite(file_str);
    Macro_RetFalseIf(!ret);

    ret = writer.Write(content.c_str(), content.size());
    ret = ret && writer.Flush();
    writer.Close();
    return ret;
}

//static
bool Path::SetFileContent(const string &file_str,
    const vector<string> &lines, const char *line_end)
{
    if (line_end == nullptr) {
        line_end = "\r\n";
    }

    BinaryFileStream writer;
    bool ret = writer.OpenForWrite(file_str);
    Macro_RetFalseIf(!ret);

    for (size_t line_idx = 0; line_idx < lines.size(); line_idx++)
    {
        ret = writer.Write(lines[line_idx].c_str(), lines[line_idx].size());
        if (line_idx + 1 < lines.size()) {
            writer.Write(line_end, strlen(line_end));
        }
    }

    ret = ret && writer.Flush();
    writer.Close();
    return ret;
}

//static
bool Path::SetFileContent(const string &file_str,
    const vector<wstring> &lines, const char *line_end)
{
    if (line_end == nullptr) {
        line_end = "\r\n";
    }

    BinaryFileStream writer;
    bool ret = writer.OpenForWrite(file_str);
    Macro_RetFalseIf(!ret);

    string line_str;
    for (size_t line_idx = 0; line_idx < lines.size(); line_idx++)
    {
        line_str = StringUtil::ToUtf8(lines[line_idx]);
        ret = writer.Write(line_str.c_str(), line_str.size());
        if (line_idx + 1 < lines.size()) {
            writer.Write(line_end, strlen(line_end));
        }
    }

    ret = ret && writer.Flush();
    writer.Close();
    return ret;
}

////////////////////////////////////////////////////////////////
// Protected

bool Path::SetFullPath(const string &fullPath, bool is_normalize_dir)
{
    Clear();
    if (fullPath.length() > MaxPathLen) {
        return false;
    }

#ifdef _WIN32
    char drive_str[MaxPathLen];
    char dir_str[MaxPathLen];
    char name_buf[MaxPathLen];
    char ext_buf[MaxPathLen];

    _splitpath_s(fullPath.c_str(), drive_str, dir_str, name_buf, ext_buf);
    drive_ = drive_str;
    dir_ = drive_ + dir_str;
    if (is_normalize_dir) {
        dir_ = NormalizeDir(dir_);
    }
    name_ = name_buf;
    ext_ = ext_buf;
#else
    (void)is_normalize_dir;
    drive_.clear();
    string::size_type pos = fullPath.find_last_of(SeparatorChar);
    if (pos != string::npos)
    {
        dir_ = fullPath.substr(0, pos + 1);
        SetNameAndExt(fullPath.substr(pos + 1));
    }
    else
    {
        dir_.clear();
        SetNameAndExt(fullPath);
    }
#endif //_WIN32

    return true;
}

//static
std::string Path::NormalizeDir(const string &dir_str)
{
    //todo: "/"
    size_t min_dir_len = 2; //For example: "d:", "\\host"

    string norm_dir_str, cur_seg_str;
    size_t len = dir_str.size();
    uint32_t seg_num = 0;
    for (size_t ch_idx = 0; ch_idx <= len; ch_idx++)
    {
        char ch = ch_idx < len ? dir_str[ch_idx] : SeparatorChar;
        if (ch_idx < len && (ch_idx < min_dir_len || (ch != '\\' && ch != '/')))
        {
            cur_seg_str += ch;
        }
        else
        {
            if (cur_seg_str.empty())
            {
                //do nothing
            }
            else if (cur_seg_str == ".")
            {
                //do nothing
            }
            else if (cur_seg_str == "..")
            {
                if (seg_num > 1 && norm_dir_str.size() > min_dir_len)
                {
                    size_t pos = norm_dir_str.size() - 1 - 1;
                    for (; pos >= min_dir_len; pos--)
                    {
                        char temp_ch = norm_dir_str[pos];
                        if (temp_ch == '\\' || temp_ch == '/')
                        {
                            norm_dir_str.resize(pos + 1);
                            seg_num--;
                            break;
                        }
                    }
                }
            }
            else
            {
                norm_dir_str += cur_seg_str;
                norm_dir_str += ch;
                seg_num++;
            }

            cur_seg_str.clear();
        }
    }

    if (!cur_seg_str.empty())
    { //for "D"
        norm_dir_str += cur_seg_str;
        norm_dir_str += SeparatorChar;
    }

    return norm_dir_str;
}

bool Path::IsAbsolute(const string &path)
{
    if (path.empty()) {
        return false;
    }

    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }

    if (path.size() > 1 && path[1] == ':') {
        return true;
    }

    return false;
}

//static
bool Path::ListDirInner(vector<string> &file_list,
    const string &dir, const string &prefix_dir,
    const ListDirOptions *options)
{
    vector<FileInfo> file_info_list;
    bool ret = ListDirInner(file_info_list, dir, prefix_dir, options);

    file_list.clear();
    for (const FileInfo &file_info : file_info_list) {
        file_list.push_back(file_info.path);
    }

    return ret;
}

#ifdef _WIN32
//static
bool Path::ListDirInner(vector<FileInfo> &file_names, const string &dir,
    const string &prefix_dir, const ListDirOptions *options)
{
    bool ret = true;
    ListDirOptions list_options;
    const ListDirOptions *options_ptr = options != nullptr ? options : &list_options;

    File dir_path;
    dir_path.SetDir(dir, true, true);

	WIN32_FIND_DATAA find_file_data;
	HANDLE find_handle = INVALID_HANDLE_VALUE;

    string src_filter_str = dir_path.GetDir() + prefix_dir
        + (!options_ptr->filter.empty() ? options_ptr->filter : "*.*");
	find_handle = FindFirstFileA(src_filter_str.c_str(), &find_file_data);
	if(find_handle == INVALID_HANDLE_VALUE)
    {
		DWORD error_code = GetLastError();
		if(error_code == ERROR_FILE_NOT_FOUND/* || error_code == ERROR_PATH_NOT_FOUND*/) {
			return true;
		}
		return false;
	}
	else
    {
		while(ret)
		{
			bool is_dir = (find_file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
			if(strcmp(find_file_data.cFileName, ".") != 0 && strcmp(find_file_data.cFileName, "..") != 0)
            {
                FileInfo file_info;
                if( (is_dir && options_ptr->including_dirs) ||
                    (!is_dir && options_ptr->including_regular_files) )
                {
                    file_info.type = is_dir ? FileTypeEnum::Dir : FileTypeEnum::Regular;
                    if(options_ptr->res_with_full_path) {
                        file_info.path = dir_path.GetDir() + prefix_dir + find_file_data.cFileName;
                    }
                    else if(options_ptr->res_with_prefix_dir) {
                        file_info.path = prefix_dir + find_file_data.cFileName;
                    }
                    else {
                        file_info.path = find_file_data.cFileName;
                    }

                    file_names.push_back(file_info);
			    }

                if(is_dir && options_ptr->is_recursive)
                {
                    string new_prefix_dir = prefix_dir + find_file_data.cFileName + SeparatorChar;
                    Path::ListDirInner(file_names, dir_path.GetDir(), new_prefix_dir, options_ptr);
                }
            }

			BOOL have_more = FindNextFileA(find_handle, &find_file_data);
			if(!have_more)
            {
				DWORD error_code = GetLastError();
				if(error_code != ERROR_NO_MORE_FILES) {
					ret = false;
				}
				break;
			}
		}

        FindClose(find_handle);
	}

	return ret;
}
#endif //def _WIN32

#ifndef _WIN32
//static
bool Path::ListDirInner(vector<FileInfo> &file_list, const string &dir,
    const string &prefix_dir, const ListDirOptions *options)
{
    bool ret = true;
    ListDirOptions list_options;
    const ListDirOptions *options_ptr = options != nullptr ? options : &list_options;
    string filter_str = options_ptr != nullptr ? options_ptr->filter : "";

    File dir_path;
    dir_path.SetDir(dir, true, true);

    string src_dir = dir_path.GetDir() + prefix_dir;
    if (options_ptr->enable_log) {
        LogKeyInfo("the dir: %s", src_dir.c_str());
    }

    struct dirent *dir_entry = nullptr;
    DIR *dir_ptr = opendir(src_dir.c_str());
    if (dir_ptr == nullptr)
    {
        if (options_ptr->enable_log) {
            LogWarning("Failed to open the dir");
        }
        return false;
    }

    while ((dir_entry = readdir(dir_ptr)) != nullptr)
    {
        if (strcmp(dir_entry->d_name, ".") == 0 || strcmp(dir_entry->d_name, "..") == 0) {
            continue;
        }

        FileInfo src_file;
        bool is_dir = dir_entry->d_type == DT_DIR;
        bool is_regular = dir_entry->d_type == DT_REG;
        //bool is_link_file = dir_entry->d_type == DT_LNK;
        //According to Linux Programmer's Manual:
        // "Currently, only some filesystems (among them: Btrfs, ext2, ext3, and ext4) 
        //  have full support for returning the file type in d_type. All applications 
        //  must properly handle a return of DT_UNKNOWN"
        bool is_unknown_file = dir_entry->d_type == DT_UNKNOWN;

        if (options_ptr->enable_log)
        {
            LogKeyInfo("file-name: %s; is_dir: %s; is_regular: %s; is_unknown: %s",
                dir_entry->d_name,
                (is_dir ? "true" : "false"),
                (is_regular ? "true" : "false"),
                (is_unknown_file ? "true" : "false"));
        }

        if ((is_dir && options_ptr->including_dirs) ||
            ((is_regular || is_unknown_file) && options_ptr->including_regular_files))
        {
            src_file.type = is_dir ? FileTypeEnum::Dir : FileTypeEnum::Regular;
            if (options_ptr->res_with_full_path) {
                src_file.path = src_dir + dir_entry->d_name;
            }
            else if (options_ptr->res_with_prefix_dir) {
                src_file.path = prefix_dir + dir_entry->d_name;
            }
            else {
                src_file.path = dir_entry->d_name;
            }

            bool is_acceptable = true;
            if (!filter_str.empty())
            {
                is_acceptable = StringUtil::WildcardMatch(src_file.path, filter_str);
            }

            if (is_acceptable) {
                file_list.push_back(src_file);
            }
        }

        if (is_dir && options_ptr->is_recursive)
        {
            string new_prefix_dir = prefix_dir + dir_entry->d_name + SeparatorChar;
            Path::ListDirInner(file_list, dir_path.GetDir().c_str(), new_prefix_dir, options_ptr);
        }
    }

    closedir(dir_ptr);
    return ret;
}
#endif //ndef _WIN32

} //end of namespace
