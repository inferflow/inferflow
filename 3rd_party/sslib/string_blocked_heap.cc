#include "string_blocked_heap.h"
#include <cstring>
#include "log.h"

using namespace std;

namespace sslib
{

///////////////////////////////////////////////////////////////////////////////
// class StringBlockedHeap

StringBlockedHeap::StringBlockedHeap(int block_size)
{
	SetBlockSize(block_size);
	cur_block_ = nullptr;
	cur_block_idx_ = -1;
	cur_position_ = 0;
}

StringBlockedHeap::~StringBlockedHeap(void)
{
	int blockSize = (int)blocks_.size();
	for(int idx = 0; idx < blockSize; idx++) {
		delete[] blocks_[idx];
	}
	blocks_.clear();
	cur_block_ = nullptr;
}

void StringBlockedHeap::SetBlockSize(int block_size)
{
	block_size_ = block_size < MinBlockSize ? MinBlockSize : block_size;
}

char* StringBlockedHeap::AddWord(const string &word)
{
    return AddString(word);
}

char* StringBlockedHeap::AddString(const std::string &word)
{
	int size = (int)word.size();
	const char *bytes = word.c_str();

	if(cur_block_ == nullptr || cur_position_ + size >= block_size_) //'>=' should not be '>'
	{
        block_size_ = block_size_ > size ? block_size_ : (size + 1);
		cur_block_ = new char[block_size_];
		blocks_.push_back(cur_block_);
		cur_block_idx_ = (int)blocks_.size() - 1;
		cur_position_ = 0;

        LogDebugInfoD("A new block added");
	}

	memcpy(cur_block_+cur_position_, bytes, size);
	cur_block_[cur_position_+size] = '\0';

	char *ret = cur_block_+cur_position_;
	cur_position_ += (size + 1);
	
	return ret;
}

char* StringBlockedHeap::AddWord(const char *word, int len)
{
    return AddString(word, len);
}

char* StringBlockedHeap::AddString(const char *word, int len)
{
    if (word == nullptr) {
        return nullptr;
    }
    int size = len > 0 ? len : (int)strlen(word);

	if(cur_block_ == nullptr || cur_position_ + size >= block_size_)	//!!! '>' --> '>='
	{
        block_size_ = block_size_ > size ? block_size_ : (size + 1);
		cur_block_ = new char[block_size_];
		blocks_.push_back(cur_block_);
		cur_block_idx_ = (int)blocks_.size() - 1;
		cur_position_ = 0;

        LogDebugInfoD("A new block added");
	}

    memcpy(cur_block_ + cur_position_, word, size);
	cur_block_[cur_position_+size] = '\0';

    char *ret = cur_block_ + cur_position_;
	cur_position_ += (size + 1);
	
	return ret;
}

char* StringBlockedHeap::New(int size)
{
	if(cur_block_ == nullptr || cur_position_ + size >= block_size_) //'>=' should not be '>'
	{
        block_size_ = block_size_ > size ? block_size_ : (size + 1);
		cur_block_ = new char[block_size_];
		blocks_.push_back(cur_block_);
		cur_block_idx_ = (int)blocks_.size() - 1;
		cur_position_ = 0;
	}

	char *ret = cur_block_ + cur_position_;
	cur_position_ += (size + 1);

	return ret;
}

char* StringBlockedHeap::NewInner(int size)
{
    if (cur_block_ == nullptr || cur_position_ + size > block_size_)
    {
        block_size_ = block_size_ > size ? block_size_ : size;
        cur_block_ = new char[block_size_];
        blocks_.push_back(cur_block_);
        cur_block_idx_ = (int)blocks_.size() - 1;
        cur_position_ = 0;
    }

    char *ret = cur_block_ + cur_position_;
    cur_position_ += size;

    return ret;
}

wchar_t* StringBlockedHeap::AddWord(const std::wstring &word)
{
    return AddWord(word.c_str(), (int)word.size());
}

wchar_t* StringBlockedHeap::AddString(const std::wstring &str)
{
    return AddString(str.c_str(), (int)str.size());
}

wchar_t* StringBlockedHeap::AddWord(const wchar_t *word, int len)
{
    return AddString(word, len);
}

wchar_t* StringBlockedHeap::AddString(const wchar_t *str, int len)
{
    if (str == nullptr) {
        return nullptr;
    }

    int size = len > 0 ? len : (int)wcslen(str);
    wchar_t *buf = (wchar_t*)NewInner((size+1) * sizeof(wchar_t));
    memcpy(buf, str, sizeof(wchar_t) * size);
    buf[size] = L'\0';
    return buf;
}

wchar_t* StringBlockedHeap::NewWchar(int size)
{
    wchar_t *buf = (wchar_t*)NewInner(size * sizeof(wchar_t));
    return buf;
}

void StringBlockedHeap::Reset()
{
	//Clear and free memory
	int blockSize = (int)blocks_.size();
	for(int i = 0; i < blockSize; i++) {
		delete[] blocks_[i];
	}
	blocks_.clear();
	cur_block_ = nullptr;

	//Reinitilaize
	cur_block_idx_ = -1;
	cur_position_ = 0;
}

///////////////////////////////////////////////////////////////////////////////
// class BasicStringBlockedHeap

void BasicStringBlockedHeap_ForCompilation()
{
    StringLocation loc;
    u16string str = u"test";

    U16StringHeap heap;
    heap.SetBlockCapacity(512);
    heap.BlockCapacity();
    heap.AddString(u"hello");
    heap.Clear();
    heap.NewString(100);
    heap.AddStringEx(loc, str);
    heap.Save("");
    heap.Load("");

    WStringHeap wstr_heap;
    wstr_heap.SetBlockCapacity(512);
    wstr_heap.BlockCapacity();
    wstr_heap.AddString(L"hello");
    wstr_heap.Clear();
    wstr_heap.NewString(100);
}

} //end of namespace
