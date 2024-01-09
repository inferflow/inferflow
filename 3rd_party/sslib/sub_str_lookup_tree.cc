#include "sub_str_lookup_tree.h"
#include "string_util.h"
#include "log.h"

namespace sslib
{

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// class SubStrLookupTree

void SubStrLookupTree::Clear()
{
	int child_num = (int)child_list_.size();
	SubStrLookupTree *child_ptr = nullptr;
	for(int i = 0; i < child_num; i++)
	{
		child_ptr = child_list_[i];
		child_ptr->Clear();
		delete child_ptr;
	}
	child_list_.clear();
}

void SubStrLookupTree::AddString(const std::wstring &key,
    const std::wstring &str_val, uint32_t int_val)
{
    size_t char_idx = 0;
    if (str_val.size() == 1 && (int_val == 0 || int_val == UINT32_MAX)) {
        int_val = (uint32_t)(uint16_t)str_val[0];
    }

    //LogKeyInfo(L"str: %ls; meaning: %ls; data:%u", str.c_str(), strMeaning.c_str(), iData);
    AddString(key, char_idx, str_val, int_val);
}

const SubStrLookupTree* SubStrLookupTree::FindString(const wstring &str) const
{
    size_t char_idx = 0;
    return FindString(str, char_idx);
}

const SubStrLookupTree* SubStrLookupTree::Find(const wchar_t &wch) const
{
    size_t child_num = child_list_.size();
	for(size_t idx = 0; idx < child_num; idx++)
	{
        if(child_list_[idx]->key() == wch) {
            return child_list_[idx];
        }
	}

	return nullptr;
}

void SubStrLookupTree::AddString(const std::wstring &key, size_t &char_idx,
    const std::wstring &str_val, uint32_t int_val)
{
    if(char_idx >= key.size()) {
        return;
    }
	wchar_t cur_char = key[char_idx];

    //
	SubStrLookupTree *child_ptr = nullptr;
	size_t child_num = child_list_.size();
	for(size_t idx = 0; idx < child_num; idx++)
	{
		SubStrLookupTree *temp_child_ptr = child_list_[idx];
		if(temp_child_ptr->key() == cur_char)
		{
			child_ptr = temp_child_ptr;
			break;
		}
	}

	//create a new sub-tree if we cannot find an appropriate sub-tree
	if(child_ptr == nullptr)
	{
		child_ptr = new SubStrLookupTree;
		child_ptr->key_ = cur_char;
		child_list_.push_back(child_ptr);
	}

	//
    char_idx++;
    if(char_idx < key.size())
    {
        child_ptr->AddString(key, char_idx, str_val, int_val);
    }
    else
    {
        child_ptr->str_value_ = str_val;
        child_ptr->int_value_ = int_val;
    }
}

const SubStrLookupTree* SubStrLookupTree::FindString(const wstring &str, size_t &char_idx) const
{
    if(char_idx >= str.size()) {
        return nullptr;
    }
	wchar_t cur_char = str[char_idx];

	//
	SubStrLookupTree *child_ptr = nullptr;
	size_t child_num = child_list_.size();
	for(size_t idx = 0; idx < child_num; idx++)
	{
		child_ptr = child_list_[idx];
		if(child_ptr->key_ == cur_char)
        {
            char_idx++;
            if(char_idx < str.size()) {
                return child_ptr->FindString(str, char_idx);
            }
            else {
                return child_ptr;
            }
		}
	}

	return nullptr;
}

} //end of namespace
