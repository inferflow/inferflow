#pragma once

#include <string>
#include <vector>
#include "prime_types.h"

namespace sslib
{

class SubStrLookupTree
{
public:
    SubStrLookupTree()
    {
        key_ = (wchar_t)0xFFFF;
        int_value_ = UINT32_MAX;
    }
    virtual ~SubStrLookupTree(){
        Clear();
    }

    wchar_t key() const { return key_; }
    std::wstring str_value() const { return str_value_; }
    uint32_t int_value() const { return int_value_; }

    uint32_t ChildCount() const {
        return (uint32_t)child_list_.size();
    }
    bool IsLeaf() const {
        return child_list_.size() <= 0;
    }
    void Clear();
    void AddString(const std::wstring &key, const std::wstring &str_val, uint32_t int_val = 0);
    const SubStrLookupTree *FindString(const std::wstring &str) const;
    const SubStrLookupTree *Find(const wchar_t &wch) const;

protected:
    wchar_t key_ = L'\0';
    std::wstring str_value_;
    uint32_t int_value_ = 0;

    std::vector<SubStrLookupTree*>	child_list_;

protected:
    void AddString(const std::wstring &key, size_t &char_idx, const std::wstring &str_val, uint32_t int_val);
    const SubStrLookupTree *FindString(const std::wstring &str, size_t &char_idx) const;
};

} //end of namespace
