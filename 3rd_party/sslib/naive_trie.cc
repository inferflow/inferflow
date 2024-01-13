#include "naive_trie.h"

namespace sslib
{

NaiveTrie::NaiveTrie()
{
}

NaiveTrie::~NaiveTrie()
{
    Clear();
}

void NaiveTrie::Clear()
{
    link_map_.Clear();
    node_list_.clear();
}

uint32_t NaiveTrie::Add(const wstring &str)
{
    return AddEx(str, UINT32_MAX, false);
}

uint32_t NaiveTrie::Add(const string &str)
{
    return AddEx(str, UINT32_MAX, false);
}

uint32_t NaiveTrie::AddEx(const wstring &str,
    uint32_t value, bool is_replacing)
{
    if (value == UINT32_MAX)
    {
        if (new_value_ == UINT32_MAX) {
            return UINT32_MAX;
        }

        value = new_value_;
        new_value_++;
    }

    uint32_t len = (uint32_t)str.size();
    Macro_RetIf(UINT32_MAX, len <= 0);

    uint32_t node_value = UINT32_MAX;
    LinkKey key;
    key.node_id = UINT32_MAX;
    for (uint32_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        if (ch_idx + 1 == len) {
            node_value = value;
        }

        key.ch = (char16_t)str[ch_idx];
        auto iter_find = link_map_.Find(key);
        if (iter_find.IsEnd())
        {
            uint32_t new_node_id = (uint32_t)node_list_.size();
            node_list_.push_back(node_value);
            link_map_.Insert(key, new_node_id);
            key.node_id = new_node_id;
        }
        else
        {
            key.node_id = iter_find.Value();
            if (ch_idx + 1 == len)
            {
                if (is_replacing || node_list_[key.node_id] == UINT32_MAX) {
                    node_list_[key.node_id] = value;
                }
                else {
                    node_value = node_list_[key.node_id];
                }
            }
        }
    }

    return node_value;
}

uint32_t NaiveTrie::AddEx(const string &str,
    uint32_t value, bool is_replacing)
{
    if (value == UINT32_MAX)
    {
        if (new_value_ == UINT32_MAX) {
            return UINT32_MAX;
        }

        value = new_value_;
        new_value_++;
    }

    uint32_t len = (uint32_t)str.size();
    Macro_RetIf(UINT32_MAX, len <= 0);

    uint32_t node_value = UINT32_MAX;
    LinkKey key;
    key.node_id = UINT32_MAX;
    for (uint32_t ch_idx = 0; ch_idx < len; ch_idx++)
    {
        if (ch_idx + 1 == len) {
            node_value = value;
        }

        key.ch = (char16_t)str[ch_idx];
        auto iter_find = link_map_.Find(key);
        if (iter_find.IsEnd())
        {
            uint32_t new_node_id = (uint32_t)node_list_.size();
            node_list_.push_back(node_value);
            link_map_.Insert(key, new_node_id);
            key.node_id = new_node_id;
        }
        else
        {
            key.node_id = iter_find.Value();
            if (ch_idx + 1 == len)
            {
                if (is_replacing || node_list_[key.node_id] == UINT32_MAX) {
                    node_list_[key.node_id] = value;
                }
                else {
                    node_value = node_list_[key.node_id];
                }
            }
        }
    }

    return node_value;
}

NaiveTrie::SearchResult NaiveTrie::PrefixSearch(const wstring &prefix) const
{
    return PrefixSearch(prefix.c_str(), (uint32_t)prefix.size());
}

NaiveTrie::SearchResult NaiveTrie::PrefixSearch(const string &prefix) const
{
    return PrefixSearch(prefix.c_str(), (uint32_t)prefix.size());
}

NaiveTrie::SearchResult NaiveTrie::PrefixSearch(const wchar_t *prefix,
    uint32_t prefix_len) const
{
    SearchResult res;
    uint32_t node_num = (uint32_t)node_list_.size();
    Macro_RetIf(res, prefix == nullptr || prefix_len <= 0);

    LinkKey key;
    key.node_id = UINT32_MAX;
    for (uint32_t ch_idx = 0; ch_idx < prefix_len; ch_idx++)
    {
        key.ch = (char16_t)prefix[ch_idx];
        auto iter_find = link_map_.Find(key);
        if (iter_find.IsEnd()) {
            return res;
        }

        key.node_id = iter_find.Value();
        if (ch_idx + 1 == prefix_len && key.node_id < node_num)
        {
            res.value = node_list_[key.node_id];
        }
    }

    res.found = true;
    return res;
}

NaiveTrie::SearchResult NaiveTrie::PrefixSearch(const char *prefix,
    uint32_t prefix_len) const
{
    SearchResult res;
    uint32_t node_num = (uint32_t)node_list_.size();
    Macro_RetIf(res, prefix == nullptr || prefix_len <= 0);

    LinkKey key;
    key.node_id = UINT32_MAX;
    for (uint32_t ch_idx = 0; ch_idx < prefix_len; ch_idx++)
    {
        key.ch = (char16_t)prefix[ch_idx];
        auto iter_find = link_map_.Find(key);
        if (iter_find.IsEnd()) {
            return res;
        }

        key.node_id = iter_find.Value();
        if (ch_idx + 1 == prefix_len && key.node_id < node_num)
        {
            res.value = node_list_[key.node_id];
        }
    }

    res.found = true;
    return res;
}

} //end of namespace
