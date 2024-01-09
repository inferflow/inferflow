#pragma once

#include <vector>
#include "prime_types.h"
#include "hashtable.h"

namespace sslib
{

using std::string;
using std::wstring;
using std::vector;

class NaiveTrie
{
public:
    struct SearchResult
    {
        bool found = false;
        uint32_t value = UINT32_MAX;
    };

public:
    NaiveTrie();
    virtual ~NaiveTrie();
    void Clear();

    //return: value (UINT32_MAX means error)
    uint32_t Add(const wstring &str);
    //return: value (UINT32_MAX means error)
    uint32_t AddEx(const wstring &str, uint32_t value, bool is_replacing = true);

    //return: value (UINT32_MAX means error)
    uint32_t Add(const string &str);
    //return: value (UINT32_MAX means error)
    uint32_t AddEx(const string &str, uint32_t value, bool is_replacing = true);

    SearchResult PrefixSearch(const wstring &prefix) const;
    SearchResult PrefixSearch(const wchar_t *prefix, uint32_t prefix_len) const;

    SearchResult PrefixSearch(const string &prefix) const;
    SearchResult PrefixSearch(const char *prefix, uint32_t prefix_len) const;

protected:
#   pragma pack(push, 1)
    struct LinkKey
    {
        uint32_t node_id = 0;
        char16_t ch = (char16_t)0;
    };
#   pragma pack(pop)

    class LinkKeyHashCompare
    {
    public:
        UInt64 operator()(const LinkKey &val) const {
            return val.node_id * 37 + (uint32_t)val.ch;
        }

        int operator()(const LinkKey &lhs, const LinkKey &rhs) const
        {
            if (lhs.node_id == rhs.node_id) {
                return lhs.ch < rhs.ch ? -1 : (lhs.ch == rhs.ch ? 0 : 1);
            }
            return lhs.node_id < rhs.node_id ? -1 : 1;
        }
    };

protected:
    HashMap<LinkKey, uint32_t, LinkKeyHashCompare> link_map_;
    vector<uint32_t> node_list_; //node values
    uint32_t new_value_ = 0;
};

} //end of namespace
