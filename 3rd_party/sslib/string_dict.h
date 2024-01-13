#pragma once

#include "prime_types.h"
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "blocked_array.h"
#include "string_blocked_heap.h"
#include "hashtable.h"
#include "hash.h"
#include "log.h"
#include "binary_file_stream.h"
#include "string_util.h"
#include "string.h"
#include "task_monitor.h"

namespace sslib
{

using namespace std;

#pragma pack(push, 1)
template<class WeightType>
struct StrEntry
{
    const char *str;
    WeightType weight;

    StrEntry(const char *s = nullptr, WeightType w = 0) {
        str = s;
        weight = w;
    }

    int Compare(const StrEntry &item) const {
        return strcmp(str, item.str);
    }
    bool operator < (const StrEntry &item) const {
        return strcmp(str, item.str) < 0;
    }
    static bool GreaterWeight(const StrEntry &lhs, const StrEntry &rhs) {
        return lhs.weight > rhs.weight;
    }
};
#pragma pack(pop)

typedef StrEntry<float> StrItem;
typedef BlockedArray<StrItem> StrItemArray;

struct StrDictOpt
{
    enum
    {
        SortBy_Mask     = 0x000F,
        SortById        = 0x0001,
        SortByStr       = 0x0002,
        SortByWeight    = 0x0003,
        SortDesc        = 0x0010,
        Print_Mask      = 0x0F00,
        Print_StrWght   = 0x0000,   //StrWght: string + weight
        Print_IdStrWght = 0x0100,   //IdStrWght: id + string + weight
        Print_IdWghtStr = 0x0200,   //IdWghtStr: id + weight + string
        Print_Str       = 0x0300,   //Str: string only
        Print_IdStr     = 0x0400,   //IdStr: id + string
    };
};

template<class WeightType>
class StringDict
{
public:
    enum PrintOptionsEnum
    {
        PO_SORT_BY_MASK     = StrDictOpt::SortBy_Mask,
        PO_SORT_BY_ID       = StrDictOpt::SortById,
        PO_SORT_BY_STR      = StrDictOpt::SortByStr,
        PO_SORT_BY_FREQ     = StrDictOpt::SortByWeight,
        PO_SORT_DESC        = StrDictOpt::SortDesc
    };

    typedef StrEntry<WeightType> Item;

public:
    explicit StringDict(bool use_location32 = false)
        : item_table_(4096, 1.5f, use_location32)
        , items_(4096)
        , str_heap_(20480)
    {
    }
    explicit StringDict(uint32_t estimated_item_num, bool use_location32 = false)
        : item_table_(estimated_item_num, 1.5f, use_location32)
        , items_(estimated_item_num)
        , str_heap_(10 * estimated_item_num)
    {
    }
    virtual ~StringDict()
    {
        Clear();
    }

    virtual uint32_t Size() const
    {
        return (uint32_t)items_.Size();
    }

    virtual const Item* Get(uint32_t item_id) const
    {
        const Item *item_ptr = item_id < (uint32_t)items_.Size() ? &items_[item_id] : nullptr;
        return item_ptr != nullptr && item_ptr->str != nullptr ? item_ptr : nullptr;
    }

    virtual const Item* Find(const char *item_str, uint32_t &item_id) const;
    virtual Item* Find(const char *item_str, uint32_t &item_id);

    const Item* Find(const char *item_str) const;
    Item* Find(const char *item_str);

    uint32_t TermId(const char *str) const;
    uint32_t TermId(const string &str) const;
    uint32_t ItemId(const char *str) const;
    uint32_t ItemId(const string &str) const;
    const char* ItemStr(uint32_t item_id) const;

    //return value: number of non-null or non-empty results
    //is_early_stop: stop when having a null or empty result
    virtual uint32_t GetItems(vector<const Item*> &items, const vector<uint32_t> &id_list,
        bool is_early_stop = false) const;
    virtual uint32_t GetItems(vector<uint32_t> &item_id_list, const vector<string> &str_list,
        bool is_early_stop = false) const;
    virtual uint32_t GetItems(vector<uint32_t> &item_id_list, const vector<const char*> &str_list,
        bool is_early_stop = false) const;

    virtual void Clear();

    //
    bool AddItem(const std::string &str, const WeightType &weight, bool is_agg_weight = false);
    bool AddItem(const char *str, const WeightType &weight, bool is_agg_weight = false);
    virtual Item* AddItem(const char *str, const WeightType &weight,
        uint32_t &item_id, bool is_agg_weight = false);
    bool AddWeight(const char *str, const WeightType &weight);
    bool SetWeight(uint32_t id, const WeightType &weight);
    bool SetWeight(const char *str, const WeightType &weight);

    virtual Item* AddItem(uint32_t item_id, const char *str, const WeightType &weight, bool is_agg_weight = true);

    virtual bool Store(const std::string &path, bool show_progress = false) const;
    virtual bool Load(const std::string &path, bool is_binary = true, bool show_progress = false);
    virtual bool Print(const std::string &path, uint32_t options, uint32_t nTopK = (uint32_t)-1) const;

    virtual bool Write(IBinStream &strm, bool show_progress = false) const;
    virtual bool Read(IBinStream &strm, bool show_progress = false);
    bool ReadFromPlainText(IBinStream &strm);

#if defined ENABLE_CRYPT_HASH
    void SetEncryptionInfo(const string &spec_str);
    void SetEncryptionInfo(uint32_t len, uint64_t seed);
    void ClearEncryptionInfo() {
        encryption_str_.clear();
    }
#endif //ENABLE_CRYPT_HASH

protected:
    class IdNameHashTraits
    {
    public:
        uint32_t operator()(const IdName<char> &key) const {
            return Hash::HashCodeUInt32(key.name);
        }

        int operator()(const IdName<char>& lhs, const IdName<char>& rhs) const {
            return strcmp(lhs.name, rhs.name);
        }
    };

    typedef Hashtable<IdName<char>, IdNameHashTraits> ItemHashTable;

protected:
    #pragma pack(push, 1)
    struct ItemEx
    {
        uint32_t id;
        const Item *item;

        ItemEx(uint32_t p_id = 0)
        {
            id = p_id;
            item = nullptr;
        }

        static bool StrItemExGreaterFreq(const ItemEx &lhs, const ItemEx &rhs)
        {
            return lhs.item->weight != rhs.item->weight
                ? lhs.item->weight > rhs.item->weight
                : lhs.item->Compare(*rhs.item) < 0;
        }

        static bool StrItemExLessStr(const ItemEx &lhs, const ItemEx &rhs)
        {
            return lhs.item->Compare(*rhs.item) < 0;
        }

    };
    #pragma pack(pop)

protected:
    ItemHashTable item_table_;
    BlockedArray<Item> items_;

    StringBlockedHeap str_heap_;
    string encryption_str_;

protected:
    void PrintItem(ostream &ps, uint32_t item_id, const Item &item, uint32_t format) const;
};

#pragma pack(push, 1)
template<class WeightType>
struct WStrEntry
{
    const wchar_t *str;
    WeightType weight;

    WStrEntry(const wchar_t *s = nullptr, WeightType w = 0) {
        str = s;
        weight = w;
    }

    int Compare(const WStrEntry &item) const {
        return wcscmp(str, item.str);
    }
    bool operator < (const WStrEntry &item) const {
        return wcscmp(str, item.str) < 0;
    }
    static bool GreaterWeight(const WStrEntry &lhs, const WStrEntry &rhs) {
        return lhs.weight > rhs.weight;
    }

    static bool LessStrWeight(const WStrEntry &lhs, const WStrEntry &rhs)
    {
        int iCmp = wcscmp(lhs.str, rhs.str);
        if (iCmp != 0) {
            return iCmp < 0;
        }
        return lhs.weight < rhs.weight;
    }
};
#pragma pack(pop)

typedef WStrEntry<float> WStrItem;
typedef BlockedArray<WStrItem> WStrItemArray;

template<class WeightType>
class WStringDict
{
public:
    enum PrintOptionsEnum
    {
        PO_SORT_BY_MASK     = StrDictOpt::SortBy_Mask,
        PO_SORT_BY_ID       = StrDictOpt::SortById,
        PO_SORT_BY_STR      = StrDictOpt::SortByStr,
        PO_SORT_BY_FREQ     = StrDictOpt::SortByWeight,
        PO_SORT_DESC        = StrDictOpt::SortDesc
    };

public:
    explicit WStringDict(bool use_location32 = false)
        : item_table_(4096, 1.5f, use_location32)
        , items_(4096)
        , str_heap_(20480)
    {
    }
    explicit WStringDict(uint32_t estimated_item_num, bool use_location32 = false)
        : item_table_(estimated_item_num, 1.5f, use_location32)
        , items_(estimated_item_num)
        , str_heap_(5 * estimated_item_num)
    {
    }
    virtual ~WStringDict()
    {
        Clear();
    }

    uint32_t Size() const
    {
        return (uint32_t)items_.Size();
    }
    const WStrEntry<WeightType>* Get(uint32_t item_id) const
    {
        const WStrEntry<WeightType> *item_ptr = item_id < (uint32_t)items_.Size() ? &items_[item_id] : nullptr;
        return item_ptr != nullptr && item_ptr->str != nullptr ? item_ptr : nullptr;
    }

    const WStrEntry<WeightType>* Find(const wchar_t *item_str, uint32_t &item_id) const;
    const WStrEntry<WeightType>* Find(const wchar_t *item_str) const;
    WStrEntry<WeightType>* Find(const wchar_t *item_str, uint32_t &item_id);

    uint32_t TermId(const wchar_t *str) const;
    uint32_t TermId(const wstring &str) const;
    uint32_t ItemId(const wchar_t *str) const;
    uint32_t ItemId(const wstring &str) const;
    const wchar_t* ItemStr(uint32_t item_id) const;

    bool Reinit(bool use_location32 = false);
    void Clear();

    bool AddItem(const std::wstring &itemStr, const WeightType &weight);
    bool AddItem(const wchar_t *item_str, const WeightType &weight);
    const WStrEntry<WeightType>* AddItem(const wchar_t *item_str, const WeightType &weight, uint32_t &item_id);
    const WStrEntry<WeightType>* AddItem(const std::wstring &itemStr, const WeightType &weight, uint32_t &item_id, bool is_agg_weight);
    const WStrEntry<WeightType>* AddItem(const wchar_t *item_str, const WeightType &weight, uint32_t &item_id, bool is_agg_weight);
    const WStrEntry<WeightType>* AddItemMax(const wchar_t *item_str, const WeightType &weight, uint32_t &item_id);

    const WStrEntry<WeightType>* AddItem(uint32_t item_id, const wchar_t *str, const WeightType &weight, bool is_agg_weight = true);

    bool AddWeight(uint32_t item_id, const WeightType &weight);
    bool AddWeight(const wstring &str, const WeightType &weight);
    bool AddWeight(const wchar_t *item_str, const WeightType &weight);
    bool SetWeight(const wchar_t *item_str, const WeightType &weight);
    bool SetWeight(uint32_t item_id, const WeightType &weight);

    bool Store(const std::string &path, bool show_progress = false) const;
    bool Load(const std::string &path, bool is_binary = true, bool show_progress = false);
    bool Print(const std::string &path, uint32_t options) const;

    bool Write(IBinStream &strm, bool show_progress = false) const;
    bool Read(IBinStream &strm, bool show_progress = false);
    bool ReadFromPlainText(IBinStream &strm);

#if defined ENABLE_CRYPT_HASH
    void SetEncryptionInfo(const wstring &spec_str);
    void SetEncryptionInfo(uint32_t len, uint64_t seed);
    void ClearEncryptionInfo() {
        encryption_str_.clear();
    }
#endif //ENABLE_CRYPT_HASH

protected:
    class IdNameHashTraits
    {
    public:
        uint32_t operator()(const IdName<wchar_t> &key) const {
            return Hash::HashCodeUInt32(key.name);
        }

        int operator()(const IdName<wchar_t> &lhs, const IdName<wchar_t> &rhs) const {
            return wcscmp(lhs.name, rhs.name);
        }
    };

    typedef Hashtable<IdName<wchar_t>, IdNameHashTraits> WItemHashTable;

protected:
    #pragma pack(push, 1)
    struct ItemEx
    {
        uint32_t id;
        const WStrEntry<WeightType> *item;

        ItemEx(uint32_t p_id = 0)
        {
            id = p_id;
            item = nullptr;
        }

        static bool WStrItemExGreaterFreq(const ItemEx &lhs, const ItemEx &rhs)
        {
            return lhs.item->weight != rhs.item->weight
                ? lhs.item->weight > rhs.item->weight
                : lhs.item->Compare(*rhs.item) < 0;
        }

        static bool WStrItemExLessStr(const ItemEx &lhs, const ItemEx &rhs)
        {
            return lhs.item->Compare(*rhs.item) < 0;
        }
    };
    #pragma pack(pop)

protected:
    WItemHashTable item_table_;
    BlockedArray<WStrEntry<WeightType>> items_;

    StringBlockedHeap str_heap_;
    string encryption_str_;

protected:
    void PrintItem(std::ostream &ps, uint32_t item_id, const std::string &item_str,
        const WeightType &weight, uint32_t format) const;
};

typedef StringDict<float> StrDict;
typedef WStringDict<float> WStrDict;

////////////////////////////////////////////////////////////////////////////////
// StringDict Implementation
////////////////////////////////////////////////////////////////////////////////

template <class WeightType>
const StrEntry<WeightType>* StringDict<WeightType>::Find(const char *item_str, uint32_t &item_id) const
{
    IdName<char> key(0, item_str);
    auto iter = item_table_.Find(key);
    if(iter.IsEnd()) {
        item_id = UINT32_MAX;
        return nullptr;
    }

    item_id = (*iter).id;
    return &items_[item_id];
}

template <class WeightType>
StrEntry<WeightType>* StringDict<WeightType>::Find(const char *item_str, uint32_t &item_id)
{
    IdName<char> key(0, item_str);
    auto iter = item_table_.Find(key);
    if(iter.IsEnd()) {
        item_id = UINT32_MAX;
        return nullptr;
    }

    item_id = (*iter).id;
    return &items_[item_id];
}

template <class WeightType>
const StrEntry<WeightType>* StringDict<WeightType>::Find(const char *item_str) const
{
    uint32_t item_id = 0;
    return Find(item_str, item_id);
}

template <class WeightType>
StrEntry<WeightType>* StringDict<WeightType>::Find(const char *item_str)
{
    uint32_t item_id = 0;
    return Find(item_str, item_id);
}

template <class WeightType>
uint32_t StringDict<WeightType>::TermId(const char *str) const
{
    uint32_t item_id = UINT32_MAX;
    Find(str, item_id);
    return item_id;
}

template <class WeightType>
uint32_t StringDict<WeightType>::TermId(const string &str) const
{
    return TermId(str.c_str());
}

template <class WeightType>
uint32_t StringDict<WeightType>::ItemId(const char *str) const
{
    uint32_t item_id = UINT32_MAX;
    Find(str, item_id);
    return item_id;
}

template <class WeightType>
uint32_t StringDict<WeightType>::ItemId(const string &str) const
{
    return ItemId(str.c_str());
}

template <class WeightType>
const char* StringDict<WeightType>::ItemStr(uint32_t item_id) const
{
    const auto *item_ptr = Get(item_id);
    return item_ptr != nullptr ? item_ptr->str : nullptr;
}

template <class WeightType>
uint32_t StringDict<WeightType>::GetItems(vector<const Item*> &vecItem,
    const vector<uint32_t> &id_list, bool is_early_stop) const
{
    uint32_t ret = 0;
    vecItem.clear();
    vecItem.reserve(id_list.size());
    for(size_t idx = 0; idx < id_list.size(); idx++)
    {
        const auto *item_ptr = id_list[idx] < (uint32_t)items_.Size() ? &items_[id_list[idx]] : nullptr;
        if(item_ptr != nullptr && item_ptr->str == nullptr) {
            item_ptr = nullptr;
        }
        if(item_ptr != nullptr) {
            ret++;
        }
        else if(is_early_stop) {
            return ret;
        }

        vecItem.push_back(item_ptr);
    }

    return ret;
}

template <class WeightType>
uint32_t StringDict<WeightType>::GetItems(vector<uint32_t> &item_id_list,
    const vector<string> &str_list, bool is_early_stop) const
{
    uint32_t ret = 0;
    item_id_list.clear();
    item_id_list.reserve(str_list.size());
    for(size_t idx = 0; idx < str_list.size(); idx++)
    {
        IdName<char> key(0, str_list[idx].c_str());
        auto iter = item_table_.Find(key);
        uint32_t item_id = iter.IsEnd() ? UINT32_MAX : (*iter).id;

        if(item_id != Number::MaxUInt32) {
            ret++;
        }
        else if(is_early_stop) {
            return ret;
        }
        item_id_list.push_back(item_id);
    }

    return ret;
}

template <class WeightType>
uint32_t StringDict<WeightType>::GetItems(vector<uint32_t> &item_id_list,
    const vector<const char*> &str_list, bool is_early_stop) const
{
    uint32_t ret = 0;
    item_id_list.clear();
    item_id_list.reserve(str_list.size());
    for(size_t idx = 0; idx < str_list.size(); idx++)
    {
        IdName<char> key(0, str_list[idx]);
        auto iter = item_table_.Find(key);
        uint32_t item_id = iter.IsEnd() ? UINT32_MAX : (*iter).id;

        if(item_id != Number::MaxUInt32) {
            ret++;
        }
        else if(is_early_stop) {
            return ret;
        }
        item_id_list.push_back(item_id);
    }

    return ret;
}

template <class WeightType>
void StringDict<WeightType>::Clear()
{
    items_.Clear();
    item_table_.Clear();
    str_heap_.Reset();
}

template <class WeightType>
bool StringDict<WeightType>::AddItem(const std::string &str, const WeightType &weight, bool is_agg_weight)
{
    uint32_t item_id = 0;
    const StrEntry<WeightType> *item_ptr = AddItem(str.c_str(), weight, item_id, is_agg_weight);
    return item_ptr != nullptr;
}

template <class WeightType>
bool StringDict<WeightType>::AddItem(const char *str, const WeightType &weight, bool is_agg_weight)
{
    uint32_t item_id = 0;
    const StrEntry<WeightType> *item_ptr = AddItem(str, weight, item_id, is_agg_weight);
    return item_ptr != nullptr;
}

template <class WeightType>
StrEntry<WeightType>* StringDict<WeightType>::AddItem(const char *str, const WeightType &weight, uint32_t &item_id, bool is_agg_weight)
{
    StrEntry<WeightType> *found_item = Find(str, item_id);
    if(found_item != nullptr)
    {
        if(is_agg_weight) {
            found_item->weight += weight;
        }
        return found_item;
    }

    StrEntry<WeightType> *new_item_ptr = items_.New();
    new_item_ptr->str = str_heap_.AddWord(str);
    new_item_ptr->weight = weight;
    item_id = (uint32_t)items_.Size() - 1;

    IdName<char> pair;
    pair.id = item_id;
    pair.name = new_item_ptr->str;
    item_table_.Insert(pair);

    return new_item_ptr;
}

template <class WeightType>
bool StringDict<WeightType>::AddWeight(const char *str, const WeightType &weight)
{
    uint32_t item_id = 0;
    const StrEntry<WeightType> *item_ptr = AddItem(str, weight, item_id, true);
    return item_ptr != nullptr;
}

template <class WeightType>
bool StringDict<WeightType>::SetWeight(uint32_t item_id, const WeightType &weight)
{
    if(item_id < (uint32_t)items_.Size())
    {
        items_[item_id].weight = weight;
        return true;
    }
    return false;
}

template <class WeightType>
bool StringDict<WeightType>::SetWeight(const char *str, const WeightType &weight)
{
    IdName<char> key(0, str);
    auto iter = item_table_.Find(key);
    if(!iter.IsEnd())
    {
        items_[(*iter).id].weight = weight;
        return true;
    }

    return false;
}

template <class WeightType>
StrEntry<WeightType>* StringDict<WeightType>::AddItem(uint32_t item_id, const char *str,
    const WeightType &weight, bool is_agg_weight)
{
    uint32_t old_item_id = UINT32_MAX;
    IdName<char> key(0, str);
    auto iter = item_table_.Find(key);
    StrEntry<WeightType> *found_item = nullptr;
    if(iter.IsEnd()) {
        old_item_id = UINT32_MAX;
    }
    else {
        old_item_id = (*iter).id;
        found_item = &items_[item_id];
    }

    if(found_item != nullptr)
    {
        if(old_item_id != item_id) {
            LogWarning("This item has been assigned with ID %u (now %u)", old_item_id, item_id);
            return nullptr;
        }

        if(is_agg_weight) {
            found_item->weight += weight;
        }

        return found_item;
    }

    uint32_t item_num_now = (uint32_t)items_.size();
    StrEntry<WeightType> *new_item_ptr = nullptr;
    if(item_id < item_num_now)
    {
        new_item_ptr = &items_[item_id];
        if(new_item_ptr->str != nullptr) {
            LogError("Something wrong with the string dictionary");
            return nullptr;
        }

        new_item_ptr->str = str_heap_.AddWord(str);
        new_item_ptr->weight = weight;
    }
    else
    {
        for(uint32_t idx = item_num_now; idx < item_id; idx++)
        {
            new_item_ptr = items_.New();
            new_item_ptr->str = nullptr;
        }

        new_item_ptr = items_.New();
        new_item_ptr->str = str_heap_.AddWord(str);
        new_item_ptr->weight = weight;
    }

    IdName<char> pair(item_id, new_item_ptr->str);
    item_table_.Insert(pair);
    return new_item_ptr;
}

template <class WeightType>
bool StringDict<WeightType>::Store(const std::string &path, bool show_progress) const
{
    BinaryFileStream strm;
    bool ret = strm.OpenForWrite(path);
    if(!ret) {
        LogError("Failed to open file %s", path.c_str());
        return false;
    }

#if defined ENABLE_CRYPT_HASH
    if (!encryption_str_.empty()) {
        strm.SetEncryptionString(encryption_str_);
    }
#endif //ENABLE_CRYPT_HASH

    ret = Write(strm, show_progress);
    ret = ret && strm.Flush();
    return ret && strm.IsGood();
}

template <class WeightType>
bool StringDict<WeightType>::Load(const std::string &path, bool is_binary, bool show_progress)
{
    BinFileStream strm;
    bool ret = strm.Open(path.c_str());
    if(!ret) {
        LogError("Failed to open file %s", path.c_str());
        return false;
    }

#if defined ENABLE_CRYPT_HASH
    if (!encryption_str_.empty()) {
        strm.SetEncryptionString(encryption_str_);
    }
#endif //ENABLE_CRYPT_HASH

    return is_binary ? Read(strm, show_progress) : ReadFromPlainText(strm);
}

template <class WeightType>
bool StringDict<WeightType>::Print(const std::string &path, uint32_t options, uint32_t nTopK) const
{
    std::ofstream ps(path.c_str(), std::ios::binary);
    if(!ps) {
        return false;
    }

    uint32_t item_num = Size();
    uint32_t sort_by = (options & StrDictOpt::SortBy_Mask);
    uint32_t print_format = (options & StrDictOpt::Print_Mask);
    if(sort_by == StrDictOpt::SortById)
    {
        for(uint32_t item_idx = 0; item_idx < min(nTopK, item_num); item_idx++)
        {
            const StrEntry<WeightType> *pCurItem = Get(item_idx);
            if(pCurItem != nullptr && pCurItem->str != nullptr)
            {
                PrintItem(ps, item_idx, *pCurItem, print_format);
            }
        }
        ps.close();
        return ps.good();
    }

    std::vector<ItemEx> item_list;
    item_list.reserve(item_num);
    ItemEx item_ex;
    for(uint32_t item_idx = 0; item_idx < item_num; item_idx++)
    {
        item_ex.id = item_idx;
        item_ex.item = Get(item_idx);
        if(item_ex.item != nullptr && item_ex.item->str != nullptr) {
            item_list.push_back(item_ex);
        }
    }

    if(sort_by == PO_SORT_BY_FREQ) {
        std::sort(item_list.begin(), item_list.end(), ItemEx::StrItemExGreaterFreq);
    }
    else if(sort_by == PO_SORT_BY_STR) {
        std::sort(item_list.begin(), item_list.end(), ItemEx::StrItemExLessStr);
    }

    for(size_t item_idx = 0; item_idx < min(nTopK, (uint32_t)item_list.size()); item_idx++)
    {
        const ItemEx &cur_item = item_list[item_idx];
        PrintItem(ps, cur_item.id, *cur_item.item, print_format);
    }

    ps.close();
    return ps.good();
}

template <class WeightType>
bool StringDict<WeightType>::Write(IBinStream &stream, bool show_progress) const
{
    uint32_t item_num = Size();
    bool ret = stream.Write((const char*)&item_num, sizeof(item_num));

    TaskMonitor tm(1000000);
    uint16_t item_len = 0;
    for(uint32_t item_idx = 0; ret && item_idx < item_num; ++item_idx)
    {
        const auto &item = items_[item_idx];
        ret = ret && stream.Write((const char*)&item_idx, sizeof(item_idx));
        item_len = item.str != nullptr ? (uint16_t)strlen(item.str) : 0;
        ret = ret && stream.Write((const char*)&item_len, sizeof(item_len));
        if(item_len > 0) {
            ret = ret && stream.Write(item.str, item_len);
        }
        ret = ret && stream.Write((const char*)&item.weight, sizeof(item.weight));

        if (show_progress) {
            tm.Progress(item_idx + 1);
        }
    }

    if (show_progress) {
        tm.End();
    }

    return ret && stream.IsGood();
}

template <class WeightType>
bool StringDict<WeightType>::Read(IBinStream &stream, bool show_progress)
{
    Clear();
    uint32_t item_num = 0;
    bool ret = stream.Read((char*)&item_num, sizeof(item_num));

    TaskMonitor tm(1000000);
    uint16_t item_len = 0;
    char buf[65536];
    uint32_t item_id = 0;
    StrEntry<WeightType> item;
    items_.resize(item_num);
    for(uint32_t item_idx = 0; ret && item_idx < item_num; item_idx++)
    {
        ret = ret && stream.Read((char*)&item_id, sizeof(item_id));
        if(ret && item_id >= (uint32_t)items_.Size())
        {
            if(item_id == Number::MaxUInt32) {
                LogError("Invalid item-id: %u", item_id);
                return false;
            }
            items_.resize(item_id + 1);
        }

        ret = ret && stream.Read((char*)&item_len, sizeof(item_len));
        if(item_len > 0) {
            ret = ret && stream.Read(buf, item_len);
        }
        buf[item_len] = '\0';
        ret = ret && stream.Read((char*)&item.weight, sizeof(item.weight));

        if(ret)
        {
            item.str = str_heap_.AddWord(buf);
            items_[item_id] = item;
            IdName<char> pair;
            pair.id = item_id;
            pair.name = item.str;
            item_table_.Insert(pair);
        }

        if (show_progress) {
            tm.Progress(item_idx + 1);
        }
    }

    if (show_progress) {
        tm.End();
    }

    ret = ret && stream.IsGood();
    return ret;
}

template <class WeightType>
bool StringDict<WeightType>::ReadFromPlainText(IBinStream &strm)
{
    Clear();

    string line_text;
    vector<string> tokens;
    while (strm.GetLine(line_text))
    {
        String::Trim(line_text);
        String::Split(line_text, tokens, "\t");
        if (tokens.empty()) {
            continue;
        }

        if (tokens.size() < 2) {
            LogError("Invalid line format");
            return false;
        }

        uint32_t str_id = (uint32_t)atoi(tokens[0].c_str());
        const auto *item_ptr = AddItem(str_id, tokens[1].c_str(), 0, false);
        if (item_ptr == nullptr) {
            return false;
        }
    }

    return true;
}

#if defined ENABLE_CRYPT_HASH
template <class WeightType>
void StringDict<WeightType>::SetEncryptionInfo(const string &spec_str)
{
    uint32_t len = 0;
    uint64_t seed = 0;
    CryptHash::ParseEncryptionSpec(len, seed, StringUtil::Utf8ToWideStr(spec_str));
    CryptHash::BuildCryptString(encryption_str_, len, seed);
}

template <class WeightType>
void StringDict<WeightType>::SetEncryptionInfo(uint32_t len, uint64_t seed)
{
    CryptHash::BuildCryptString(encryption_str_, len, seed);
}
#endif //ENABLE_CRYPT_HASH

template <class WeightType>
void StringDict<WeightType>::PrintItem(ostream &ps, uint32_t item_id, const Item &item, uint32_t format) const
{
    switch(format)
    {
    case StrDictOpt::Print_Str:
        ps << item.str << "\r\n";
        break;
    case StrDictOpt::Print_IdStr:
        ps << item_id << "\t" << item.str << "\r\n";
        break;
    case StrDictOpt::Print_IdStrWght:
        ps << item_id << "\t" << item.str << "\t" << item.weight << "\r\n";
        break;
    case StrDictOpt::Print_IdWghtStr:
        ps << item_id << "\t" << item.weight << "\t" << item.str << "\r\n";
        break;
    case StrDictOpt::Print_StrWght:
    default:
        ps << item.str << "\t" << item.weight << "\r\n";
        break;
    }
}

////////////////////////////////////////////////////////////////////////////////
// WStringDict Implementation
////////////////////////////////////////////////////////////////////////////////

template <class WeightType>
const WStrEntry<WeightType>* WStringDict<WeightType>::Find(const wchar_t *item_str, uint32_t &item_id) const
{
    IdName<wchar_t> key(0, item_str);
    auto iter = item_table_.Find(key);
    if(iter.IsEnd()) {
        item_id = UINT32_MAX;
        return nullptr;
    }

    item_id = (*iter).id;
    return &items_[item_id];
}

template <class WeightType>
const WStrEntry<WeightType>* WStringDict<WeightType>::Find(const wchar_t *item_str) const
{
    IdName<wchar_t> key(0, item_str);
    auto iter = item_table_.Find(key);
    return iter.IsEnd() ? nullptr : &items_[(*iter).id];
}

template <class WeightType>
WStrEntry<WeightType>* WStringDict<WeightType>::Find(const wchar_t *item_str, uint32_t &item_id)
{
    IdName<wchar_t> key(0, item_str);
    auto iter = item_table_.Find(key);
    if(iter.IsEnd()) {
        item_id = UINT32_MAX;
        return nullptr;
    }

    item_id = (*iter).id;
    return &items_[item_id];
}

template <class WeightType>
uint32_t WStringDict<WeightType>::TermId(const wchar_t *str) const
{
    uint32_t item_id = UINT32_MAX;
    Find(str, item_id);
    return item_id;
}

template <class WeightType>
uint32_t WStringDict<WeightType>::TermId(const wstring &str) const
{
    return TermId(str.c_str());
}

template <class WeightType>
uint32_t WStringDict<WeightType>::ItemId(const wchar_t *str) const
{
    uint32_t item_id = UINT32_MAX;
    Find(str, item_id);
    return item_id;
}

template <class WeightType>
uint32_t WStringDict<WeightType>::ItemId(const wstring &str) const
{
    return ItemId(str.c_str());
}

template <class WeightType>
const wchar_t* WStringDict<WeightType>::ItemStr(uint32_t item_id) const
{
    const auto *item_ptr = Get(item_id);
    return item_ptr != nullptr ? item_ptr->str : nullptr;
}

template <class WeightType>
bool WStringDict<WeightType>::Reinit(bool use_location32)
{
    Clear();
    return item_table_.Reinit(0, 1.5f, use_location32);
}

template <class WeightType>
void WStringDict<WeightType>::Clear()
{
    items_.Clear();
    item_table_.Clear();
    str_heap_.Reset();
}

template <class WeightType>
bool WStringDict<WeightType>::AddItem(const wstring &itemStr, const WeightType &weight)
{
    uint32_t item_id = 0;
    const WStrEntry<WeightType> *item_ptr = AddItem(itemStr.c_str(), weight, item_id);
    return item_ptr != nullptr;
}

template <class WeightType>
bool WStringDict<WeightType>::AddItem(const wchar_t *item_str, const WeightType &weight)
{
    uint32_t item_id = 0;
    const WStrEntry<WeightType> *item_ptr = AddItem(item_str, weight, item_id);
    return item_ptr != nullptr;
}

template <class WeightType>
const WStrEntry<WeightType>* WStringDict<WeightType>::AddItem(const wchar_t *item_str,
    const WeightType &weight, uint32_t &item_id)
{
    return AddItem(item_str, weight, item_id, false);
}

template <class WeightType>
const WStrEntry<WeightType>* WStringDict<WeightType>::AddItem(const wstring &itemStr,
    const WeightType &weight, uint32_t &item_id, bool is_agg_weight)
{
    return AddItem(itemStr.c_str(), weight, item_id, is_agg_weight);
}

template <class WeightType>
const WStrEntry<WeightType>* WStringDict<WeightType>::AddItem(const wchar_t *item_str,
    const WeightType &weight, uint32_t &item_id, bool is_agg_weight)
{
    WStrEntry<WeightType> *found_item = Find(item_str, item_id);
    if(found_item != nullptr)
    {
        if(!is_agg_weight) {
            return nullptr;
        }

        found_item->weight += weight;
        return found_item;
    }

    WStrEntry<WeightType> *new_item_ptr = items_.New();
    new_item_ptr->str = str_heap_.AddWord(item_str);
    new_item_ptr->weight = weight;
    item_id = (uint32_t)items_.Size() - 1;

    IdName<wchar_t> pair;
    pair.id = item_id;
    pair.name = new_item_ptr->str;
    item_table_.Insert(pair);

    return new_item_ptr;
}

template <class WeightType>
const WStrEntry<WeightType>* WStringDict<WeightType>::AddItemMax(const wchar_t *item_str,
    const WeightType &weight, uint32_t &item_id)
{
    WStrEntry<WeightType> *found_item = Find(item_str, item_id);
    if (found_item != nullptr)
    {
        if (weight > found_item->weight) {
            found_item->weight = weight;
        }
        return found_item;
    }

    WStrEntry<WeightType> *new_item_ptr = items_.New();
    new_item_ptr->str = str_heap_.AddWord(item_str);
    new_item_ptr->weight = weight;
    item_id = (uint32_t)items_.Size() - 1;

    IdName<wchar_t> pair;
    pair.id = item_id;
    pair.name = new_item_ptr->str;
    item_table_.Insert(pair);

    return new_item_ptr;
}

template <class WeightType>
bool WStringDict<WeightType>::AddWeight(uint32_t item_id, const WeightType &weight)
{
    if(item_id < (uint32_t)items_.Size())
    {
        items_[item_id].weight += weight;
        return true;
    }
    return false;
}

template <class WeightType>
bool WStringDict<WeightType>::AddWeight(const wstring &str, const WeightType &weight)
{
    uint32_t item_id = 0;
    const WStrEntry<WeightType> *item_ptr = AddItem(str, weight, item_id, true);
    return item_ptr != nullptr;
}

template <class WeightType>
bool WStringDict<WeightType>::AddWeight(const wchar_t *item_str, const WeightType &weight)
{
    uint32_t item_id = 0;
    const WStrEntry<WeightType> *item_ptr = AddItem(item_str, weight, item_id, true);
    return item_ptr != nullptr;
}

template <class WeightType>
bool WStringDict<WeightType>::SetWeight(const wchar_t *item_str, const WeightType &weight)
{
    IdName<wchar_t> key(0, item_str);
    auto iter = item_table_.Find(key);
    if(!iter.IsEnd())
    {
        items_[(*iter).id].weight = weight;
        return true;
    }

    return false;
}

template <class WeightType>
bool WStringDict<WeightType>::SetWeight(uint32_t item_id, const WeightType &weight)
{
    if(item_id < (uint32_t)items_.Size())
    {
        items_[item_id].weight = weight;
        return true;
    }
    return false;
}

template <class WeightType>
const WStrEntry<WeightType>* WStringDict<WeightType>::AddItem(uint32_t item_id,
    const wchar_t *item_str, const WeightType &weight, bool is_agg_weight)
{
    uint32_t old_item_id = UINT32_MAX;
    WStrEntry<WeightType> *found_item = Find(item_str, old_item_id);
    if(found_item != nullptr)
    {
        if(old_item_id != item_id) {
            LogWarning("This item has been assigned with ID %u (now %u)", old_item_id, item_id);
            return nullptr;
        }

        if(!is_agg_weight) {
            return nullptr;
        }

        found_item->weight += weight;
        return found_item;
    }

    uint32_t item_num_now = (uint32_t)items_.size();
    WStrEntry<WeightType> *new_item_ptr = nullptr;
    if(item_id < item_num_now)
    {
        new_item_ptr = &items_[item_id];
        if(new_item_ptr->str != nullptr) {
            LogError("Something wrong with the string dictionary");
            return nullptr;
        }

        new_item_ptr->str = str_heap_.AddWord(item_str);
        new_item_ptr->weight = weight;
    }
    else
    {
        for(uint32_t idx = item_num_now; idx < item_id; idx++)
        {
            new_item_ptr = items_.New();
            new_item_ptr->str = nullptr;
        }

        new_item_ptr = items_.New();
        new_item_ptr->str = str_heap_.AddWord(item_str);
        new_item_ptr->weight = weight;
    }

    IdName<wchar_t> pair(item_id, new_item_ptr->str);
    item_table_.Insert(pair);
    return new_item_ptr;
}

template <class WeightType>
bool WStringDict<WeightType>::Store(const std::string &path, bool show_progress) const
{
    BinaryFileStream strm;
    bool ret = strm.OpenForWrite(path);
    if(!ret) {
        LogError("Failed to open file %s", path.c_str());
        return false;
    }

#if defined ENABLE_CRYPT_HASH
    if (!encryption_str_.empty()) {
        strm.SetEncryptionString(encryption_str_);
    }
#endif //ENABLE_CRYPT_HASH

    ret = Write(strm, show_progress);
    ret = ret && strm.Flush();
    return ret && strm.IsGood();
}

template <class WeightType>
bool WStringDict<WeightType>::Load(const std::string &path, bool is_binary, bool show_progress)
{
    BinFileStream strm;
    bool ret = strm.Open(path.c_str());
    if(!ret) {
        LogError("Failed to open file %s", path.c_str());
        return false;
    }

#if defined ENABLE_CRYPT_HASH
    if (!encryption_str_.empty()) {
        strm.SetEncryptionString(encryption_str_);
    }
#endif //ENABLE_CRYPT_HASH

    return is_binary ? Read(strm, show_progress) : ReadFromPlainText(strm);
}

template <class WeightType>
bool WStringDict<WeightType>::Print(const std::string &path, uint32_t options) const
{
    std::ofstream ps(path.c_str(), std::ios::binary);
    if(!ps) {
        return false;
    }

    std::string item_str;
    uint32_t sort_by = (options & StrDictOpt::SortBy_Mask);
    uint32_t print_format = (options & StrDictOpt::Print_Mask);
    if(sort_by == StrDictOpt::SortById)
    {
        for(uint32_t item_idx = 0; item_idx < (uint32_t)items_.size(); item_idx++)
        {
            const WStrEntry<WeightType> &cur_item = items_[item_idx];
            if(cur_item.str != nullptr) {
                item_str = StringUtil::ToUtf8(cur_item.str);
            }
            else {
                item_str.clear();
            }

            PrintItem(ps, item_idx, item_str, cur_item.weight, print_format);
        }
        ps.close();
        return ps.good();
    }

    std::vector<ItemEx> item_list;
    item_list.reserve((size_t)items_.Size());
    ItemEx item_ex;
    for(uint32_t item_idx = 0; item_idx < (uint32_t)items_.size(); item_idx++)
    {
        item_ex.id = item_idx;
        item_ex.item = &items_[item_idx];
        item_list.push_back(item_ex);
    }

    if(sort_by == PO_SORT_BY_FREQ) {
        std::sort(item_list.begin(), item_list.end(), ItemEx::WStrItemExGreaterFreq);
    }
    else if(sort_by == PO_SORT_BY_STR) {
        std::sort(item_list.begin(), item_list.end(), ItemEx::WStrItemExLessStr);
    }

    for(uint32_t item_idx = 0; item_idx < (uint32_t)item_list.size(); item_idx++)
    {
        const ItemEx &cur_item = item_list[item_idx];
        item_str = StringUtil::ToUtf8(cur_item.item->str);
        PrintItem(ps, cur_item.id, item_str, cur_item.item->weight, print_format);
    }

    ps.close();
    return ps.good();
}

template <class WeightType>
bool WStringDict<WeightType>::Write(IBinStream &stream, bool show_progress) const
{
    uint32_t item_num = Size();
    bool ret = stream.Write(item_num);

    TaskMonitor tm(1000000);
    std::string item_str;
    uint16_t item_len = 0;
    for(uint32_t item_idx = 0; ret && item_idx < item_num; ++item_idx)
    {
        const auto &item = items_[item_idx];
        ret = ret && stream.Write(item_idx);
        item_str = StringUtil::ToUtf8(item.str);
        item_len = (uint16_t)item_str.size();
        ret = ret && stream.Write(item_len);
        ret = ret && stream.Write(item_str.c_str(), item_len);
        ret = ret && stream.Write(item.weight);

        if (show_progress)
        {
            tm.Progress(item_idx + 1);
            if (!ret) {
                LogError("Error occurred in writing item %u", item_idx);
            }
        }
    }

    if (show_progress) {
        tm.End();
    }

    return ret && stream.IsGood();
}

template <class WeightType>
bool WStringDict<WeightType>::Read(IBinStream &stream, bool show_progress)
{
    Clear();
    uint32_t item_num = 0;
    bool ret = stream.Read((char*)&item_num, sizeof(item_num));

    TaskMonitor tm(1000000);
    uint16_t item_len = 0;
    char buf[65536];
    std::wstring witem_str;
    uint32_t item_id = 0;
    WStrEntry<WeightType> item;
    for(uint32_t item_idx = 0; ret && item_idx < item_num; item_idx++)
    {
        ret = ret && stream.Read((char*)&item_id, sizeof(item_id));
        if (!ret) {
            return false;
        }

        if (item_id >= item_num)
        {
            LogError("Invalid item-id: %u", item_id);
            return false;
        }

        if (item_id >= (uint32_t)items_.Size())
        {
            items_.resize(item_id + 1);
        }

        ret = ret && stream.Read((char*)&item_len, sizeof(item_len));
        ret = ret && stream.Read(buf, item_len);
        buf[item_len] = '\0';
        ret = ret && stream.Read((char*)&item.weight, sizeof(item.weight));

        if(ret)
        {
            witem_str = StringUtil::Utf8ToWideStr(buf);
            item.str = str_heap_.AddWord(witem_str);
            items_[item_id] = item;
            IdName<wchar_t> pair;
            pair.id = item_id;
            pair.name = item.str;
            item_table_.Insert(pair);
        }

        if (show_progress) {
            tm.Progress(item_idx + 1);
        }
    }

    if (show_progress) {
        tm.End();
    }

    ret = ret && stream.IsGood();
    return ret;
}

template<class WeightType>
inline bool WStringDict<WeightType>::ReadFromPlainText(IBinStream & strm)
{
    Clear();

    wstring line_text;
    vector<wstring> tokens;
    while (strm.GetLine(line_text))
    {
        WString::Trim(line_text);
        WString::Split(line_text, tokens, L"\t");
        if (tokens.empty()) {
            continue;
        }

        if (tokens.size() < 2) {
            LogError("Invalid line format");
            return false;
        }

        uint32_t str_id = (uint32_t)wcstol(tokens[0].c_str(), nullptr, 10);
        const auto *item_ptr = AddItem(str_id, tokens[1].c_str(), 0, false);
        if (item_ptr == nullptr) {
            return false;
        }
    }

    return true;
}

#if defined ENABLE_CRYPT_HASH
template <class WeightType>
void WStringDict<WeightType>::SetEncryptionInfo(const wstring &spec_str)
{
    uint32_t len = 0;
    uint64_t seed = 0;
    CryptHash::ParseEncryptionSpec(len, seed, spec_str);
    CryptHash::BuildCryptString(encryption_str_, len, seed);
}

template <class WeightType>
void WStringDict<WeightType>::SetEncryptionInfo(uint32_t len, uint64_t seed)
{
    CryptHash::BuildCryptString(encryption_str_, len, seed);
}
#endif //ENABLE_CRYPT_HASH

template <class WeightType>
void WStringDict<WeightType>::PrintItem(ostream &ps, uint32_t item_id, const string &item_str,
    const WeightType &weight, uint32_t format) const
{
    switch(format)
    {
    case StrDictOpt::Print_Str:
        ps << item_str << "\r\n";
        break;
    case StrDictOpt::Print_IdStr:
        ps << item_id << "\t" << item_str << "\r\n";
        break;
    case StrDictOpt::Print_IdStrWght:
        ps << item_id << "\t" << item_str << "\t" << weight << "\r\n";
        break;
    case StrDictOpt::Print_IdWghtStr:
        ps << item_id << "\t" << weight << "\t" << item_str << "\r\n";
        break;
    case StrDictOpt::Print_StrWght:
    default:
        ps << item_str << "\t" << weight << "\r\n";
        break;
    }
}

} //end of namespace
