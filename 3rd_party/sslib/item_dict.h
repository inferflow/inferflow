#pragma once

#include "prime_types.h"
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include "string_blocked_heap.h"
#include "hashtable.h"
#include "hash.h"

namespace sslib
{

#pragma pack(push, 1)
struct ItemInfo
{
    uint32_t id;
    const char *str;
    uint32_t weight;

    ItemInfo(uint32_t p_id = 0, const char *s = nullptr, uint32_t w = 0)
    {
        id = p_id;
        str = s;
        weight = w;
    }

    int Compare(const ItemInfo &item) const {
        return strcmp(str, item.str);
    }
    bool operator < (const ItemInfo &item) const {
        return strcmp(str, item.str) < 0;
    }
    static bool GreaterWeight(const ItemInfo &lhs, const ItemInfo &rhs) {
        return lhs.weight > rhs.weight;
    }
};
#pragma pack(pop)

extern bool ItemInfoPtrGreaterWeight(const ItemInfo *lhs, const ItemInfo *rhs);
extern bool ItemInfoPtrLessStr(const ItemInfo *lhs, const ItemInfo *rhs);
extern bool ItemInfoPtrLessId(const ItemInfo *lhs, const ItemInfo *rhs);

class ItemInfoHashTraits
{
public:
    uint32_t operator()(const ItemInfo& key) const {
        return Hash::HashCodeUInt32(key.str);
    }

    int operator()(const ItemInfo& lhs, const ItemInfo& rhs) const {
        return lhs.Compare(rhs);
    }
};

#pragma pack(push, 1)
struct IdAndItemPtr
{
    uint32_t id;
    const ItemInfo *item_ptr;

    IdAndItemPtr(uint32_t p_id = 0, const ItemInfo *p_item_ptr = nullptr) {
        id = p_id;
        item_ptr = p_item_ptr;
    }
};
#pragma pack(pop)

class IdAndItemPtrHashTraits
{
public:
    uint32_t operator()(const IdAndItemPtr& key) const {
        return key.id;
    }

    int operator()(const IdAndItemPtr& lhs, const IdAndItemPtr& rhs) const {
        return lhs.id < rhs.id ? -1 : (lhs.id == rhs.id ? 0 : 1);
    }
};

typedef Hashtable<ItemInfo, ItemInfoHashTraits> ItemInfoTable;
typedef Hashtable<IdAndItemPtr, IdAndItemPtrHashTraits> MapId2Item;
//typedef std::map<uint32_t, ItemInfo*> MapId2Item;

class ItemDict
{
public:
    enum PrintOptionsEnum {
        PO_SORT_BY_MASK     = 0x00FF,
        PO_SORT_BY_ID       = 0x0001,
        PO_SORT_BY_STR      = 0x0002,
        PO_SORT_BY_FREQ     = 0x0003,
        PO_SORT_DESC        = 0x0100
    };

    struct Iter
    {
    public:
        bool Next() {
            return iter_.Next();
        }
        bool IsEnd() {
            return iter_.IsEnd();
        }
        const ItemInfo* GetValue() {
            return &(*iter_);
        }
    protected:
        ItemInfoTable::ConstIterator iter_;
        friend class ItemDict;
    };

public:
    ItemDict();
    virtual ~ItemDict();

    uint32_t GetItemNum() const {
        return (uint32_t)items_.Size();
    }
    uint32_t GetMaxItemId() const {
        return max_item_id_;
    }

    const ItemInfo* Find(uint32_t nId) const;
    const ItemInfo* Find(const char *strItem) const;
    ItemInfo* Find(const char *strItem);
    uint32_t ItemId(const char *strItem) const;
    uint32_t ItemId(const std::string &strItem) const;
    const char* ItemStr(uint32_t iItemId) const;

    void Clear();
    bool AddItem(uint32_t nId, const char *szItem, uint32_t nFreq);
    const ItemInfo* AddItemEx(uint32_t nId, const char *szItem, uint32_t nFreq);

    Iter Begin() const
    {
        Iter iter;
        iter.iter_ = items_.Begin();
        return iter;
    }

    bool Store(const std::string &strPath) const;
    bool Load(const std::string &strPath, uint32_t nFreqThreshold = 0);
    bool Print(const std::string &strPath, uint32_t nOptions) const;

    bool Write(std::ostream &stream) const;
    bool Read(std::istream &stream, uint32_t nFreqThreshold = 0);

protected:
    ItemInfoTable items_;
    MapId2Item id_to_item_map_;

    uint32_t max_item_id_;
    StringBlockedHeap str_heap_;
};

} //end of namespace
