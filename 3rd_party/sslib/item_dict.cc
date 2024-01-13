#include "item_dict.h"
#include <fstream>
#include <algorithm>
#include "number.h"

using namespace std;

namespace sslib
{

bool ItemInfoPtrGreaterWeight(const ItemInfo *lhs, const ItemInfo *rhs) {
    return lhs->weight != rhs->weight ? lhs->weight > rhs->weight : (*lhs) < (*rhs);
}
bool ItemInfoPtrLessStr(const ItemInfo *lhs, const ItemInfo *rhs) {
    return (*lhs) < (*rhs);
}
bool ItemInfoPtrLessId(const ItemInfo *lhs, const ItemInfo *rhs) {
    return lhs->id < rhs->id;
}

ItemDict::ItemDict()
{
    max_item_id_ = 0;
}

ItemDict::~ItemDict()
{
    Clear();
}

const ItemInfo* ItemDict::Find(uint32_t id) const
{
    IdAndItemPtr key;
    key.id = id;
    MapId2Item::ConstIterator iter = id_to_item_map_.Find(key);
    return !iter.IsEnd() ? (*iter).item_ptr : nullptr;
}

const ItemInfo* ItemDict::Find(const char *item_str) const
{
    ItemInfo item(0, item_str, 0);
    ItemInfoTable::ConstIterator iter = items_.Find(item);
    return !iter.IsEnd() ? &(*iter) : nullptr;
}

ItemInfo* ItemDict::Find(const char *item_str)
{
    ItemInfo item(0, item_str, 0);
    ItemInfoTable::Iterator iter = items_.Find(item);
    return !iter.IsEnd() ? &(*iter) : nullptr;
}

uint32_t ItemDict::ItemId(const char *item_str) const
{
    ItemInfo item(0, item_str, 0);
    ItemInfoTable::ConstIterator iter = items_.Find(item);
    return !iter.IsEnd() ? (*iter).id : UINT32_MAX;
}

uint32_t ItemDict::ItemId(const string &item_str) const
{
    ItemInfo item(0, item_str.c_str(), 0);
    ItemInfoTable::ConstIterator iter = items_.Find(item);
    return !iter.IsEnd() ? (*iter).id : UINT32_MAX;
}

const char* ItemDict::ItemStr(uint32_t item_id) const
{
    IdAndItemPtr key;
    key.id = item_id;
    MapId2Item::ConstIterator iter = id_to_item_map_.Find(key);
    return !iter.IsEnd() ? (*iter).item_ptr->str : nullptr;
}

void ItemDict::Clear()
{
    items_.Clear();
    id_to_item_map_.Clear();
    str_heap_.Reset();
}

bool ItemDict::AddItem(uint32_t id, const char *item_str, uint32_t freq)
{
    const ItemInfo *item_ptr = AddItemEx(id, item_str, freq);
    return item_ptr != nullptr;
}

const ItemInfo* ItemDict::AddItemEx(uint32_t id, const char *item_str, uint32_t freq)
{
    const ItemInfo *item_find1 = Find(id);
    const ItemInfo *item_find2 = Find(item_str);
    if(item_find1 != nullptr || item_find2 != nullptr) {
        return nullptr;
    }

    ItemInfo item(id, nullptr, freq);
    item.str = str_heap_.AddWord(item_str);
    if(id > max_item_id_) {
        max_item_id_ = id;
    }

    ItemInfoTable::Iterator iter = items_.Insert(item);
    IdAndItemPtr key(id, &(*iter));
    id_to_item_map_.Insert(key);
    return &(*iter);
}

bool ItemDict::Store(const std::string &path) const
{
    ofstream stream(path.c_str(), ios::binary);
    if(!stream) {
        return false;
    }

    bool ret = Write(stream);
    stream.close();
    return ret && stream.good();
}

bool ItemDict::Load(const std::string &path, uint32_t freq_threshold)
{
    ifstream stream(path.c_str(), ios::binary);
    if(!stream) {
        return false;
    }
    bool ret = Read(stream, freq_threshold);
    stream.close();
    return ret;
}

bool ItemDict::Print(const std::string &path, uint32_t options) const
{
    ofstream ps(path.c_str());
    if(!ps) {
        return false;
    }

    uint32_t sort_by = (options & PO_SORT_BY_MASK);
    vector<const ItemInfo*> items;
    items.reserve((size_t)id_to_item_map_.Size());
    MapId2Item::ConstIterator iter = id_to_item_map_.Begin();
    for(; !iter.IsEnd(); iter.Next())
    {
        const ItemInfo *item_ptr = (*iter).item_ptr;
        items.push_back(item_ptr);
    }

    if(sort_by == PO_SORT_BY_FREQ) {
        std::sort(items.begin(), items.end(), ItemInfoPtrGreaterWeight);
    }
    else if(sort_by == PO_SORT_BY_STR) {
        std::sort(items.begin(), items.end(), ItemInfoPtrLessStr);
    }
    else {
        std::sort(items.begin(), items.end(), ItemInfoPtrLessId);
    }

    for(size_t item_idx = 0; item_idx < items.size(); item_idx++)
    {
        const ItemInfo *item_ptr = items[item_idx];
        ps << item_ptr->id << "\t" << item_ptr->weight << "\t" << item_ptr->str << endl;
    }

    ps.close();
    return ps.good();
}

bool ItemDict::Write(ostream &stream) const
{
    uint32_t item_num = (uint32_t)id_to_item_map_.Size();
    stream.write((const char*)&item_num, sizeof(item_num));

    uint16_t item_len = 0;
    MapId2Item::ConstIterator iter = id_to_item_map_.Begin();
    for(; !iter.IsEnd(); iter.Next())
    {
        const ItemInfo &item = *(*iter).item_ptr;
        stream.write((const char*)&item.id, sizeof(item.id));
        item_len = (uint16_t)strlen(item.str);
        stream.write((const char*)&item_len, sizeof(item_len));
        stream.write(item.str, item_len);
        stream.write((const char*)&item.weight, sizeof(item.weight));
    }

    return stream.good();
}

bool ItemDict::Read(istream &stream, uint32_t freq_threshold)
{
    bool ret = true;
    Clear();
    uint32_t item_num = 0;
    stream.read((char*)&item_num, sizeof(item_num));

    uint16_t item_len = 0;
    char buf[65536];
    ItemInfo item;
    for(uint32_t item_idx = 0; ret && item_idx < item_num; item_idx++)
    {
        stream.read((char*)&item.id, sizeof(item.id));
        stream.read((char*)&item_len, sizeof(item_len));
        stream.read(buf, item_len);
        buf[item_len] = '\0';
        stream.read((char*)&item.weight, sizeof(item.weight));

        if(item.weight >= freq_threshold) {
            ret = AddItem(item.id, buf, item.weight);
        }
    }
    
    ret = ret && stream.good();
    return ret;
}

} //end of namespace
