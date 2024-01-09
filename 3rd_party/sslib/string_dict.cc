#include "string_dict.h"

namespace sslib
{

void StrDict_ForBuildingOnly()
{
    StrEntry<float> entry1("abc", 1.0f), entry2;
    entry1.Compare(entry2);
    (void)(entry1 < entry2);
    StrEntry<float>::GreaterWeight(entry1, entry2);

    uint32_t item_id = 0;
    StringDict<uint32_t> dict(true);
    dict.Size();
    const StrEntry<uint32_t> *entry_ptr = dict.Get(0);
    (void)entry_ptr;
    dict.Find("abc");
    dict.Find("abc", item_id);
    dict.AddItem("a new item", 3125);
    dict.AddItem("a new item", 7890, item_id, true);
    dict.AddItem(50000, "item", 12, true);
    std::string path;
    dict.Store(path);
    dict.Load(path);
    dict.Print(path, StrDictOpt::SortByStr);

    WStrEntry<double> wentry1, wentry2(L"123", 5.5);
    wentry1.Compare(wentry2);
    (void)(wentry1 < wentry2);
    WStrEntry<double>::GreaterWeight(wentry1, wentry2);

    WStringDict<uint32_t> wdict(true);
    wdict.Size();
    const WStrEntry<uint32_t> *entry_ptr_w = wdict.Get(0);
    (void)entry_ptr_w;
    wdict.Find(L"abc");
    wdict.Find(L"abc", item_id);
    wdict.AddItem(L"a new item", 3125);
    wdict.AddItem(L"a new item", 7890, item_id, true);
    wdict.AddItem(50000, L"item", 12, true);
    wdict.Store(path);
    wdict.Load(path);
    wdict.Print(path, StrDictOpt::SortByStr);
}

} //end of namespace
