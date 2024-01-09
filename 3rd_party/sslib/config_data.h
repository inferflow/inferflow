#pragma once

#include <string>
#include <cstring>
#include <set>
#include <iostream>
#include "map.h"
#include "binary_stream.h"
#include "log.h"

namespace sslib
{

using std::string;
using std::vector;

class ConfigData
{
public:
    struct SectionInfo
    {
        std::string name;
        std::string comments;
        uint32_t order = UINT32_MAX;
    };

    struct ConfigItem
    {
        std::string key, value;
        std::string comments;
        uint32_t order = UINT32_MAX;

        ConfigItem(const std::string p_name = "")
        {
            key = p_name;
            order = UINT32_MAX;
        }
    };

    template <class ItemType>
    struct TypedItem
    {
        ItemType type;
        wstring name, params;
    };

    typedef std::map<std::string, uint32_t, StrLessNoCase> ValueMap;

public:
    ConfigData(void);
    virtual ~ConfigData(void);

    bool Load(const std::string &file_path);
    bool Load(IBinStream &strm);
    bool Store(const std::string &file_path) const;
    bool Store(std::ostream &strm) const;
    void Clear();

    bool GetItem(const std::string &section, const std::string &key,
        int &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        unsigned int &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        uint64_t &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        int64_t &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        std::string &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        std::wstring &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        float &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        double &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        bool &value, bool log_upon_error = false) const;
    bool GetItem(const std::string &section, const std::string &key,
        const ValueMap &value_map, const std::string &strSep,
        uint32_t &value, bool log_upon_error = false) const;

    bool GetItemList(vector<wstring> &item_list,
        const string &section, const string &name_prefix,
        bool log_upon_error = false) const;
    bool GetItemList(vector<string> &item_list,
        const string &section, const string &name_prefix,
        bool log_upon_error = false) const;

    template <class ItemType>
    bool GetItemList(vector<ItemType> &item_list, const WStrMap<ItemType> &vmap,
        const string &section, const string &name_prefix,
        bool log_upon_error = false) const
    {
        vector<wstring> strList;
        bool ret = GetItemList(strList, section, name_prefix, log_upon_error);

        for (const wstring &str : strList)
        {
            auto iterFind = vmap.find(str);
            if (iterFind == vmap.end())
            {
                if (log_upon_error) {
                    LogError(L"Invalid item: %ls", str.c_str());
                }
                return false;
            }

            item_list.push_back(iterFind->second);
        }

        return ret;
    }

    template <class ItemType>
    bool GetItemList(vector<TypedItem<ItemType>> &item_list, const WStrMap<ItemType> &vmap,
        const string &section, const string &name_prefix,
        bool log_upon_error = false) const
    {
        vector<wstring> str_list;
        bool ret = GetItemList(str_list, section, name_prefix, log_upon_error);

        for (const wstring &str : str_list)
        {
            auto offset = str.find_first_of(L";,:");
            wstring item_str = offset != wstring::npos ? str.substr(0, offset) : str;
            wstring param_str = offset != wstring::npos ? str.substr(offset + 1) : L"";
            WString::Trim(item_str);
            WString::Trim(param_str);

            auto iter_find = vmap.find(item_str);
            if (iter_find == vmap.end())
            {
                if (log_upon_error) {
                    LogError(L"Invalid item: %ls", item_str.c_str());
                }
                return false;
            }

            TypedItem<ItemType> typed_item;
            typed_item.type = iter_find->second;
            typed_item.name = item_str;
            typed_item.params = param_str;
            item_list.push_back(typed_item);
        }

        return ret;
    }

    bool GetItems(const std::string &section,
        std::vector<ConfigItem> &item_list,
        bool log_upon_error = false) const;
    bool GetItems(std::vector<std::wstring> &value_list, const std::string &section,
        const std::string &item_base, int from_idx, int to_idx,
        bool log_upon_error = false) const;
    bool GetItems(const std::string &section, const std::string &key_base,
        int from_idx, int to_idx, std::vector<std::string> &value_list,
        bool log_upon_error = false) const;
    bool GetItems(const std::string &section, const std::string &key_base,
        int from_idx, int to_idx, std::vector<int> &value_list,
        bool log_upon_error = false) const;

    bool HasSection(const std::string &name) const;
    bool GetSections(std::vector<SectionInfo> &section_list) const;
    bool AddSection(const std::string &name, const std::string &comments = "", uint32_t order = UINT32_MAX);
    bool AddGroup(const std::string &name, const std::string &comments = "", uint32_t order = UINT32_MAX) {
        return AddSection(name, comments, order);
    }

    bool SetItem(const std::string &section, const std::string &key,
        long nVal, bool add_if_not_exist = false,
        const std::string &comments = "",
        uint32_t order = UINT32_MAX);
    bool SetItem(const std::string &section, const std::string &key,
        int nVal, bool add_if_not_exist = false,
        const std::string &comments = "",
        uint32_t order = UINT32_MAX);
    bool SetItem(const std::string &section, const std::string &key,
        unsigned int nVal, bool add_if_not_exist = false,
        const std::string &comments = "",
        uint32_t order = UINT32_MAX);
    bool SetItem(const std::string &section, const std::string &key,
        const std::string &strVal, bool add_if_not_exist = false,
        const std::string &comments = "",
        uint32_t order = UINT32_MAX);

    /// advanced functionalities
        //macros
    void EnableMacros(bool enable_explicit, bool enable_implicit);
    bool AddMacro(const std::string &macro_name, const std::string &macro_value);

protected:
    typedef std::map<std::string, ConfigItem, StrLessNoCase> ItemMap;

    struct Section
    {
        std::string name;
        ItemMap items;
        std::string comments;
        uint32_t order = UINT32_MAX;

        Section(const std::string &p_name = "")
        {
            name = p_name;
            order = UINT32_MAX;
        }
        bool operator < (const Section &group) const;
    };

    static bool LessOrderSectionPtr(const Section *lhs, const Section *rhs)
    {
        return lhs->order < rhs->order;
    }

    static bool LessOrderItemPtr(const ConfigItem *lhs, const ConfigItem *rhs)
    {
        return lhs->order < rhs->order;
    }

    typedef std::map<std::string, Section, StrLessNoCase> SectionMap;
    typedef std::map<std::string, std::string, StrLessNoCase> MacroMap;

protected:
    SectionMap groups_;

    bool enable_explicit_macros_ = true;
    bool enable_implicit_macros_ = false;
    MacroMap macros_;

protected:
    bool IsGroupNameLine(const std::string &line_str) const;
    bool ParseGroupLine(const std::string &line_str, Section &group);
    bool ParseItemLine(const std::string &line_str, ConfigItem &item);

    const ConfigData::Section* GetGroup(const std::string &section) const;
    const ConfigItem* GetItemInner(const std::string &section, const std::string &key, bool log_upon_error) const;
    ConfigItem* GetItemInner(const std::string &section, const std::string &key, bool log_upon_error);
    ConfigItem* AddItem(const std::string &section, const std::string &key, uint32_t order);

    std::string TransItemValue(const std::string &item_value, const std::string &section, int depth = 0) const;
};

} //end of namespace
