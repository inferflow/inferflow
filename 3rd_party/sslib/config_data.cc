#include "config_data.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "string.h"
#include "string_util.h"
#include "log.h"
#include "app_environment.h"
#include "binary_file_stream.h"

using namespace std;

namespace sslib
{

bool ConfigData::Section::operator < (const Section &group) const
{
    return strcasecmp(name.c_str(), group.name.c_str()) < 0;
}

ConfigData::ConfigData()
{
}

ConfigData::~ConfigData()
{
    Clear();
}

void ConfigData::Clear()
{
	groups_.clear();

    enable_explicit_macros_ = true;
    enable_implicit_macros_ = false;
    macros_.clear();
}

bool ConfigData::Load(const string &strFile)
{
    BinaryFileStream reader;
    bool ret = reader.OpenForRead(strFile);
    if (!ret) {
        LogError("Failed to open the configuration file: %s", strFile.c_str());
        return false;
    }

    ret = Load(reader);
    reader.Close();
    if (!ret) {
        LogError("Failed to load the configuration information into memory (file: %s)", strFile.c_str());
        return false;
    }

    AddMacro("DataHomeDir", AppEnv::DataRootDir());
    AddMacro("DataRootDir", AppEnv::DataRootDir());
    AddMacro("AppDir", AppEnv::AppDir());
    AddMacro("RunName", AppEnv::RunName());

    AddMacro("data_home_dir", AppEnv::DataRootDir());
    AddMacro("data_root_dir", AppEnv::DataRootDir());
    AddMacro("app_dir", AppEnv::AppDir());
    AddMacro("run_name", AppEnv::RunName());
    AddMacro("config_dir", AppEnv::ConfigDir());

    return ret;
}

bool ConfigData::Load(IBinStream &reader)
{
    groups_.clear();
    bool ret = true;

    String line_str;
    Section cur_section;
    ConfigItem cur_item;
    string cur_comments;

    while (ret && reader.GetLine(line_str))
    {
        line_str.Trim();
        if (line_str.size() <= 0 || line_str[0] == '#' || line_str[0] == ';')
        {
            if (cur_comments.size() > 0) {
                cur_comments += "\n";
            }
            cur_comments += line_str;
            continue;
        }

        if (IsGroupNameLine(line_str))
        {
            if (cur_section.items.size() > 0) {
                groups_[cur_section.name] = cur_section;
            }

            ret = ParseGroupLine(line_str, cur_section);
            cur_section.items.clear();
            cur_section.comments = cur_comments;
        }
        else
        {
            ret = ParseItemLine(line_str, cur_item);
            cur_item.comments = cur_comments;
            cur_item.order = (uint32_t)cur_section.items.size();
            cur_section.items[cur_item.key] = cur_item;
            macros_[cur_section.name + "/" + cur_item.key] = cur_item.value;
        }

        if (!ret) {
            LogWarning("Invalid line: %s", line_str.c_str());
        }
    }

    if (cur_section.items.size() > 0)
    {
        cur_section.order = (uint32_t)groups_.size();
        groups_[cur_section.name] = cur_section;
    }

    return ret;
}

bool ConfigData::Store(const std::string &strFile) const
{
    ofstream fo(strFile.c_str());
    if (!fo) {
        return false;
    }

    bool ret = Store(fo);
    if (ret)
    {
        fo.flush();
        ret = fo.good();
    }
    return ret;
}

bool ConfigData::Store(std::ostream &stream) const
{
    vector<const Section*> section_list;
    section_list.reserve(groups_.size());
    SectionMap::const_iterator iter_group = groups_.begin();
    for (; iter_group != groups_.end(); iter_group++) {
        section_list.push_back(&iter_group->second);
    }
    std::sort(section_list.begin(), section_list.end(), LessOrderSectionPtr);

    vector<const ConfigItem*> item_list;
    for (size_t section_idx = 0; section_idx < section_list.size(); section_idx++)
    {
        const Section &group = *section_list[section_idx];

        stream << "\n";
        if (!group.comments.empty()) {
            stream << group.comments << "\n";
        }
        stream << "[" << group.name << "]" << "\n";

        item_list.clear();
        item_list.reserve(group.items.size());
        ItemMap::const_iterator item_iter = group.items.begin();
        for (; item_iter != group.items.end(); item_iter++)
        {
            item_list.push_back(&item_iter->second);
        }
        std::sort(item_list.begin(), item_list.end(), LessOrderItemPtr);

        for (size_t iItem = 0; iItem < item_list.size(); iItem++)
        {
            const ConfigItem &item = *item_list[iItem];
            if (!item.comments.empty()) {
                stream << item.comments << "\n";
            }
            stream << item.key << " = " << item.value << "\n";
        }
    }

    bool ret = stream.good();
    return ret;
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    int &value, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr) {
        value = atoi(TransItemValue(item_ptr->value, section).c_str());
        return true;
    }
    else {
        return false;
    }
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    unsigned int &value, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr) {
        value = atoi(TransItemValue(item_ptr->value, section).c_str());
        return true;
    }
    else {
        return false;
    }
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    uint64_t &val, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr)
    {
        sscanf(TransItemValue(item_ptr->value, section).c_str(), "%ju", &val);
        //The following line generates wrong results for large uint64_t numbers
        //val = (uint64_t)_atoi64(TransItemValue(item_ptr->value).c_str());
        return true;
    }

    return false;
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    int64_t &val, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr)
    {
        val = (int64_t)atoll(TransItemValue(item_ptr->value, section).c_str());
        return true;
    }

    return false;
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    std::string &str_value, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr) {
        str_value = TransItemValue(item_ptr->value, section);
        return true;
    }
    else {
        return false;
    }
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    std::wstring &value, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr) {
        value = StringUtil::Utf8ToWideStr(TransItemValue(item_ptr->value, section));
        return true;
    }
    else {
        return false;
    }
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    float &value, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr) {
        value = (float)atof(TransItemValue(item_ptr->value, section).c_str());
        return true;
    }
    else {
        return false;
    }
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    double &value, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr) {
        value = atof(TransItemValue(item_ptr->value, section).c_str());
        return true;
    }
    else {
        return false;
    }
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    bool &value, bool log_if_not_exist) const
{
    const ConfigItem *item_ptr = GetItemInner(section, key, log_if_not_exist);
    if (item_ptr != nullptr)
    {
        string str_value = TransItemValue(item_ptr->value, section).c_str();
        value = false;
        if (strcasecmp(str_value.c_str(), "true") == 0
            || strcasecmp(str_value.c_str(), "yes") == 0
            || atoi(str_value.c_str()) != 0)
        {
            value = true;
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool ConfigData::GetItem(const std::string &section, const std::string &key,
    const std::map<std::string, uint32_t, StrLessNoCase> &value_map,
    const std::string &sep_str, uint32_t &value, bool log_if_not_exist) const
{
    string str_value;
    bool ret = GetItem(section, key, str_value, log_if_not_exist);
    if (!ret) {
        return false;
    }

    value = 0;
    vector<string> tokens;
    String::Tokenize(str_value, tokens, sep_str);
    for (size_t iToken = 0; iToken < tokens.size(); iToken++)
    {
        const string &token_str = tokens[iToken];
        std::map<std::string, uint32_t, StrLessNoCase>::const_iterator iter = value_map.find(token_str);
        if (iter != value_map.end())
        {
            value |= iter->second;
        }
        else
        {
            LogError("In section %s and item %s: Invalid token \"%s\"",
                section.c_str(), key.c_str(), token_str.c_str());
            return false;
        }
    }

    return ret;
}

bool ConfigData::GetItemList(vector<wstring> &item_list, const string &section,
    const string &name_prefix, bool log_if_not_exist) const
{
    char buf[4096];
    wstring value_str;
    int item_count = 0;
    bool ret = GetItem(section, name_prefix + "_count", item_count, log_if_not_exist);
    for (int idx = 0; ret && idx < item_count; idx++)
    {
        sprintf(buf, "%s%u", name_prefix.c_str(), idx + 1);
        ret = GetItem(section, buf, value_str, log_if_not_exist);
        if (!ret) {
            return false;
        }

        if (value_str.empty() || value_str[0] == L'-') {
            continue;
        }

        item_list.push_back(value_str);
    }

    return true;
}

bool ConfigData::GetItemList(vector<string> &item_list, const string &section,
    const string &name_prefix, bool log_if_not_exist) const
{
    char buf[4096];
    string value_str;
    int item_count = 0;
    bool ret = GetItem(section, name_prefix + "_count", item_count, log_if_not_exist);
    for (int idx = 0; ret && idx < item_count; idx++)
    {
        sprintf(buf, "%s%u", name_prefix.c_str(), idx + 1);
        ret = GetItem(section, buf, value_str, log_if_not_exist);
        if (!ret) {
            return false;
        }

        if (value_str.empty() || value_str[0] == '-') {
            continue;
        }

        item_list.push_back(value_str);
    }

    return true;
}

bool ConfigData::GetItems(const string &section, vector<ConfigItem> &item_list,
    bool log_if_not_exist) const
{
    item_list.clear();
    const Section *section_ptr = GetGroup(section);
    if (section_ptr == nullptr)
    {
        if (log_if_not_exist) {
            LogWarning("Section \"%s\" does not exist", section.c_str());
        }
        return false;
    }

    ConfigItem cur_item;
    ItemMap::const_iterator iter = section_ptr->items.begin();
    for (; iter != section_ptr->items.end(); iter++)
    {
        cur_item = iter->second;
        cur_item.value = TransItemValue(cur_item.value, section);
        item_list.push_back(cur_item);
    }
    return true;
}

bool ConfigData::GetItems(std::vector<std::wstring> &value_list, const string &section,
    const string &key_base, int from_idx, int to_idx, bool log_if_not_exist) const
{
    bool ret = true;
    value_list.clear();

    string value_str;
    char item_buf[256];
    int idx = from_idx;
    while (to_idx < 0 || idx <= to_idx)
    {
        snprintf(item_buf, 255, "%s%d", key_base.c_str(), idx);
        const ConfigItem *item_ptr = GetItemInner(section, item_buf, to_idx >= 0 ? true : log_if_not_exist);
        if (item_ptr != nullptr)
        {
            value_str = TransItemValue(item_ptr->value, section);
            value_list.push_back(StringUtil::Utf8ToWideStr(value_str));
        }
        else
        {
            if (to_idx < 0) {
                break;
            }
            else {
                ret = false;
            }
        }

        idx++;
    }

    return ret;
}

bool ConfigData::GetItems(const string &section, const string &key_base,
    int from_idx, int to_idx, std::vector<std::string> &value_list,
    bool log_if_not_exist) const
{
    bool ret = true;
    value_list.clear();

    char item_buf[256];
    int idx = from_idx;
    while (to_idx < 0 || idx <= to_idx)
    {
        snprintf(item_buf, 255, "%s%d", key_base.c_str(), idx);
        const ConfigItem *item_ptr = GetItemInner(section, item_buf, to_idx >= 0 ? true : log_if_not_exist);
        if (item_ptr != nullptr)
        {
            value_list.push_back(TransItemValue(item_ptr->value, section));
        }
        else
        {
            if (to_idx < 0) {
                break;
            }
            else {
                ret = false;
            }
        }

        idx++;
    }

    return ret;
}

bool ConfigData::GetItems(const string &section, const string &key_base, int from_idx, int to_idx,
    vector<int> &value_list, bool log_if_not_exist) const
{
    bool ret = true;
    value_list.clear();

    char item_buf[256];
    int idx = from_idx;
    while (to_idx < 0 || idx <= to_idx)
    {
        snprintf(item_buf, 255, "%s%d", key_base.c_str(), idx);
        const ConfigItem *item_ptr = GetItemInner(section, item_buf, to_idx >= 0 ? true : log_if_not_exist);
        if (item_ptr != nullptr)
        {
            value_list.push_back(atoi(TransItemValue(item_ptr->value, section).c_str()));
        }
        else
        {
            if (to_idx < 0) {
                break;
            }
            else {
                ret = false;
            }
        }

        idx++;
    }

    return ret;
}

bool ConfigData::HasSection(const string &name) const
{
    SectionMap::const_iterator iter_group = groups_.find(name);
    return iter_group != groups_.end();
}

bool ConfigData::GetSections(vector<SectionInfo> &section_list) const
{
    section_list.clear();
    SectionInfo sec;
    SectionMap::const_iterator iter = groups_.begin();
    for (; iter != groups_.end(); iter++)
    {
        sec.name = iter->second.name;
        sec.comments = iter->second.comments;
        sec.order = iter->second.order;
        section_list.push_back(sec);
    }

    return true;
}

bool ConfigData::AddSection(const string &name, const string &comments, uint32_t order)
{
    Section group;
    SectionMap::iterator iter_group = groups_.find(name);
    if (iter_group != groups_.end()) {
        return false;
    }

    if (!comments.empty()) {
        group.comments = ";" + comments;
    }
    group.order = order != UINT32_MAX ? order : (uint32_t)groups_.size();
    groups_[name] = group;
    return true;
}

bool ConfigData::SetItem(const string &section, const string &key, int value,
    bool ddd_if_not_exist, const string &comments, uint32_t order)
{
    ConfigItem *item_ptr = GetItemInner(section, key, false);
    if (item_ptr == nullptr && ddd_if_not_exist) {
        item_ptr = AddItem(section, key, order);
    }

    if (item_ptr != nullptr)
    {
        char buf[256];
        if (snprintf(buf, 255, "%d", value) == -1) {
            return false;
        }
        item_ptr->value = buf;
        if (!comments.empty()) {
            item_ptr->comments = ";" + comments;
        }
        if (order != UINT32_MAX) {
            item_ptr->order = order;
        }
        return true;
    }

    return false;
}

bool ConfigData::SetItem(const string &section, const string &key, unsigned int value,
    bool ddd_if_not_exist, const string &comments, uint32_t order)
{
    ConfigItem *item_ptr = GetItemInner(section, key, false);
    if (item_ptr == nullptr && ddd_if_not_exist) {
        item_ptr = AddItem(section, key, order);
    }

    if (item_ptr != nullptr)
    {
        char buf[256];
        if (snprintf(buf, 255, "%d", value) == -1) {
            return false;
        }
        item_ptr->value = buf;
        if (!comments.empty()) {
            item_ptr->comments = ";" + comments;
        }
        if (order != UINT32_MAX) {
            item_ptr->order = order;
        }
        return true;
    }

    return false;
}

bool ConfigData::SetItem(const string &section, const string &key, long value,
    bool ddd_if_not_exist, const string &comments, uint32_t order)
{
    ConfigItem *item_ptr = GetItemInner(section, key, false);
    if (item_ptr == nullptr && ddd_if_not_exist) {
        item_ptr = AddItem(section, key, order);
    }

    if (item_ptr != nullptr)
    {
        char buf[256];
        if (snprintf(buf, 255, "%ld", value) == -1) {
            return false;
        }
        item_ptr->value = buf;
        if (!comments.empty()) {
            item_ptr->comments = ";" + comments;
        }
        if (order != UINT32_MAX) {
            item_ptr->order = order;
        }
        return true;
    }

    return false;
}

bool ConfigData::SetItem(const string &section, const string &key, const string &str_value,
    bool ddd_if_not_exist, const std::string &comments, uint32_t order)
{
    ConfigItem *item_ptr = GetItemInner(section, key, false);
    if (item_ptr == nullptr && ddd_if_not_exist) {
        item_ptr = AddItem(section, key, order);
    }

    if (item_ptr != nullptr)
    {
        item_ptr->value = str_value;
        if (!comments.empty()) {
            item_ptr->comments = ";" + comments;
        }
        if (order != UINT32_MAX) {
            item_ptr->order = order;
        }
        return true;
    }

    return false;
}

void ConfigData::EnableMacros(bool enable_explicit, bool enable_implicit)
{
    enable_explicit_macros_ = enable_explicit;
    enable_implicit_macros_ = enable_implicit;
}

bool ConfigData::AddMacro(const string &macro_name, const string &macro_value)
{
    if (macro_name.empty()) {
        return false;
    }
    macros_[macro_name] = macro_value;
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Non-public methods
///////////////////////////////////////////////////////////////////////////////

bool ConfigData::IsGroupNameLine(const string &line_str) const
{
    if (line_str[0] == '[' && line_str.find(']') != string::npos) {
        return true;
    }
    return false;
}

bool ConfigData::ParseGroupLine(const std::string& line_str, Section &group)
{
    string::size_type begin_pos = line_str.find("[");
    string::size_type end_pos = line_str.find("]");

    if (begin_pos != string::npos && end_pos != string::npos && end_pos > begin_pos) {
        group.name = line_str.substr(begin_pos + 1, end_pos - begin_pos - 1);
    }

    return true;
}

bool ConfigData::ParseItemLine(const string &line_str, ConfigItem &item)
{
    string::size_type pos = line_str.find("=");
    if (pos == string::npos) {
        return false;
    }

    String key = line_str.substr(0, pos);
    String str_value = line_str.substr(pos + 1);

    key.Trim();
    item.key = key;
    str_value.Trim();
    item.value = str_value;

    return key.size() > 0;
}

const ConfigData::Section* ConfigData::GetGroup(const string &section) const
{
    SectionMap::const_iterator iter_group = groups_.find(section);
    if (iter_group == groups_.end()) {
        return nullptr;
    }

    const Section *group_ptr = &iter_group->second;
    return group_ptr;
}

const ConfigData::ConfigItem* ConfigData::GetItemInner(const string &section,
    const string &key, bool log_if_not_exist) const
{
    SectionMap::const_iterator iter_group = groups_.find(section);
    if (iter_group == groups_.end())
    {
        if (log_if_not_exist) {
            LogWarning("Group does not exist: %s", section.c_str());
        }
        return nullptr;
    }
    const Section *group_ptr = &iter_group->second;

    ItemMap::const_iterator item_iter = group_ptr->items.find(key);
    if (item_iter == group_ptr->items.end())
    {
        if (log_if_not_exist) {
            LogWarning("Item \"%s\" does not exist in group \"%s\"",
                key.c_str(), section.c_str());
        }
        return nullptr;
    }
    const ConfigItem *item_ptr = &item_iter->second;

    return item_ptr;
}

ConfigData::ConfigItem* ConfigData::GetItemInner(const string &section,
    const string &key, bool log_if_not_exist)
{
    SectionMap::iterator iter_group = groups_.find(section);
    if (iter_group == groups_.end())
    {
        if (log_if_not_exist) {
            LogWarning("Group does not exist: %s", section.c_str());
        }
        return nullptr;
    }
    Section *group_ptr = &iter_group->second;

    ItemMap::iterator item_iter = group_ptr->items.find(key);
    if (item_iter == group_ptr->items.end())
    {
        if (log_if_not_exist) {
            LogWarning("Item \"%s\" does not exist in group \"%s\"",
                key.c_str(), section.c_str());
        }
        return nullptr;
    }
    ConfigItem *item_ptr = &item_iter->second;

    return item_ptr;
}

ConfigData::ConfigItem* ConfigData::AddItem(const string &section,
    const string &key, uint32_t order)
{
    Section *group_ptr = nullptr;
    SectionMap::iterator iter_group = groups_.find(section);
    if (iter_group == groups_.end())
    {
        Section group(section);
        pair<SectionMap::iterator, bool> ib = groups_.insert(make_pair(section, group));
        if (ib.second) {
            group_ptr = &ib.first->second;
        }
    }
    else
    {
        group_ptr = &iter_group->second;
    }

    ConfigItem *item_ptr = nullptr;
    if (group_ptr != nullptr)
    {
        ItemMap::iterator item_iter = group_ptr->items.find(key);
        if (item_iter == group_ptr->items.end())
        {
            ConfigItem item(key);
            pair<ItemMap::iterator, bool> ib = group_ptr->items.insert(make_pair(key, item));
            if (ib.second) {
                item_ptr = &ib.first->second;
                item_ptr->order = order != UINT32_MAX ? order : (uint32_t)group_ptr->items.size();
            }
        }
    }

    return item_ptr;
}

std::string ConfigData::TransItemValue(const string &item_value,
    const string &group_name, int depth) const
{
    string ret_str = item_value;
    if (enable_explicit_macros_ && !macros_.empty())
    {
        ret_str.clear();
        bool is_in_macro = false;
        string cur_macro_value;
        size_t value_size = item_value.size();
        for (size_t char_idx = 0; char_idx < value_size; char_idx++)
        {
            char ch = item_value[char_idx];
            if (is_in_macro)
            {
                if (ch == ')' || ch == '}')
                {
                    MacroMap::const_iterator iterMacro = macros_.find(cur_macro_value);
                    if (iterMacro == macros_.end()) {
                        iterMacro = macros_.find(group_name + "/" + cur_macro_value);
                    }

                    if (iterMacro != macros_.end())
                    {
                        if (depth < 10)
                        {
                            ret_str += TransItemValue(iterMacro->second, group_name, depth++);
                        }
                        else
                        {
                            ret_str += iterMacro->second;
                        }
                    }
                    else
                    {
                        ret_str += cur_macro_value;
                    }
                    cur_macro_value.clear();
                    is_in_macro = false;
                }
                else
                {
                    cur_macro_value += ch;
                }
            }
            else
            {
                if (ch == '$' && char_idx + 1 < value_size
                    && (item_value[char_idx + 1] == '(' || item_value[char_idx + 1] == '{'))
                {
                    char_idx++;
                    is_in_macro = true;
                }
                else {
                    ret_str += ch;
                }
            }
        }
    }

    return ret_str;
}

} //end of namespace
