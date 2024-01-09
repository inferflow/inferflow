#pragma once

#include "string_dict.h"

namespace sslib
{

class DictHelper
{
public:
    struct MessageSpec
    {
        string dict_name;
        bool show_info = true;
        bool show_error = true;

        MessageSpec(const string &p_dict_name = "", bool p_show_info = true, bool p_show_error = true)
        {
            dict_name = p_dict_name;
            show_info = p_show_info;
            show_error = p_show_error;
        }
    };

public:
    static bool LoadDict(StringDict<uint32_t> &dict, const string &path, const MessageSpec &spec);
    static bool LoadDict(StringDict<float> &dict, const string &path, const MessageSpec &spec);
    static bool LoadDict(WStringDict<uint32_t> &dict, const string &path, const MessageSpec &spec);
    static bool LoadDict(WStringDict<float> &dict, const string &path, const MessageSpec &spec);

    static bool LoadDict(StrDict &dict, const string &path, const string &dict_name);
    static bool LoadDict(StringDict<uint32_t> &dict, const string &path, const string &dict_name);
    static bool LoadDict(WStrDict &dict, const string &path, const string &dict_name);
    static bool LoadDict(WStringDict<uint32_t> &dict, const string &path, const string &dict_name);

    static bool SaveDict(const StrDict &dict, const string &path, const string &dict_name);
    static bool SaveDict(const WStrDict &dict, const string &path, const string &dict_name);
    static bool SaveDict(const StringDict<uint32_t> &dict, const string &path, const string &dict_name);
    static bool SaveDict(const WStringDict<uint32_t> &dict, const string &path, const string &dict_name);

    template <class T1, class T2>
    static void AssignDict(WStringDict<T1> &dst, const WStringDict<T2> &src)
    {
        dst.Clear();
        uint32_t term_count = src.Size();
        for (uint32_t term_id = 0; term_id < term_count; term_id++)
        {
            const auto *item_ptr = src.Get(term_id);
            dst.AddItem(term_id, item_ptr->str, item_ptr->weight, false);
        }
    }
};

} //end of namespace
