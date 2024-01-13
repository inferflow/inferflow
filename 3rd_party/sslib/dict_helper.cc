#include "dict_helper.h"

namespace sslib
{

//static
bool DictHelper::LoadDict(StrDict &dict, const string &path, const string &dict_name)
{
    LogKeyInfo("Loading the %s dict...", dict_name.c_str());
    bool ret = dict.Load(path);
    if(!ret) {
        LogError("Failed to load the dict from %s", path.c_str());
        return false;
    }

    return ret;
}

//static
bool DictHelper::LoadDict(StringDict<uint32_t> &dict,
    const string &path, const DictHelper::MessageSpec &spec)
{
    if (spec.show_info) {
        LogKeyInfo("Loading the %s dict...", spec.dict_name.c_str());
    }
    bool ret = dict.Load(path);
    if (!ret)
    {
        if (spec.show_error) {
            LogError("Failed to load the dict from %s", path.c_str());
        }
        return false;
    }

    return ret;
}

//static
bool DictHelper::LoadDict(StringDict<float> &dict,
    const string &path, const DictHelper::MessageSpec &spec)
{
    if (spec.show_info) {
        LogKeyInfo("Loading the %s dict...", spec.dict_name.c_str());
    }
    bool ret = dict.Load(path);
    if (!ret)
    {
        if (spec.show_error) {
            LogError("Failed to load the dict from %s", path.c_str());
        }
        return false;
    }

    return ret;
}

//static
bool DictHelper::LoadDict(WStringDict<uint32_t> &dict,
    const string &path, const DictHelper::MessageSpec &spec)
{
    if (spec.show_info) {
        LogKeyInfo("Loading the %s dict...", spec.dict_name.c_str());
    }
    bool ret = dict.Load(path);
    if (!ret)
    {
        if (spec.show_error) {
            LogError("Failed to load the dict from %s", path.c_str());
        }
        return false;
    }

    return ret;
}

//static
bool DictHelper::LoadDict(WStringDict<float> &dict,
    const string &path, const DictHelper::MessageSpec &spec)
{
    if (spec.show_info) {
        LogKeyInfo("Loading the %s dict...", spec.dict_name.c_str());
    }
    bool ret = dict.Load(path);
    if (!ret)
    {
        if (spec.show_error) {
            LogError("Failed to load the dict from %s", path.c_str());
        }
        return false;
    }

    return ret;
}

//static
bool DictHelper::LoadDict(StringDict<uint32_t> &dict, const string &path, const string &dict_name)
{
    MessageSpec spec(dict_name);
    return LoadDict(dict, path, spec);
}

//static
bool DictHelper::LoadDict(WStrDict &dict, const string &path, const string &dict_name)
{
    LogKeyInfo("Loading the %s dict...", dict_name.c_str());
    bool ret = dict.Load(path);
    if(!ret) {
        LogError("Failed to load the dict from %s", path.c_str());
        return false;
    }

    return ret;
}

//static
bool DictHelper::LoadDict(WStringDict<uint32_t> &dict, const string &path, const string &dict_name)
{
    LogKeyInfo("Loading the %s dict...", dict_name.c_str());
    bool ret = dict.Load(path);
    if (!ret) {
        LogError("Failed to load the dict from %s", path.c_str());
        return false;
    }

    return ret;
}

//static
bool DictHelper::SaveDict(const StrDict &dict, const string &path, const string &dict_name)
{
    LogKeyInfo("Saving the %s dict...", dict_name.c_str());
    bool ret = dict.Store(path);
    if(!ret) {
        LogError("Failed to save the dict to %s", path.c_str());
        return false;
    }

    return ret;
}

//static
bool DictHelper::SaveDict(const WStrDict &dict, const string &path, const string &dict_name)
{
    LogKeyInfo("Saving the %s dict...", dict_name.c_str());
    bool ret = dict.Store(path);
    if(!ret) {
        LogError("Failed to save the dict to %s", path.c_str());
        return false;
    }

    return ret;
}

//static
bool DictHelper::SaveDict(const StringDict<uint32_t> &dict, const string &path, const string &dict_name)
{
    LogKeyInfo("Saving the %s dict...", dict_name.c_str());
    bool ret = dict.Store(path);
    if(!ret) {
        LogError("Failed to save the dict to %s", path.c_str());
        return false;
    }

    return ret;
}

//static
bool DictHelper::SaveDict(const WStringDict<uint32_t> &dict, const string &path, const string &dict_name)
{
    LogKeyInfo("Saving the %s dict...", dict_name.c_str());
    bool ret = dict.Store(path);
    if (!ret) {
        LogError("Failed to save the dict to %s", path.c_str());
        return false;
    }

    return ret;
}

} //end of namespace
