#include "json.h"
#include "string_util.h"
#include <algorithm>

namespace sslib
{

using namespace std;

JsonValue::JsonValue()
{
    type_ = JsonValueType::Null;
    size_ = 0;
    ptr_ = nullptr;
}

bool JsonValue::GetJObject(JsonObject &obj) const
{
    Macro_RetFalseIf(type_ != JsonValueType::Object);

    obj.id = id_;
    obj.size = size_;
    obj.items = entry_;
    obj.hit = hit_;
    return true;
}

JsonObject JsonValue::GetJObject() const
{
    JsonObject obj;
    if (type_ == JsonValueType::Object)
    {
        obj.id = id_;
        obj.size = size_;
        obj.items = entry_;
        obj.hit = hit_;
    }
    else
    {
        obj.id = Number::MaxUInt32;
        obj.size = 0;
        obj.items = nullptr;
        obj.hit.Set(0, 0);
    }
    return obj;
}

bool JsonValue::GetArray(JsonArray &arr) const
{
    Macro_RetFalseIf(type_ != JsonValueType::Array);

    arr.size = size_;
    arr.items = (JsonValue*)ptr_;
    arr.hit = hit_;
    return true;
}

bool JsonValue::GetValueList(std::vector<JsonValue> &value_list) const
{
    value_list.clear();
    if (type_ == JsonValueType::Array)
    {
        for (uint32_t idx = 0; idx < size_; idx++) {
            value_list.push_back(value_[idx]);
        }
    }
    else
    {
        value_list.push_back(*this);
    }

    return true;
}

bool JsonValue::GetString(JsonString &str) const
{
    Macro_RetFalseIf(type_ != JsonValueType::String
        && type_ != JsonValueType::Utf8String);

    str.len = size_;
    str.is_wide = type_ == JsonValueType::String;
    if (str.is_wide)
    {
        str.data = str_;
        str.utf8_data = nullptr;
    }
    else
    {
        str.data = nullptr;
        str.utf8_data = utf8_str_;
    }

    return true;
}

bool JsonValue::GetString(wstring &str) const
{
    str.clear();

    JsonString jstr;
    bool ret = GetString(jstr);
    Macro_RetFalseIf(!ret);

    jstr.ToString(str);
    return true;
}

bool JsonValue::GetString(string &str) const
{
    str.clear();

    JsonString jstr;
    bool ret = GetString(jstr);
    Macro_RetFalseIf(!ret);

    jstr.ToString(str);
    return true;
}

bool JsonValue::GetNumber(MathDecimal &num) const
{
    Macro_RetFalseIf(type_ != JsonValueType::Number);

    num.Set(data_, (int16_t)size_);
    return true;
}

MathDecimal JsonValue::GetNumber() const
{
    MathDecimal num;
    if (type_ == JsonValueType::Number)
    {
        num.Set(data_, (UInt16)size_);
    }

    return num;
}

int JsonValue::GetIntValue() const
{
    MathDecimal num;
    if (type_ == JsonValueType::Number)
    {
        num.Set(data_, (UInt16)size_);
    }

    return num.ToInt32();
}

uint32_t JsonValue::GetUInt32Value() const
{
    MathDecimal num;
    if (type_ == JsonValueType::Number)
    {
        num.Set(data_, (UInt16)size_);
    }

    return (uint32_t)num.ToInt32();
}

Int64 JsonValue::GetInt64Value() const
{
    MathDecimal num;
    if (type_ == JsonValueType::Number)
    {
        num.Set(data_, (UInt16)size_);
    }

    return num.ToInt64();
}

UInt64 JsonValue::GetUInt64Value() const
{
    MathDecimal num;
    if (type_ == JsonValueType::Number)
    {
        num.Set(data_, (UInt16)size_);
    }

    return (UInt64)num.ToInt64();
}

float JsonValue::GetFloatValue() const
{
    MathDecimal num;
    if (type_ == JsonValueType::Number)
    {
        num.Set(data_, (UInt16)size_);
    }

    return (float)num.ToDouble();
}

bool JsonValue::GetBoolValue() const
{
    if (type_ == JsonValueType::Number || type_ == JsonValueType::Boolean)
    {
        return data_ != 0;
    }

    return false;
}

void JsonValue::SetObject(const JsonObject &obj)
{
    type_ = JsonValueType::Object;
    id_ = obj.id;
    size_ = obj.size;
    ptr_ = obj.items;
}

void JsonValue::SetString(const JsonString &jstr)
{
    type_ = jstr.is_wide ? JsonValueType::String : JsonValueType::Utf8String;
    size_ = jstr.len;
    if (jstr.is_wide) {
        str_ = jstr.data;
    }
    else {
        utf8_str_ = jstr.utf8_data;
    }
}

void JsonValue::SetNumber(const MathDecimal &num)
{
    type_ = JsonValueType::Number;
    size_ = (unsigned int)num.GetExp();
    data_ = num.GetSig();
}

void JsonValue::SetBool(bool bTrue)
{
    type_ = JsonValueType::Boolean;
    data_ = bTrue ? 1 : 0;
}

void JsonValue::SetNull()
{
    type_ = JsonValueType::Null;
}

////////////////////////////////////////////////////////////////////////////////
// class JsonObject
////////////////////////////////////////////////////////////////////////////////

const JsonObjectEntry* JsonObject::GetFieldValue(
    const wstring &fld_name, const JsonDoc &doc) const
{
    uint32_t fld_name_id = doc.str_dict_.ItemId(fld_name);
    const auto *cell_ptr = doc.matrix_.GetCell(id, fld_name_id);
    return cell_ptr != nullptr && cell_ptr->weight < size ? &items[cell_ptr->weight] : nullptr;
}

//keep: keep value if the field name is not available
bool JsonObject::GetFieldValue(wstring &fld_value, const wstring &fld_name,
    const JsonDoc &doc, bool keep) const
{
    //keep the value if the field name is not available
    if (!keep) {
        fld_value.clear();
    }

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    return fld->value.GetString(fld_value);
}

//keep: keep value if the field name is not available
bool JsonObject::GetFieldValue(string &fld_value, const wstring &fld_name,
    const JsonDoc &doc, bool keep) const
{
    //keep the value if the field name is not available
    if (!keep) {
        fld_value.clear();
    }

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    return fld->value.GetString(fld_value);
}

//keep: keep value if the field name is not available
bool JsonObject::GetFieldValue(int &fld_value, const wstring &fld_name,
    const JsonDoc &doc, bool keep) const
{
    //keep the value if the field name is not available
    if (!keep) {
        fld_value = 0;
    }

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    fld_value = fld->value.GetIntValue();
    return true;
}

//keep: keep value if the field name is not available
bool JsonObject::GetFieldValue(uint32_t &fld_value, const wstring &fld_name,
    const JsonDoc &doc, bool keep) const
{
    //keep the value if the field name is not available
    if (!keep) {
        fld_value = 0;
    }

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    fld_value = fld->value.GetUInt32Value();
    return true;
}

bool JsonObject::GetFieldValue(UInt64 &fld_value, const wstring &fld_name,
    const JsonDoc &doc, bool keep) const
{
    //keep the value if the field name is not available
    if (!keep) {
        fld_value = 0;
    }

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    fld_value = fld->value.GetUInt64Value();
    return true;
}

//keep: keep value if the field name is not available
bool JsonObject::GetFieldValue(float &fld_value, const wstring &fld_name,
    const JsonDoc &doc, bool keep) const
{
    //keep the value if the field name is not available
    if (!keep) {
        fld_value = 0;
    }

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    fld_value = fld->value.GetFloatValue();
    return true;
}

//keep: keep value if the field name is not available
bool JsonObject::GetFieldValue(bool &fld_value, const wstring &fld_name,
    const JsonDoc &doc, bool keep) const
{
    //keep the value if the field name is not available
    if (!keep) {
        fld_value = false;
    }

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    fld_value = fld->value.GetBoolValue();
    return true;
}

bool JsonObject::GetFieldValue(JsonArray &fld_value, const wstring &fld_name, const JsonDoc &doc) const
{
    fld_value.size = 0;
    fld_value.items = nullptr;

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    return fld->value.GetArray(fld_value);
}

bool JsonObject::GetFieldValue(JsonObject &fld_value, const wstring &fld_name, const JsonDoc &doc) const
{
    fld_value.size = 0;
    fld_value.items = nullptr;

    const auto *fld = GetFieldValue(fld_name, doc);
    if (fld == nullptr) {
        return false;
    }

    return fld->value.GetJObject(fld_value);
}

////////////////////////////////////////////////////////////////////////////////
// class JsonDoc
////////////////////////////////////////////////////////////////////////////////

JsonDoc::JsonDoc()
    : str_dict_(256, false)
    , matrix_(1024)
{
}

JsonDoc::~JsonDoc()
{
    Clear();
}

void JsonDoc::Clear()
{
    Free((JsonValue&)(*this));
    str_dict_.Clear();
    matrix_.Clear();
}

//static
void JsonDoc::Free(JsonString &jstr)
{
    if (jstr.len > 0 && jstr.data != nullptr) {
        delete[] jstr.data;
    }
    if (jstr.len > 0 && jstr.utf8_data != nullptr) {
        delete[] jstr.utf8_data;
    }
    jstr.data = nullptr;
    jstr.utf8_data = nullptr;
    jstr.len = 0;
}

//static
void JsonDoc::Free(JsonValue &jval)
{
    if (jval.size_ > 0 && jval.ptr_ != nullptr)
    {
        switch (jval.type_)
        {
        case JsonValueType::String:
            delete[] jval.str_;
            jval.str_ = nullptr;
            break;
        case JsonValueType::Utf8String:
            delete[] jval.utf8_str_;
            jval.utf8_str_ = nullptr;
            break;
        case JsonValueType::Array:
            {
                JsonValue *arr_value = (JsonValue*)jval.ptr_;
                for (unsigned int idx = 0; idx < jval.size_; idx++) {
                    Free(arr_value[idx]);
                }
                delete[] arr_value;
            }
            break;
        case JsonValueType::Object:
            {
                JsonObjectEntry *arr_entry = (JsonObjectEntry*)jval.ptr_;
                for (unsigned int idx = 0; idx < jval.size_; idx++) {
                    Free(arr_entry[idx]);
                }
                delete arr_entry;
            }
            break;
        default:
            break;
        }
    }

    jval.ptr_ = nullptr;
    jval.size_ = 0;
    jval.type_ = JsonValueType::Null;
}

//static
void JsonDoc::Free(JsonObject &jobj)
{
    if (jobj.size > 0 && jobj.items != nullptr)
    {
        for (unsigned int idx = 0; idx < jobj.size; idx++) {
            Free(jobj.items[idx]);
        }
        delete[] jobj.items;
        jobj.items = nullptr;
    }

    jobj.size = 0;
}

//static
void JsonDoc::Free(JsonObjectEntry &jentry)
{
    Free(jentry.name);
    Free(jentry.value);
}

//static
void JsonDoc::Free(std::vector<JsonObjectEntry> &entry_list)
{
    for (size_t idx = 0; idx < entry_list.size(); idx++) {
        Free(entry_list[idx]);
    }
}

//static
void JsonDoc::Free(std::vector<JsonValue> &value_list)
{
    for (size_t idx = 0; idx < value_list.size(); idx++) {
        Free(value_list[idx]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// class JsonParser
////////////////////////////////////////////////////////////////////////////////

JsonParser::JsonParser()
{
}

JsonParser::~JsonParser()
{
}

bool JsonParser::Init()
{
    bool bRet = unicode_table_.Init();
    is_initialized_ = bRet;
    return bRet;
}

bool JsonParser::Parse(JsonDoc &output, const wstring &input_str) const
{
    return Parse(output, input_str.c_str(), input_str.size());
}

bool JsonParser::Parse(JsonDoc &output, const wchar_t *input_str,
    size_t input_len) const
{
    Macro_RetFalseIf(!is_initialized_);

    output.Clear();
    size_t offset_ret = ParseValue(output, output, input_str, input_len, 0);
    Macro_RetFalseIf(offset_ret == 0);

    //skip separators
    offset_ret = SkipSeparators(input_str, input_len, offset_ret);
    return offset_ret == input_len;
}

bool JsonParser::ParseUtf8(JsonDoc &output, const string &input_str) const
{
    return ParseUtf8(output, input_str.c_str(), input_str.size());
}

bool JsonParser::ParseUtf8(JsonDoc &output, const char *input_str,
    size_t input_len) const
{
    Macro_RetFalseIf(!is_initialized_);

    output.Clear();
    size_t offset_ret = ParseValue(output, output, input_str, input_len, 0);
    Macro_RetFalseIf(offset_ret == 0);

    //skip separators
    offset_ret = SkipSeparators(input_str, input_len, offset_ret);
    return offset_ret == input_len;
}

//static
void JsonParser::DecodeString(wstring &output, const wstring &input)
{
    wstring str = L"\"";
    str += input;
    str += L"\"";
    ParseString(output, str.c_str(), str.size(), 0);
}

//static
void JsonParser::DecodeString(string &output, const string &input)
{
    string str = "\"";
    str += input;
    str += "\"";
    ParseString(output, str.c_str(), str.size(), 0);
}

size_t JsonParser::ParseValue(JsonValue &output, JsonDoc &doc,
    const wchar_t *input_str, size_t input_len, size_t offset) const
{
    size_t new_offset = SkipSeparators(input_str, input_len, offset);
    Macro_RetIf(offset, new_offset >= input_len);

    output.hit_.Set(new_offset, new_offset);
    size_t offset_ret = offset;
    wchar_t ch = input_str[new_offset];
    switch (ch)
    {
    case L'{':
        {
            JsonObject obj;
            offset_ret = ParseObject(obj, doc, input_str, input_len, new_offset);
            Macro_RetIf(offset, offset_ret == new_offset);
            output.SetObject(obj);
        }
        break;
    case L'[':
        offset_ret = ParseArray(output, doc, input_str, input_len, new_offset);
        Macro_RetIf(offset, offset_ret == new_offset);
        break;
    case L'"':
        {
            JsonString jstr;
            offset_ret = ParseString(jstr, input_str, input_len, new_offset);
            Macro_RetIf(offset, offset_ret == new_offset);
            output.SetString(jstr);
        }
        break;
    case L't': //true
    case L'T':
    case L'f': //false
    case L'F':
    case L'n': //null
    case L'N':
        offset_ret = ParseConstValue(output, input_str, input_len, new_offset);
        Macro_RetIf(offset, offset_ret == new_offset);
        break;
    default:
        MathDecimal num;
        offset_ret = ParseNumber(num, input_str, input_len, new_offset);
        output.SetNumber(num);
        break;
    }

    output.hit_.end = offset_ret;
    return offset_ret;
}

size_t JsonParser::ParseValue(JsonValue &output, JsonDoc &doc,
    const char *input_str, size_t input_len, size_t offset) const
{
    size_t new_offset = SkipSeparators(input_str, input_len, offset);
    Macro_RetIf(offset, new_offset >= input_len);

    output.hit_.Set(new_offset, new_offset);
    size_t offset_ret = offset;
    char ch = input_str[new_offset];
    switch (ch)
    {
    case '{':
    {
        JsonObject obj;
        offset_ret = ParseObject(obj, doc, input_str, input_len, new_offset);
        Macro_RetIf(offset, offset_ret == new_offset);
        output.SetObject(obj);
    }
    break;
    case '[':
        offset_ret = ParseArray(output, doc, input_str, input_len, new_offset);
        Macro_RetIf(offset, offset_ret == new_offset);
        break;
    case '"':
    {
        JsonString jstr;
        offset_ret = ParseString(jstr, input_str, input_len, new_offset);
        Macro_RetIf(offset, offset_ret == new_offset);
        output.SetString(jstr);
    }
    break;
    case 't': //true
    case 'T':
    case 'f': //false
    case 'F':
    case 'n': //null
    case 'N':
        offset_ret = ParseConstValue(output, input_str, input_len, new_offset);
        Macro_RetIf(offset, offset_ret == new_offset);
        break;
    default:
        MathDecimal num;
        offset_ret = ParseNumber(num, input_str, input_len, new_offset);
        output.SetNumber(num);
        break;
    }

    output.hit_.end = offset_ret;
    return offset_ret;
}

size_t JsonParser::ParseObject(JsonObject &output, JsonDoc &doc,
    const wchar_t *input_str, size_t input_len, size_t offset) const
{
    output.size = 0;
    output.items = nullptr;
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != L'{');

    size_t new_offset = SkipSeparators(input_str, input_len, offset+1);
    Macro_RetIf(new_offset + 1, new_offset < input_len && input_str[new_offset] == L'}');

    vector<JsonObjectEntry> entry_list;
    JsonObjectEntry entry;
    bool is_succ = true, be_need_more_entry = true;
    size_t offset_ret = offset;
    while (is_succ && be_need_more_entry && new_offset < input_len)
    {
        entry.hit.Set(new_offset, new_offset);
        offset_ret = ParseString(entry.name, input_str, input_len, new_offset);
        if (offset_ret == new_offset) {
            is_succ = false;
            break;
        }

        new_offset = SkipSeparators(input_str, input_len, offset_ret);
        if (new_offset >= input_len || input_str[new_offset] != L':') {
            is_succ = false;
            break;
        }

        new_offset++; //skip the ':' character
        offset_ret = ParseValue(entry.value, doc, input_str, input_len, new_offset);
        if (offset_ret == new_offset)
        {
            JsonDoc::Free(entry.name);
            is_succ = false;
            break;
        }

        entry.hit.end = offset_ret;
        entry_list.push_back(entry);
        new_offset = SkipSeparators(input_str, input_len, offset_ret);
        if (new_offset >= input_len) {
            is_succ = false;
            break;
        }

        if (input_str[new_offset] == L'}')
        {
            be_need_more_entry = false;
            offset_ret = new_offset + 1;
        }
        else if (input_str[new_offset] == L',')
        {
            be_need_more_entry = true;
            new_offset++;
            new_offset = SkipSeparators(input_str, input_len, new_offset);
        }
        else
        {
            is_succ = false;
        }
    }

    if (!is_succ || be_need_more_entry || entry_list.empty())
    {
        JsonDoc::Free(entry_list);
        return offset;
    }

    uint32_t fld_name_id = 0;
    wstring fld_name;
    output.id = doc.NextObjectId();

    output.size = (unsigned int)entry_list.size();
    output.items = new JsonObjectEntry[output.size];
    for (unsigned int idx = 0; idx < output.size; idx++)
    {
        output.items[idx] = entry_list[idx];
        output.items[idx].name.ToString(fld_name);
        doc.str_dict_.AddItem(fld_name, 0, fld_name_id, 0);
        doc.matrix_.AddCell(output.id, fld_name_id, idx);
    }

    output.hit.end = offset_ret;
    return offset_ret;
}

size_t JsonParser::ParseObject(JsonObject &output, JsonDoc &doc,
    const char *input_str, size_t input_len, size_t offset) const
{
    output.size = 0;
    output.items = nullptr;
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != '{');

    size_t new_offset = SkipSeparators(input_str, input_len, offset + 1);
    Macro_RetIf(new_offset + 1, new_offset < input_len && input_str[new_offset] == '}');

    vector<JsonObjectEntry> entry_list;
    JsonObjectEntry entry;
    bool is_succ = true, be_need_more_entry = true;
    size_t offset_ret = offset;
    while (is_succ && be_need_more_entry && new_offset < input_len)
    {
        entry.hit.Set(new_offset, new_offset);
        offset_ret = ParseString(entry.name, input_str, input_len, new_offset);
        if (offset_ret == new_offset) {
            is_succ = false;
            break;
        }

        new_offset = SkipSeparators(input_str, input_len, offset_ret);
        if (new_offset >= input_len || input_str[new_offset] != ':') {
            is_succ = false;
            break;
        }

        new_offset++; //skip the ':' character
        offset_ret = ParseValue(entry.value, doc, input_str, input_len, new_offset);
        if (offset_ret == new_offset)
        {
            JsonDoc::Free(entry.name);
            is_succ = false;
            break;
        }

        entry.hit.end = offset_ret;
        entry_list.push_back(entry);
        new_offset = SkipSeparators(input_str, input_len, offset_ret);
        if (new_offset >= input_len) {
            is_succ = false;
            break;
        }

        if (input_str[new_offset] == '}')
        {
            be_need_more_entry = false;
            offset_ret = new_offset + 1;
        }
        else if (input_str[new_offset] == ',')
        {
            be_need_more_entry = true;
            new_offset++;
            new_offset = SkipSeparators(input_str, input_len, new_offset);
        }
        else
        {
            is_succ = false;
        }
    }

    if (!is_succ || be_need_more_entry || entry_list.empty())
    {
        JsonDoc::Free(entry_list);
        return offset;
    }

    uint32_t fld_name_id = 0;
    wstring fld_name;
    output.id = doc.NextObjectId();

    output.size = (unsigned int)entry_list.size();
    output.items = new JsonObjectEntry[output.size];
    for (unsigned int idx = 0; idx < output.size; idx++)
    {
        output.items[idx] = entry_list[idx];
        output.items[idx].name.ToString(fld_name);
        doc.str_dict_.AddItem(fld_name, 0, fld_name_id, 0);
        doc.matrix_.AddCell(output.id, fld_name_id, idx);
    }

    output.hit.end = offset_ret;
    return offset_ret;
}

size_t JsonParser::ParseArray(JsonValue &output, JsonDoc &doc,
    const wchar_t *input_str, size_t input_len, size_t offset) const
{
    output.type_ = JsonValueType::Array;
    output.size_ = 0;
    output.ptr_ = nullptr;
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != L'[');

    size_t new_offset = SkipSeparators(input_str, input_len, offset + 1);
    Macro_RetIf(new_offset + 1, new_offset < input_len && input_str[new_offset] == L']');

    vector<JsonValue> items;
    JsonValue jvalue;
    bool is_succ = true, be_need_more_value = true;
    size_t offset_ret = offset;
    while (is_succ && be_need_more_value && new_offset < input_len)
    {
        offset_ret = ParseValue(jvalue, doc, input_str, input_len, new_offset);
        if (offset_ret == new_offset) {
            is_succ = false;
            break;
        }

        items.push_back(jvalue);
        new_offset = SkipSeparators(input_str, input_len, offset_ret);
        if (new_offset >= input_len) {
            is_succ = false;
            break;
        }

        if (input_str[new_offset] == L']') {
            be_need_more_value = false;
            offset_ret = new_offset + 1;
        }
        else if (input_str[new_offset] == L',') {
            be_need_more_value = true;
            new_offset++;
        }
        else {
            is_succ = false;
        }
    }

    if (!is_succ || be_need_more_value || items.empty())
    {
        JsonDoc::Free(items);
        return offset;
    }

    output.size_ = (unsigned int)items.size();
    JsonValue *array_ptr = new JsonValue[output.size_];
    for (unsigned int idx = 0; idx < output.size_; idx++) {
        array_ptr[idx] = items[idx];
    }
    output.ptr_ = array_ptr;

    output.hit_.end = offset_ret;
    return offset_ret;
}

size_t JsonParser::ParseArray(JsonValue &output, JsonDoc &doc,
    const char *input_str, size_t input_len, size_t offset) const
{
    output.type_ = JsonValueType::Array;
    output.size_ = 0;
    output.ptr_ = nullptr;
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != '[');

    size_t new_offset = SkipSeparators(input_str, input_len, offset + 1);
    Macro_RetIf(new_offset + 1, new_offset < input_len && input_str[new_offset] == ']');

    vector<JsonValue> items;
    JsonValue jvalue;
    bool is_succ = true, be_need_more_value = true;
    size_t offset_ret = offset;
    while (is_succ && be_need_more_value && new_offset < input_len)
    {
        offset_ret = ParseValue(jvalue, doc, input_str, input_len, new_offset);
        if (offset_ret == new_offset) {
            is_succ = false;
            break;
        }

        items.push_back(jvalue);
        new_offset = SkipSeparators(input_str, input_len, offset_ret);
        if (new_offset >= input_len) {
            is_succ = false;
            break;
        }

        if (input_str[new_offset] == ']') {
            be_need_more_value = false;
            offset_ret = new_offset + 1;
        }
        else if (input_str[new_offset] == ',') {
            be_need_more_value = true;
            new_offset++;
        }
        else {
            is_succ = false;
        }
    }

    if (!is_succ || be_need_more_value || items.empty())
    {
        JsonDoc::Free(items);
        return offset;
    }

    output.size_ = (unsigned int)items.size();
    JsonValue *array_ptr = new JsonValue[output.size_];
    for (unsigned int idx = 0; idx < output.size_; idx++) {
        array_ptr[idx] = items[idx];
    }
    output.ptr_ = array_ptr;

    output.hit_.end = offset_ret;
    return offset_ret;
}

//static
size_t JsonParser::ParseString(JsonString &jstr, const wchar_t *input_str,
    size_t input_len, size_t offset)
{
    jstr.Clear();
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != L'"');

    wstring output_str;
    size_t new_offset = ParseString(output_str, input_str, input_len, offset);

    jstr.len = (unsigned int)output_str.size();
    if (jstr.len > 0)
    {
        wchar_t *data_ptr = new wchar_t[jstr.len + 1];
        memcpy(data_ptr, output_str.c_str(), jstr.len * sizeof(wchar_t));
        data_ptr[jstr.len] = L'\0';
        jstr.data = data_ptr;
        jstr.is_wide = true;
    }
    else
    {
        jstr.data = nullptr;
    }

    return new_offset;
}

//static
size_t JsonParser::ParseString(JsonString &jstr, const char *input_str,
    size_t input_len, size_t offset)
{
    jstr.Clear();
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != '"');

    string output_str;
    size_t new_offset = ParseString(output_str, input_str, input_len, offset);

    jstr.len = (unsigned int)output_str.size();
    if (jstr.len > 0)
    {
        char *data_ptr = new char[jstr.len + 1];
        memcpy(data_ptr, output_str.c_str(), jstr.len * sizeof(char));
        data_ptr[jstr.len] = '\0';
        jstr.utf8_data = data_ptr;
        jstr.is_wide = false;
    }
    else
    {
        jstr.data = nullptr;
    }

    return new_offset;
}

//static
size_t JsonParser::ParseString(wstring &str, const wchar_t *input_str,
    size_t input_len, size_t offset)
{
    str.clear();
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != L'"');

    bool has_end_sign = false;
    size_t new_offset = offset + 1, delta_offset = 0;
    for (; new_offset < input_len; new_offset++)
    {
        wchar_t ch = input_str[new_offset];
        if (ch == L'"') {
            new_offset++;
            has_end_sign = true;
            break;
        }

        if (ch != L'\\')
        {
            str += ch;
        }
        else
        {
            Macro_RetIf(offset, offset + 1 >= input_len);
            delta_offset = 1;

            switch (input_str[new_offset+1])
            {
            case L'"':
                str += L'"';
                break;
            case L'\\':
                str += L'\\';
                break;
            case L'/':
                str += L'/';
                break;
            case L'b':
                str += L'\b';
                break;
            case L'f': //formfeed
                str += L'\f';
                break;
            case L'n':
                str += L'\n';
                break;
            case L'r':
                str += L'\r';
                break;
            case L't':
                str += L'\t';
                break;
            case L'u':
                {
                    Macro_RetIf(offset, new_offset + 5 >= input_len);

                    int unicode_value = 0, digit_value = 0;
                    for (int idx = 0; idx < 4; idx++)
                    {
                        wchar_t uchar = input_str[new_offset + 2 + idx];
                        if (uchar >= L'0' && uchar <= L'9') {
                            digit_value = (uchar - L'0');
                        }
                        else if (uchar >= L'A' && uchar <= L'F') {
                            digit_value = 10 + (uchar - L'A');
                        }
                        else if (uchar >= L'a' && uchar <= L'f') {
                            digit_value = 10 + (uchar - L'a');
                        }
                        else {
                            return offset;
                        }

                        unicode_value = 16 * unicode_value + digit_value;
                    }

                    str += (wchar_t)unicode_value;
                    delta_offset = 5;
                }
                break;
            default: //invalid format
                return offset;
                break;
            }

            new_offset += delta_offset;
        }
    }

    Macro_RetIf(offset, !has_end_sign);
    return new_offset;
}

//static
size_t JsonParser::ParseString(string &str, const char *input_str,
    size_t input_len, size_t offset)
{
    str.clear();
    Macro_RetIf(offset, offset >= input_len || input_str[offset] != '"');

    wstring wstr;
    bool has_end_sign = false;
    size_t new_offset = offset + 1, delta_offset = 0;
    for (; new_offset < input_len; new_offset++)
    {
        char ch = input_str[new_offset];
        if (ch == '"') {
            new_offset++;
            has_end_sign = true;
            break;
        }

        if (ch != '\\')
        {
            str += ch;
        }
        else
        {
            Macro_RetIf(offset, offset + 1 >= input_len);
            delta_offset = 1;

            switch (input_str[new_offset + 1])
            {
            case '"':
                str += '"';
                break;
            case '\\':
                str += '\\';
                break;
            case '/':
                str += '/';
                break;
            case 'b':
                str += '\b';
                break;
            case 'f': //formfeed
                str += '\f';
                break;
            case 'n':
                str += '\n';
                break;
            case 'r':
                str += '\r';
                break;
            case 't':
                str += '\t';
                break;
            case 'u':
            {
                Macro_RetIf(offset, new_offset + 5 >= input_len);

                int unicode_value = 0, digit_value = 0;
                for (int idx = 0; idx < 4; idx++)
                {
                    wchar_t uchar = input_str[new_offset + 2 + idx];
                    if (uchar >= '0' && uchar <= '9') {
                        digit_value = (uchar - '0');
                    }
                    else if (uchar >= 'A' && uchar <= 'F') {
                        digit_value = 10 + (uchar - 'A');
                    }
                    else if (uchar >= 'a' && uchar <= 'f') {
                        digit_value = 10 + (uchar - 'a');
                    }
                    else {
                        return offset;
                    }

                    unicode_value = 16 * unicode_value + digit_value;
                }

                wstr = (wchar_t)unicode_value;
                str += StringUtil::ToUtf8(wstr);
                delta_offset = 5;
            }
            break;
            default: //invalid format
                return offset;
                break;
            }

            new_offset += delta_offset;
        }
    }

    Macro_RetIf(offset, !has_end_sign);
    return new_offset;
}

size_t JsonParser::ParseNumber(MathDecimal &num, const wchar_t *input_str,
    size_t input_len, size_t offset) const
{
    num.Set(0);

    size_t new_offset = offset;
    uint32_t dot_num = 0, non_space_char_num = 0, char_num_after_exp_sign = 0;
    uint16_t digits_after_dot = 0;
    bool is_neg_value = false, is_neg_exp = false;
    int32_t exp_value = 0;
    bool is_exp = false, has_digit = false;
    for (; new_offset < input_len; new_offset++)
    {
        wchar_t wch = input_str[new_offset];
        const auto &uci = unicode_table_.Get(wch);

        non_space_char_num++;
        char_num_after_exp_sign++;

        if (wch == L'-' || wch == L'+')
        {
            if (non_space_char_num > 1 && char_num_after_exp_sign > 1) {
                break;
            }

            if (wch == L'-')
            {
                if (is_exp) {
                    is_neg_exp = true;
                }
                else {
                    is_neg_value = true;
                }
            }
        }
        else if (wch == L'.' && !is_exp && dot_num <= 0)
        {
            dot_num++;
        }
        else if (uci.lower_case == L'e' && new_offset > offset)
        {
            Macro_RetIf(offset, is_exp);
            is_exp = true;
            char_num_after_exp_sign = 0;
        }
        else if (wch >= L'0' && wch <= L'9')
        {
            uint32_t digit_value = (uint32_t)(wch - L'0');
            has_digit = true;
            if (is_exp)
            {
                exp_value *= 10;
                exp_value += digit_value;
            }
            else
            {
                if (dot_num > 0)
                {
                    digits_after_dot++;
                    num += MathDecimal(digit_value, -1 * digits_after_dot);
                }
                else
                {
                    num *= 10;
                    num += digit_value;
                }
            }
        }
        else
        {
            break;
        }
    }

    if (is_neg_value) {
        num *= -1;
    }

    if (exp_value != 0)
    {
        if (is_neg_exp) {
            exp_value = -exp_value;
        }
        num *= MathDecimal(1, (int16_t)exp_value);
    }

    bool bRet = has_digit && non_space_char_num > 0;
    return bRet ? new_offset : offset;
}

size_t JsonParser::ParseNumber(MathDecimal &num, const char *input_str,
    size_t input_len, size_t offset) const
{
    num.Set(0);

    size_t new_offset = offset;
    uint32_t dot_num = 0, non_space_char_num = 0, char_num_after_exp_sign = 0;
    uint16_t digits_after_dot = 0;
    bool is_neg_value = false, is_neg_exp = false;
    int32_t exp_value = 0;
    bool is_exp = false, has_digit = false;
    for (; new_offset < input_len; new_offset++)
    {
        wchar_t wch = input_str[new_offset];
        const auto &uci = unicode_table_.Get(wch);

        non_space_char_num++;
        char_num_after_exp_sign++;

        if (wch == '-' || wch == '+')
        {
            if (non_space_char_num > 1 && char_num_after_exp_sign > 1) {
                break;
            }

            if (wch == '-')
            {
                if (is_exp) {
                    is_neg_exp = true;
                }
                else {
                    is_neg_value = true;
                }
            }
        }
        else if (wch == '.' && !is_exp && dot_num <= 0)
        {
            dot_num++;
        }
        else if (uci.lower_case == 'e' && new_offset > offset)
        {
            Macro_RetIf(offset, is_exp);
            is_exp = true;
            char_num_after_exp_sign = 0;
        }
        else if (wch >= '0' && wch <= '9')
        {
            uint32_t digit_value = (uint32_t)(wch - '0');
            has_digit = true;
            if (is_exp)
            {
                exp_value *= 10;
                exp_value += digit_value;
            }
            else
            {
                if (dot_num > 0)
                {
                    digits_after_dot++;
                    num += MathDecimal(digit_value, -1 * digits_after_dot);
                }
                else
                {
                    num *= 10;
                    num += digit_value;
                }
            }
        }
        else
        {
            break;
        }
    }

    if (is_neg_value) {
        num *= -1;
    }

    if (exp_value != 0)
    {
        if (is_neg_exp) {
            exp_value = -exp_value;
        }
        num *= MathDecimal(1, (int16_t)exp_value);
    }

    bool bRet = has_digit && non_space_char_num > 0;
    return bRet ? new_offset : offset;
}

size_t JsonParser::ParseConstValue(JsonValue &output, const wchar_t *input_str,
    size_t input_len, size_t offset) const
{
    output.type_ = JsonValueType::Null;
    wstring str;
    size_t offset_ret = offset;
    for (; offset_ret < input_len && offset_ret < offset + 10; offset_ret++)
    {
        const auto &uci = unicode_table_.Get(input_str[offset_ret]);
        if (uci.IsAsciiLetterOrDigit()) {
            str += uci.lower_case;
        }
        else {
            break;
        }
    }

    if (str.compare(L"true") == 0) {
        output.SetBool(true);
    }
    else if (str.compare(L"false") == 0) {
        output.SetBool(false);
    }
    else if (str.compare(L"null") == 0) {
        output.SetNull();
    }
    else {
        offset_ret = offset;
    }

    return offset_ret;
}

size_t JsonParser::ParseConstValue(JsonValue &output, const char *input_str,
    size_t input_len, size_t offset) const
{
    output.type_ = JsonValueType::Null;
    string utf8_str;
    utf8_str.clear();
    size_t offset_ret = offset;
    for (; offset_ret < input_len && offset_ret < offset + 10; offset_ret++)
    {
        const auto &uci = unicode_table_.Get(input_str[offset_ret]);
        if (uci.IsAsciiLetterOrDigit()) {
            utf8_str += (char)uci.lower_case;
        }
        else {
            break;
        }
    }

    if (utf8_str.compare("true") == 0) {
        output.SetBool(true);
    }
    else if (utf8_str.compare("false") == 0) {
        output.SetBool(false);
    }
    else if (utf8_str.compare("null") == 0) {
        output.SetNull();
    }
    else {
        offset_ret = offset;
    }

    return offset_ret;
}

size_t JsonParser::SkipSeparators(const wchar_t *input_str,
    size_t input_len, size_t offset) const
{
    size_t offset_ret = offset;
    for (; offset_ret < input_len; offset_ret++)
    {
        const auto &uci = unicode_table_.Get(input_str[offset_ret]);
        Macro_RetIf(offset_ret, !uci.IsSeparator());
    }

    return offset_ret;
}

size_t JsonParser::SkipSeparators(const char *input_str,
    size_t input_len, size_t offset) const
{
    size_t offset_ret = offset;
    for (; offset_ret < input_len; offset_ret++)
    {
        const auto &uci = unicode_table_.Get(input_str[offset_ret]);
        Macro_RetIf(offset_ret, !uci.IsSeparator());
    }

    return offset_ret;
}

//static
const int JsonBuilder::ReservedSpaceCount;

JsonBuilder::JsonBuilder()
{
    line_end = L"\n";
    for (int idx = 0; idx < ReservedSpaceCount; idx++) {
        space_array_[idx] = L' ';
    }
}

bool JsonBuilder::Build(const JsonValue &jvalue, wostream &writer,
    int level, bool continue_line) const
{
    if (jvalue.type_ == JsonValueType::Array)
    {
        JsonArray jarr(jvalue.value_, jvalue.size_);
        return Build(jarr, writer, level, continue_line);
    }
    else if (jvalue.type_ == JsonValueType::Object)
    {
        JsonObject jobj(jvalue.entry_, jvalue.size_);
        return Build(jobj, writer, level, continue_line);
    }

    if (space_num_per_level > 0 && level > 0 && !continue_line)
    {
        int char_num = min(ReservedSpaceCount, space_num_per_level * level);
        writer.write(space_array_, char_num);
    }

    switch (jvalue.type_)
    {
    case JsonValueType::Null:
        writer << L"null";
        break;
    case JsonValueType::Boolean:
        writer << (jvalue.data_ == 0 ? L"false" : L"true");
        break;
    case JsonValueType::Number:
        {
            MathDecimal num(jvalue.data_, (int16_t)jvalue.size_);
            writer << StringUtil::Utf8ToWideStr(num.ToString());
        }
        break;
    case JsonValueType::String:
        if (jvalue.str_ != nullptr && jvalue.size_ > 0) {
            writer << jvalue.str_;
        }
        else {
            writer << L"";
        }
        break;
    default:
        break;
    }

    return writer.good();
}

bool JsonBuilder::Build(const JsonArray &jarray, std::wostream &writer,
    int level, bool continue_line) const
{
    if (space_num_per_level > 0 && level > 0 && !continue_line) {
        int char_num = min(ReservedSpaceCount, space_num_per_level * level);
        writer.write(space_array_, char_num);
    }

    if (jarray.size == 0)
    {
        writer << L"[]";
    }
    else
    {
        writer << L"[" << line_end;
        for (uint32_t idx = 0; idx < jarray.size; idx++)
        {
            Build(jarray.items[idx], writer, level + 1, false);
            if (idx + 1 < jarray.size) {
                writer << L',';
            }
            writer << line_end;
        }

        if (space_num_per_level > 0 && level > 0) {
            int char_num = min(ReservedSpaceCount, space_num_per_level * level);
            writer.write(space_array_, char_num);
        }
        writer << L"]";
    }

    return writer.good();
}

bool JsonBuilder::Build(const JsonObject &jobject, std::wostream &writer,
    int level, bool continue_line) const
{
    if (space_num_per_level > 0 && level > 0 && !continue_line) {
        int char_num = min(ReservedSpaceCount, space_num_per_level * level);
        writer.write(space_array_, char_num);
    }

    if (jobject.size == 0)
    {
        writer << L"{}";
    }
    else
    {
        writer << L"{" << line_end;
        for (uint32_t idx = 0; idx < jobject.size; idx++)
        {
            const JsonObjectEntry &entry = jobject.items[idx];
            if (space_num_per_level > 0) {
                int char_num = min(ReservedSpaceCount, space_num_per_level * (level+1));
                writer.write(space_array_, char_num);
            }
            if (entry.name.data != nullptr) {
                writer << entry.name.data << L':';
            }
            else {
                writer << L':';
            }
            Build(entry.value, writer, level+1, true);

            if (idx + 1 < jobject.size) {
                writer << L',';
            }
            writer << line_end;
        }

        if (space_num_per_level > 0 && level > 0) {
            int char_num = min(ReservedSpaceCount, space_num_per_level * level);
            writer.write(space_array_, char_num);
        }
        writer << L"}";
    }

    return writer.good();
}

//static
void JsonBuilder::EncodeString(std::wstring &dst, const std::wstring &src)
{
    EncodeString(dst, src.c_str(), (unsigned int)src.size());
}

//static
wstring JsonBuilder::EncodeString(const wstring &str)
{
    wstring res;
    EncodeString(res, str.c_str(), (unsigned int)str.size());
    return res;
}

//static
void JsonBuilder::EncodeString(std::wstring &dst, const JsonString &src)
{
    EncodeString(dst, src.data, src.len);
}

//static
void JsonBuilder::EncodeString(std::wstring &dst, const wchar_t *src, int64_t len)
{
    dst.clear();
    Macro_RetVoidIf(src == nullptr);

    for (int64_t char_idx = 0; (len < 0 && src[char_idx] != L'\0') || char_idx < len; char_idx++)
    {
        wchar_t ch = src[char_idx];
        switch (ch)
        {
        case L'"':
        case L'\\':
        case L'/':
            dst += L'\\';
            dst += ch;
            break;
        case L'\b':
            dst += L'\\';
            dst += L'b';
            break;
        case L'\f': //formfeed
            dst += L'\\';
            dst += L'f';
            break;
        case L'\n':
            dst += L'\\';
            dst += L'n';
            break;
        case L'\r':
            dst += L'\\';
            dst += L'r';
            break;
        case L'\t':
            dst += L'\\';
            dst += L't';
            break;
        default:
            if ((ch >= L'\0' && ch < L' ') || ch == 0x7F)
            {
                dst += L"\\u00";
                int digit_value = ch / 16;
                dst += (wchar_t)(digit_value < 10 ? (L'0' + digit_value) : (L'a' + (digit_value - 10)));
                digit_value = ch % 16;
                dst += (wchar_t)(digit_value < 10 ? (L'0' + digit_value) : (L'a' + (digit_value - 10)));
            }
            else
            {
                dst += ch;
            }
            break;
        }
    }
}

//static
void JsonBuilder::EncodeString(std::string &dst, const std::string &src)
{
    EncodeString(dst, src.c_str(), (unsigned int)src.size());
}

//static
string JsonBuilder::EncodeString(const string &str)
{
    string res;
    EncodeString(res, str.c_str(), (unsigned int)str.size());
    return res;
}

//static
void JsonBuilder::EncodeString(std::string &dst, const char *src, int64_t len)
{
    dst.clear();
    Macro_RetVoidIf(src == nullptr);

    for (int64_t char_idx = 0; (len < 0 && src[char_idx] != L'\0') || char_idx < len; char_idx++)
    {
        char ch = src[char_idx];
        switch (ch)
        {
        case '"':
        case '\\':
        case '/':
            dst += '\\';
            dst += ch;
            break;
        case '\b':
            dst += '\\';
            dst += 'b';
            break;
        case '\f': //formfeed
            dst += '\\';
            dst += 'f';
            break;
        case '\n':
            dst += '\\';
            dst += 'n';
            break;
        case '\r':
            dst += '\\';
            dst += 'r';
            break;
        case '\t':
            dst += '\\';
            dst += 't';
            break;
        default:
            if ((ch >= '\0' && ch < ' ') || ch == 0x7F)
            {
                dst += "\\u00";
                int digit_value = ch / 16;
                dst += (char)(digit_value < 10 ? ('0' + digit_value) : ('a' + (digit_value - 10)));
                digit_value = ch % 16;
                dst += (char)(digit_value < 10 ? ('0' + digit_value) : ('a' + (digit_value - 10)));
            }
            else
            {
                dst += ch;
            }
            break;
        }
    }
}

//static
void JsonBuilder::AppendField(std::wostream &writer, const wstring &key, const wstring &value)
{
    writer << "\"" << EncodeString(key) << "\":\"" << EncodeString(value) << "\"";
}

void JsonBuilder::AppendField(std::wostream & writer, const wstring & key, const wchar_t *value)
{
    writer << "\"" << EncodeString(key) << "\":\"" << EncodeString(value) << "\"";
}

//static
void JsonBuilder::AppendField(std::wostream &writer, const wstring &key, uint32_t value)
{
    writer << "\"" << EncodeString(key) << "\":" << value;
}

void JsonBuilder::AppendField(std::wostream & writer, const wstring & key, int32_t value)
{
    writer << "\"" << EncodeString(key) << "\":" << value;
}

//static
void JsonBuilder::AppendField(std::wostream &writer, const wstring &key, float value)
{
    writer << "\"" << EncodeString(key) << "\":" << value;
}

//static
void JsonBuilder::AppendField(std::wostream &writer, const wstring &key, double value)
{
    writer << "\"" << EncodeString(key) << "\":" << value;
}

//static
void JsonBuilder::AppendField(std::wostream &writer, const wstring &key, bool value)
{
    writer << "\"" << EncodeString(key) << "\":" << (value ? L"true" : L"false");
}

void JsonBuilder::AppendFieldName(std::wostream & writer, const wstring &key)
{
    writer << "\"" << EncodeString(key) << "\"";
}

void JsonBuilder::AppendFieldValue(std::wostream & writer, const wstring &value)
{
    writer << "\"" << EncodeString(value) << "\"";
}

} //end of namespace
