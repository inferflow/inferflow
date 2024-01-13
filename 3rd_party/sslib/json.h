#pragma once

#include <string>
#include <vector>
#include "prime_types.h"
#include "math_decimal.h"
#include "string_dict.h"
#include "naive_matrix.h"

////////////////////////////////////////////////////////////////////////////////
// JSON (JavaScript Object Notation) is a lightweight data-interchange format.
// Refer to http://json.org/ for more details
////////////////////////////////////////////////////////////////////////////////

namespace sslib
{

using std::wstring;

struct JsonHit
{
    size_t start = 0, end = 0;

    JsonHit(size_t s = 0, size_t e = 0)
    {
        start = s;
        end = e;
    }

    void Set(size_t s = 0, size_t e = 0)
    {
        start = s;
        end = e;
    }

    size_t Length() const
    {
        return end >= start ? end - start : 0;
    }
};

struct JsonString
{
    unsigned int len = 0;
    const wchar_t *data = nullptr;
    const char *utf8_data = nullptr;
    bool is_wide = true;

    JsonString(const wchar_t *p_data = nullptr, int p_len = 0)
    {
        data = p_data;
        len = p_len;
        is_wide = true;
    }

    JsonString(const char *p_data, int p_len = 0)
    {
        utf8_data = p_data;
        len = p_len;
        is_wide = false;
    }

    void Clear()
    {
        len = 0;
        data = nullptr;
        utf8_data = nullptr;
        is_wide = true;
    }

    void ToString(wstring &str) const
    {
        str.clear();
        if (is_wide && data != nullptr) {
            str.assign(data, len);
        }
        else if (!is_wide && utf8_data != nullptr) {
            StringUtil::Utf8ToWideStr(str, utf8_data, len);
        }
    }

    void ToString(string &str) const
    {
        if (is_wide && data != nullptr) {
            StringUtil::ToUtf8(str, data, len);
        }
        else if (!is_wide && utf8_data != nullptr) {
            str.assign(utf8_data, len);
        }
    }
};

enum class JsonValueType
{
    Null = 0, Boolean, Number, String, Utf8String, Array, Object
};

struct JsonArray;
struct JsonObjectEntry;
class JsonObject;
class JsonDoc;

class JsonValue
{
public:
    JsonValue();

    JsonValueType GetType() const
    {
        return type_;
    }

    bool GetJObject(JsonObject &obj) const;
    JsonObject GetJObject() const;
    bool GetArray(JsonArray &arr) const;
    bool GetValueList(std::vector<JsonValue> &value_list) const;
    bool GetString(JsonString &str) const;
    bool GetString(std::wstring &str) const;
    bool GetString(std::string &str) const;
    bool GetNumber(MathDecimal &num) const;
    MathDecimal GetNumber() const;
    int GetIntValue() const;
    uint32_t GetUInt32Value() const;
    Int64 GetInt64Value() const;
    UInt64 GetUInt64Value() const;
    float GetFloatValue() const;
    bool GetBoolValue() const;

    void SetObject(const JsonObject &obj);
    void SetString(const JsonString &str);
    void SetNumber(const MathDecimal &num);
    void SetBool(bool bTrue);
    void SetNull();

    const JsonHit& GetHit() const {
        return hit_;
    }

protected:
    JsonValueType type_;
    uint32_t id_ = 0;
    JsonHit hit_;

    unsigned int size_;
    union
    {
        int64_t data_;
        const wchar_t *str_;
        const char *utf8_str_;
        JsonValue *value_;
        JsonObjectEntry *entry_;
        void *ptr_;
    };

protected:
    friend class JsonDoc;
    friend class JsonParser;
    friend class JsonBuilder;
};

struct JsonObjectEntry
{
    JsonString name;
    JsonValue value;
    JsonHit hit;
};

class JsonObject
{
public:
    uint32_t id = 0;
    unsigned int size = 0;
    JsonObjectEntry *items = nullptr;
    JsonHit hit;

    JsonObject(JsonObjectEntry *p_items = nullptr, unsigned int p_size = 0)
    {
        size = p_size;
        items = p_items;
    }

    const JsonObjectEntry* GetFieldValue(const wstring &fld_name, const JsonDoc &doc) const;
    //keep: keep value if the field name is not available
    bool GetFieldValue(wstring &fld_value, const wstring &fld_name, const JsonDoc &doc, bool keep = true) const;
    bool GetFieldValue(int &fld_value, const wstring &fld_name, const JsonDoc &doc, bool keep = true) const;
    bool GetFieldValue(uint32_t &fld_value, const wstring &fld_name, const JsonDoc &doc, bool keep = true) const;
    bool GetFieldValue(UInt64 &fld_value, const wstring &fld_name, const JsonDoc &doc, bool keep = true) const;
    bool GetFieldValue(float &fld_value, const wstring &fld_name, const JsonDoc &doc, bool keep = true) const;
    bool GetFieldValue(bool &fld_value, const wstring &fld_name, const JsonDoc &doc, bool keep = true) const;
    bool GetFieldValue(JsonArray &fld_value, const wstring &fld_name, const JsonDoc &doc) const;
    bool GetFieldValue(JsonObject &fld_value, const wstring &fld_name, const JsonDoc &doc) const;

    bool GetFieldValue(string &fld_value, const wstring &fld_name, const JsonDoc &doc, bool keep = true) const;
};

struct JsonArray
{
    unsigned int size;
    JsonValue *items;
    JsonHit hit;

    JsonArray(JsonValue *p_items = nullptr, unsigned int p_size = 0)
    {
        size = p_size;
        items = p_items;
    }
};

class JsonDoc : public JsonValue
{
public:
    JsonDoc();
    virtual ~JsonDoc();

    void Clear();

    static void Free(JsonString &jstr);
    static void Free(JsonValue &jval);
    static void Free(JsonObject &jobj);
    static void Free(JsonObjectEntry &jentry);
    static void Free(std::vector<JsonObjectEntry> &entry_list);
    static void Free(std::vector<JsonValue> &value_list);

    uint32_t NextObjectId()
    {
        uint32_t id = next_object_id_;
        next_object_id_++;
        return id;
    }

protected:
    JsonDoc(const JsonDoc &rhs) = delete;
    JsonDoc& operator = (JsonDoc &rhs) = delete;

protected:
    WStrDict str_dict_;
    //row: object-id; column: field-id; value: field-index
    NaiveMatrix<uint32_t> matrix_;
    uint32_t next_object_id_ = 0;

protected:
    friend class JsonParser;
    friend class JsonObject;
};

class JsonParser
{
public:
    JsonParser();
    virtual ~JsonParser();

    bool Init();

    bool Parse(JsonDoc &output, const std::wstring &input_str) const;
    bool Parse(JsonDoc &output, const wchar_t *input_str, size_t input_len) const;

    bool ParseUtf8(JsonDoc &output, const std::string &input_str) const;
    bool ParseUtf8(JsonDoc &output, const char *input_str, size_t input_len) const;

    static void DecodeString(wstring &str, const wstring &src);
    static void DecodeString(string &str, const string &src);

protected:
    //return: new offset
    size_t ParseValue(JsonValue &output, JsonDoc &doc, const wchar_t *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseObject(JsonObject &output, JsonDoc &doc, const wchar_t *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseArray(JsonValue &output, JsonDoc &doc, const wchar_t *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseNumber(MathDecimal &num, const wchar_t *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseConstValue(JsonValue &output, const wchar_t *input_str,
        size_t input_len, size_t offset) const;

    size_t SkipSeparators(const wchar_t *input_str, size_t input_len, size_t offset) const;

    static size_t ParseString(JsonString &jstr, const wchar_t *input_str,
        size_t input_len, size_t offset);
    static size_t ParseString(wstring &str, const wchar_t *input_str,
        size_t input_len, size_t offset);

    //return: new offset
    size_t ParseValue(JsonValue &output, JsonDoc &doc, const char *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseObject(JsonObject &output, JsonDoc &doc, const char *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseArray(JsonValue &output, JsonDoc &doc, const char *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseNumber(MathDecimal &num, const char *input_str,
        size_t input_len, size_t offset) const;
    size_t ParseConstValue(JsonValue &output, const char *input_str,
        size_t input_len, size_t offset) const;

    size_t SkipSeparators(const char *input_str, size_t input_len, size_t offset) const;

    static size_t ParseString(JsonString &jstr, const char *input_str,
        size_t input_len, size_t offset);
    static size_t ParseString(string &str, const char *input_str,
        size_t input_len, size_t offset);

protected:
    bool is_initialized_ = false;
    UnicodeTable unicode_table_;
};

class JsonBuilder
{
public:
    int multi_line_level_threshold = 1;
    int space_num_per_level = 0;
    std::wstring line_end;

public:
    JsonBuilder();

    bool Build(const JsonValue &jvalue, std::wostream &writer,
        int level = 0, bool continue_line = false) const;
    bool Build(const JsonArray &jarray, std::wostream &writer,
        int level = 0, bool continue_line = false) const;
    bool Build(const JsonObject &jobject, std::wostream &writer,
        int level = 0, bool continue_line = false) const;

    static void EncodeString(std::wstring &dst, const std::wstring &src);
    static wstring EncodeString(const wstring &str);
    static void EncodeString(std::wstring &dst, const JsonString &src);
    static void EncodeString(std::wstring &dst, const wchar_t *src, int64_t len = -1);

    static void EncodeString(std::string &dst, const std::string &src);
    static string EncodeString(const string &str);
    static void EncodeString(std::string &dst, const char *src, int64_t len = -1);

    static void AppendField(std::wostream &writer, const wstring &key, const wstring &value);
    static void AppendField(std::wostream &writer, const wstring &key, const wchar_t *value);
    static void AppendField(std::wostream &writer, const wstring &key, uint32_t value);
    static void AppendField(std::wostream &writer, const wstring &key, int32_t value);
    static void AppendField(std::wostream &writer, const wstring &key, float value);
    static void AppendField(std::wostream &writer, const wstring &key, double value);
    static void AppendField(std::wostream &writer, const wstring &key, bool value);

    static void AppendFieldName(std::wostream &writer, const wstring &key);
    static void AppendFieldValue(std::wostream &writer, const wstring &value);

protected:
    static const int ReservedSpaceCount = 1024;
    wchar_t space_array_[ReservedSpaceCount];
};

} //end of namespace
