#pragma once

#include <string>
#include <vector>
#include <map>
#include "../namespace.inc"

INFER_FLOW_BEGIN

using std::string;
using std::vector;
using std::map;
using std::pair;

struct StdVocabulary
{
public:
    struct Token
    {
        int id = 0;
        string str;
        float score = 0;
        int type = 0;
    };

public:
    map<string, int> str_to_id;
    vector<Token> token_array;

    map<pair<string, string>, int> merge_map;

public:
    void Clear()
    {
        str_to_id.clear();
        token_array.clear();
        merge_map.clear();
        eos_id_map_.clear();
    }

    int Size() const {
        return (int)token_array.size();
    }

    int unk() const {
        return unk_id_;
    }

    int bos() const {
        return bos_id_;
    }

    int eos() const {
        return eos_id_;
    }

    void SetUnk(int id) {
        this->unk_id_ = id;
    }

    void SetUnk(const string &str) {
        unk_id_ = StrToId(str);
    };

    void SetBos(int id) {
        bos_id_ = id;
    }

    void SetBos(const string &str) {
        bos_id_ = StrToId(str);
    }

    void SetEos(int id) {
        eos_id_ = id;
    }

    void SetEos(const vector<int> &ids);

    void SetEos(const string &str) {
        eos_id_ = StrToId(str);
    }

    bool IsEos(int id) const;

    int StrToId(const string &str) const
    {
        auto iter = str_to_id.find(str);
        return iter != str_to_id.end() ? iter->second : -1;
    }

    string IdToStr(int id, bool enable_decoding = true) const;

protected:
    int unk_id_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    map<int, int> eos_id_map_;

protected:
};

INFER_FLOW_END
