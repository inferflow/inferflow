#pragma once

#include <map>
#include <queue>
#include "sslib/string.h"
#include "sslib/naive_trie.h"
#include "std_vocabulary.h"

INFER_FLOW_BEGIN

using std::map;
using sslib::NaiveTrie;
using sslib::StrLessNoCase;

//tokenization algorithms
enum class TokenizationAlg
{
    Auto = 0,
    Std,
    FMM,    //Forward Maximum Matching
    FMM2,   //Forward Maximum Matching with space ignored
    BPE,    //Byte Pair Encoding
    ULM     //Unigram Language Model
};

typedef map<string, TokenizationAlg, StrLessNoCase> TokenizationAlgMap;

class TextTokenizer
{
public:
    TextTokenizer();
    virtual ~TextTokenizer();
    void Clear();

    bool Init(const StdVocabulary &vocab, int byte_token_id_start = 3);
    bool Tokenize(vector<int> &output, const string &input_text, bool add_bos = false,
        TokenizationAlg alg = TokenizationAlg::Auto);

    const StdVocabulary* vocabulary() const {
        return vocab_;
    }

    TokenizationAlg AlgorithmNameToId(const string &alg_name) const;

    static bool InitAlgorithmMap(TokenizationAlgMap &algorithm_map);

protected:
    struct Symbol
    {
        int prev = 0;
        int next = 0;
        const char *text = nullptr;
        int len = 0;
    };

    struct Bigram
    {
        struct Comparator
        {
            bool operator()(Bigram &lhs, Bigram &rhs)
            {
                return (lhs.score < rhs.score) || (lhs.score == rhs.score && lhs.left > rhs.left);
            }
        };

        int left = 0;
        int right = 0;
        float score = 0;
        int size = 0;
    };

    typedef std::priority_queue<Bigram, std::vector<Bigram>, Bigram::Comparator> BigramQueue;

protected:
    const StdVocabulary *vocab_ = nullptr;
    int byte_token_id_start_ = 3;
    NaiveTrie token_trie_;

    TokenizationAlgMap algorithm_map_;

protected:
    //FMM
    bool ForwardMaximumMatching(vector<int> &output, const string &input_text,
        bool ignore_space);
    //BPE
    bool BytePairEncoding(vector<int> &output, const string &input_text);

    void TryAddBigram(BigramQueue &work_queue, const vector<Symbol> &symbols,
        int left, int right);

    static int Utf8Len(char src);
};

INFER_FLOW_END
