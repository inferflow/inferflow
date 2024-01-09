#include "text_tokenizer.h"

INFER_FLOW_BEGIN

TextTokenizer::TextTokenizer()
{
}

TextTokenizer::~TextTokenizer()
{
    Clear();
}

void TextTokenizer::Clear()
{
}

bool TextTokenizer::Init(const StdVocabulary &vocab, int byte_token_id_start)
{
    vocab_ = &vocab;
    byte_token_id_start_ = byte_token_id_start;

    //build the token trie
    token_trie_.Clear();
    for (const auto &token : vocab_->token_array)
    {
        token_trie_.AddEx(token.str, token.id);
    }

    InitAlgorithmMap(algorithm_map_);
    return true;
}

//static
int TextTokenizer::Utf8Len(char src)
{
    const int lookup_table[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t high_bits = uint8_t(src) >> 4;
    return lookup_table[high_bits];
}

//static
bool TextTokenizer::InitAlgorithmMap(TokenizationAlgMap &algorithm_map)
{
    algorithm_map.clear();
    algorithm_map["fmm"] = TokenizationAlg::FMM;
    algorithm_map["fmm2"] = TokenizationAlg::FMM2;
    algorithm_map["bpe"] = TokenizationAlg::BPE;
    algorithm_map["ulm"] = TokenizationAlg::ULM;
    return true;
}

TokenizationAlg TextTokenizer::AlgorithmNameToId(const string &alg_name) const
{
    auto iter = algorithm_map_.find(alg_name);
    return iter == algorithm_map_.end() ? TokenizationAlg::Auto: iter->second;
}

//Do NOT clear the output
bool TextTokenizer::ForwardMaximumMatching(std::vector<int> &output,
    const string &input_text, bool ignore_space)
{
    (void)ignore_space;
    bool ret = true;
    if (input_text.empty()) {
        return true;
    }

    //longest match
    int input_len = (int)input_text.size();
    int start_pos = 0;
    NaiveTrie::SearchResult res;
    while (start_pos < input_len)
    {
        int max_len = 0;
        int best_token_id = 0;
        for (int len = 1; start_pos + len <= input_len; len++)
        {
            res = token_trie_.PrefixSearch(input_text.c_str() + start_pos, len);
            if (!res.found) {
                break;
            }

            if (res.value != UINT32_MAX)
            {
                best_token_id = res.value;
                max_len = len;
            }
        }

        if (max_len > 0)
        {
            start_pos += max_len;
            output.push_back(best_token_id);
        }
        else
        {
            start_pos++;
        }
    }

    return ret;
}

bool TextTokenizer::BytePairEncoding(vector<int> &output, const string &input_text)
{
    vector<Symbol> symbols;
    BigramQueue work_queue;
    int input_text_len = (int)input_text.size();

    // Each symbol is a UTF-8 sequence corresponding to a Unicode character
    int index = 0, offset = 0;
    while (offset < input_text_len)
    {
        Symbol sym;
        int char_len = std::min(input_text_len - offset, Utf8Len(input_text[offset]));
        sym.text = input_text.c_str() + offset;
        sym.len = char_len;
        offset += char_len;
        sym.prev = index - 1;
        sym.next = offset == input_text_len ? -1 : index + 1;
        index++;
        symbols.emplace_back(sym);
    }

    // Try to add all possible 2-symbol tokens
    for (int idx = 1; idx < (int)symbols.size(); idx++) {
        TryAddBigram(work_queue, symbols, idx - 1, idx);
    }

    // Keep choosing the highest frequency pairs
    while (!work_queue.empty())
    {
        auto bigram = work_queue.top();
        work_queue.pop();

        auto &left_sym = symbols[bigram.left];
        auto &right_sym = symbols[bigram.right];

        // skip if one of the symbols already got merged
        if (left_sym.len == 0 || right_sym.len == 0 ||
            left_sym.len + right_sym.len != bigram.size)
        {
            continue;
        }

        // merge
        left_sym.len += right_sym.len;
        right_sym.len = 0;

        // remove the right sym
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) {
            symbols[right_sym.next].prev = bigram.left;
        }

        // find more bigrams
        TryAddBigram(work_queue, symbols, left_sym.prev, bigram.left);
        TryAddBigram(work_queue, symbols, bigram.left, left_sym.next);
    }

    for (int idx = 0; idx != -1; idx = symbols[idx].next)
    {
        auto &symbol = symbols[idx];
        auto token_iter = vocab_->str_to_id.find(string(symbol.text, symbol.len));

        if (token_iter == vocab_->str_to_id.end())
        { //the token is not in the vocabulary
            for (int byte_idx = 0; byte_idx < (int)symbol.len; ++byte_idx)
            {
                int token_id = uint8_t(symbol.text[byte_idx]) + byte_token_id_start_;
                output.push_back(token_id);
            }
        }
        else
        {
            output.push_back(token_iter->second);
        }
    }

    return true;
}

bool TextTokenizer::Tokenize(std::vector<int> &output, const string &input_text,
    bool add_bos, TokenizationAlg alg)
{
    bool ret = true;
    output.clear();
    if (input_text.empty()) {
        return true;
    }

    if (add_bos)
    {
        int bos_id = vocab_->bos();
        output.push_back(bos_id);
    }

    if (alg == TokenizationAlg::FMM || alg == TokenizationAlg::FMM2)
    {
        bool ignore_space = alg == TokenizationAlg::FMM2;
        ret = ForwardMaximumMatching(output, input_text, ignore_space);
        return ret;
    }

    ret = BytePairEncoding(output, input_text);

    return ret;
}

void TextTokenizer::TryAddBigram(BigramQueue &work_queue,
    const vector<Symbol> &symbols, int left, int right)
{
    if (left == -1 || right == -1) {
        return;
    }

    float token_score = 0;
    bool has_merge_data = !vocab_->merge_map.empty();
    if (has_merge_data)
    {
        string left_str(symbols[left].text, symbols[left].len);
        string right_str(symbols[right].text, symbols[right].len);
        auto item_iter = vocab_->merge_map.find(std::make_pair(left_str, right_str));
        if (item_iter == vocab_->merge_map.end()) {
            return;
        }

        token_score = 1.0f / (1 + item_iter->second);
    }
    else
    {
        string text(symbols[left].text, symbols[left].len + symbols[right].len);
        auto token_iter = vocab_->str_to_id.find(text);
        if (token_iter == vocab_->str_to_id.end()) {
            return;
        }

        int token_id = token_iter->second;
        if (token_id >= (int)vocab_->token_array.size()) {
            return;
        }

        const auto &token = vocab_->token_array[token_id];
        token_score = token.score;
    }

    Bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.score = token_score;
    bigram.size = symbols[left].len + symbols[right].len;
    work_queue.push(bigram);
}

INFER_FLOW_END
