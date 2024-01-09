#include "vector_util.h"
#include "naive_matrix.h"
#include "log.h"
#include "macro.h"
#include "task_monitor.h"

namespace sslib
{

//static
double VectorUtil::CosineSimilarity(const RawSparseVector &v1, const RawSparseVector &v2)
{
    return v1.CosineSimilarity(v2);
}

//static
double VectorUtil::JaccardSimilarity(const RawSparseVector &v1, const RawSparseVector &v2)
{
    return v1.JaccardSimilarity(v2);
}

//please guarantee that the similarity scores in $sim_graph is normalized to [0, 1]
//static
double VectorUtil::JaccardSimilarity(const RawSparseVector &v1,
    const RawSparseVector &v2, const StdStaticGraph<float> &sim_graph)
{
    //all possible pairs
    vector<ItemPair> item_pair_list;
    float sim = 0, sim1 = 0, sim2 = 0;
    for (UInt32 idx1 = 0; idx1 < v1.size; idx1++)
    {
        const auto &t1 = v1.elements[idx1];
        for (UInt32 idx2 = 0; idx2 < v2.size; idx2++)
        {
            const auto &t2 = v2.elements[idx2];
            sim = 0;
            if (t1.id == t2.id)
            {
                sim = 1.0f;
            }
            else
            {
                sim1 = sim_graph.GetEdgeWeight(t1.id, t2.id);
                sim2 = sim_graph.GetEdgeWeight(t2.id, t1.id);
                sim = max(sim1, sim2);
            }

            if (sim > 0.01f)
            {
                ItemPair new_pair(t1.id, t2.id);
                new_pair.similarity = sim;
                new_pair.score = min(t1.weight, t2.weight) * sim;
                new_pair.max_score = max(t1.weight, t2.weight);
                item_pair_list.push_back(new_pair);
            }
        }
    }

    std::sort(item_pair_list.begin(), item_pair_list.end(), ItemPair::GreaterScore);

    float u = 0, d = 0;
    set<UInt32> set1, set2;
    for (const auto &the_pair : item_pair_list)
    {
        if (set1.find(the_pair.item1) != set1.end()
            || set2.find(the_pair.item2) != set2.end())
        {
            continue;
        }

        u += the_pair.score;
        d += the_pair.max_score;
        set1.insert(the_pair.item1);
        set2.insert(the_pair.item2);
    }

    for (UInt32 idx1 = 0; idx1 < v1.size; idx1++)
    {
        const auto &t1 = v1.elements[idx1];
        if (set1.find(t1.id) == set1.end()) {
            d += t1.weight;
        }
    }

    for (UInt32 idx2 = 0; idx2 < v2.size; idx2++)
    {
        const auto &t2 = v2.elements[idx2];
        if (set2.find(t2.id) == set2.end()) {
            d += t2.weight;
        }
    }

    if (Number::IsAlmostZero(d)) {
        return 0.0;
    }

    return u / d;
}

//static
bool VectorUtil::LoadVectors(StdStaticMatrix<float> &mtrx,
    WStrDict &dict, const string &file_path, bool be_normalize)
{
    BinaryFileStream strm;
    bool ret = strm.OpenForRead(file_path);
    Macro_RetxFalseIf(!ret, LogError("Failed to open the vector file: %s", file_path.c_str()));

    vector<IdWeight<float>> row_data;
    UInt32 item_id = 0;
    UInt32 dup_count = 0, max_cols = 0, min_cols = Number::MaxUInt32;
    wstring item_text, dup_item_text;
    string line_str;
    vector<string> tokens;

    //skip the first line
    strm.GetLine(line_str);

    TaskMonitor tm(10000);
    UInt32 line_count = 0;
    while (strm.GetLine(line_str))
    {
        String::TrimRight(line_str, "\t ");
        String::Split(line_str, tokens, "\t ");
        UInt32 cols = (UInt32)tokens.size();
        if (cols < 2) {
            continue;
        }

        if (max_cols < cols) {
            max_cols = cols;
        }
        if (min_cols > cols) {
            min_cols = cols;
        }

        StringUtil::Utf8ToWideStr(item_text, tokens[0]);
        dict.AddItem(item_text, 0, item_id, false);
        if (item_id != mtrx.GetRowCount())
        {
            dup_count++;
            if (dup_count == 1) {
                dup_item_text = item_text;
            }
            continue;
        }

        row_data.clear();
        for (int dim_idx = 0; dim_idx + 1 < (int)tokens.size(); dim_idx++)
        {
            float score = (float)atof(tokens[dim_idx + 1].c_str());
            row_data.push_back(IdWeight<float>(dim_idx, score));
        }

        if (be_normalize)
        {
            float norm_score = (float)RawSparseVector::Norm(row_data);
            for (int dim_idx = 0; dim_idx < (int)row_data.size(); dim_idx++)
            {
                row_data[dim_idx].weight = row_data[dim_idx].weight / norm_score;
            }
        }

        mtrx.AddRow(item_id, row_data);
        line_count++;
        tm.Progress(line_count);
    }
    tm.End();

    LogKeyInfo("Item count: %u", mtrx.GetRowCount());
    if (dup_count > 0) {
        LogWarning("Duplicate count: %u; Example: %s",
            dup_count, StringUtil::ToUtf8(dup_item_text).c_str());
    }
    LogKeyInfo("Max columns: %u, min columns: %u", max_cols, min_cols);

    return ret;
}

} //end of namespace
