#pragma once

#include "sparse_vector.h"
#include "std_static_matrix.h"
#include "std_static_graph.h"

namespace sslib
{

class VectorUtil
{
public:
    static double CosineSimilarity(const RawSparseVector &v1, const RawSparseVector &v2);
    static double JaccardSimilarity(const RawSparseVector &v1, const RawSparseVector &v2);

    static double JaccardSimilarity(const RawSparseVector &v1,
        const RawSparseVector &v2, const StdStaticGraph<float> &sim_graph);

    static bool LoadVectors(StdStaticMatrix<float> &mtrx, WStrDict &dict,
        const string &file_path, bool be_normalize);

protected:
    struct ItemPair
    {
        uint32_t item1 = 0, item2 = 0;
        float similarity = 0;
        float score = 0;
        float max_score = 0;

        ItemPair(uint32_t item1 = 0, uint32_t item2 = 0)
        {
            item1 = item1;
            item2 = item2;
        }

        static bool GreaterScore(const ItemPair &lhs, const ItemPair &rhs)
        {
            return lhs.score > rhs.score;
        }
    };
};

} //end of namespace
