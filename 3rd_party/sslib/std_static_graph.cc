#include "std_static_graph.h"

using namespace std;

namespace sslib
{

void StaticGraph_ForCompiling()
{
    StdStaticGraph<float> graph;
    graph.Clear();

    graph.GetVertexNum();

    StdStaticGraph<float>::NeighborArray nbr_array;
    graph.GetNeighbors(0, nbr_array);
    graph.GetNeighbor(0, 1);
    graph.SetEdgeWeight(0, 1, 0.5f);

    graph.Sort(StdStaticGraph<float>::SORT_BY_WEIGHT, false);

    graph.LoadFmtCompactBin("");
    graph.LoadFmtCompactBin("", 1000);
    graph.StoreFmtCompactBin("");
    graph.LoadFmtEdgeListBin("");
    graph.StoreFmtEdgeListBin("");
    graph.LoadFmtVertexListBin("");
    graph.StoreFmtVertexListBin("");
    graph.LoadFmtSimpleBin("");
    graph.StoreFmtSimpleBin("");

    graph.LoadFmtYaleSparseMatrix("");
    graph.StoreFmtYaleSparseMatrix("");
}

} //end of namespace
