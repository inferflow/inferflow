#include "std_static_matrix.h"
#include <fstream>

namespace sslib
{

//for compiling purpose only
void For_Compiling_StdStaticMatrix()
{
    StdStaticMatrix<float> mtx;
    mtx.GetRowCount();
    vector<StdStaticMatrix<float>::Column> row_data;
    mtx.AddRow(0, row_data);
    mtx.GetRow(0);

    mtx.Load("");
    mtx.Save("");
    mtx.SaveFmtCompactBin("");
    mtx.LoadFmtCompactBin("");
    mtx.Print("");
}

} //end of namespace
