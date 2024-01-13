#include "naive_matrix.h"
#include <fstream>

using namespace std;

namespace sslib
{

//for compiling purpose only
void For_Compiling_NaiveMatrix()
{
    FloatNaiveMatrix mtx;
    mtx.Size();
    mtx.Clear();
    mtx.AddCell(0, 0, 0);
    float value = mtx.GetCellValue(0, 0);
    cout << "Cel value: " << value << endl;
    FloatNaiveMatrixCell cell(0, 0);
    value = mtx.GetCellValue(cell);
    cell.weight = 5;
    mtx.AddWeight(cell);

    const FloatNaiveMatrix::CellTable &cell_table = mtx.GetCellTable();
    FloatNaiveMatrix::CellTable::ConstIterator iter = cell_table.Begin();
    for(; iter.IsEnd(); iter.Next())
    {
        const auto &cur_cell = (*iter);
        cout << cur_cell.weight << endl;
    }

    mtx.Load("");
    mtx.Save("");
    mtx.SaveFmtCompactBin("");
    mtx.LoadFmtCompactBin("");
    mtx.Print("");
}

} //end of namespace
