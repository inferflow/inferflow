#pragma once

namespace inferflow
{

template <typename EType>
bool InitMatrix(vector<EType> &matrix, int depth, int rows, int columns,
    int alg = 0, float param1 = 0, float param2 = 0, int seed = 0)
{
    matrix.resize(depth * rows * columns, (EType)0.0f);

    if (alg >= 0 || alg <= 5)
    {
        for (int d_idx = 0; d_idx < depth; d_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < columns; col_idx++)
                {
                    float value = param1;
                    if (alg == 1) {
                        value = param1 + param2 * col_idx;
                    }
                    else if (alg == 2) {
                        value = param1 * row_idx + param2 * col_idx;
                    }
                    else if (alg == 3) {
                        value = param1 * (1 + row_idx % 10) + param2 * (1 + col_idx % 10);
                    }
                    else {
                        value = param1 * (col_idx % 10);
                    }

                    int offset = col_idx + row_idx * columns + d_idx * rows * columns;
                    matrix[offset] = (EType)value;
                } //column
            } //row
        } //depth
    }
    else //random
    {
        int const k = 16807;
        int const m = 16;
        for (int d_idx = 0; d_idx < depth; d_idx++)
        {
            for (int row_idx = 0; row_idx < rows; row_idx++)
            {
                for (int col_idx = 0; col_idx < columns; col_idx++)
                {
                    int offset = col_idx + row_idx * columns + d_idx * rows * columns;
                    float value = float(((offset + seed) * k % m) - m / 2);
                    matrix[offset] = value;
                } //column
            } //row
        } //depth
    }

    return true;
}

} //end of namespace

