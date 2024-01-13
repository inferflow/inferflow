#pragma once

#include <string>
#include "prime_types.h"

namespace sslib
{

struct StrEditOpr
{
    enum
    {
        Insertion = 0,
        Deletion,
        Substitution,
        Count
    };
};

class EditDistance
{
public:
    struct Options
    {
        float opr_cost[StrEditOpr::Count];

        Options()
        {
            for(uint32_t opr_idx = 0; opr_idx < StrEditOpr::Count; opr_idx++) {
                opr_cost[opr_idx] = 1.0f;
            }
        };
    };

public:
    EditDistance();
    virtual ~EditDistance();

    float LevenshteinDistance(const std::string &str1, const std::string &str2, const Options &opt);
    float LevenshteinDistance(const std::wstring &str1, const std::wstring &str2, const Options &opt);

protected:
    const static uint32_t DefaultCostMatrixSize = 4096;
    float *cost_matrix_ = nullptr;
};

} //end of namespace
