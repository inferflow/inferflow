#include "edit_distance.h"
#include "log.h"
#include <algorithm>

using namespace std;

namespace sslib
{

EditDistance::EditDistance()
{
    cost_matrix_ = new float[DefaultCostMatrixSize];
}

EditDistance::~EditDistance()
{
    if(cost_matrix_ != NULL) {
        delete []cost_matrix_;
        cost_matrix_ = NULL;
    }
}

float EditDistance::LevenshteinDistance(const string &str1,
    const string &str2, const Options &opt)
{
    uint32_t m = (uint32_t)str1.size() + 1;
    uint32_t n = (uint32_t)str2.size() + 1;
    bool is_new_memory = m * n > DefaultCostMatrixSize;
    float *d = is_new_memory ? new float[m * n] : cost_matrix_;

    for(uint32_t i = 0; i < m; i++) {
        d[i * n] = i * opt.opr_cost[StrEditOpr::Deletion];
    }
    for(uint32_t j = 0; j < n; j++) {
        d[j] = j * opt.opr_cost[StrEditOpr::Insertion];
    }

    float del_cost = 0, ins_cost = 0, sub_cost = 0;
    for(uint32_t j = 1; j < n; j++)
    {
        for(uint32_t i = 1; i < m; i++)
        {
            if(str1[i-1] == str2[j-1])
            {
                d[i * n + j] = d[(i-1) * n + (j-1)];
            }
            else
            {
                del_cost = d[(i-1) * n + j] + opt.opr_cost[StrEditOpr::Deletion];
                ins_cost = d[i * n + (j-1)] + opt.opr_cost[StrEditOpr::Insertion];
                sub_cost = d[(i-1) * n + (j-1)] + opt.opr_cost[StrEditOpr::Substitution];
                d[i * n + j] = min(min(del_cost, ins_cost), sub_cost);
            }
        }
    }

    float dist = d[m * n - 1];
    if(is_new_memory) {
        delete[] d;
    }
    return dist;
}

float EditDistance::LevenshteinDistance(const wstring &str1,
    const wstring &str2, const Options &opt)
{
    uint32_t m = (uint32_t)str1.size() + 1;
    uint32_t n = (uint32_t)str2.size() + 1;
    bool is_new_memory = m * n > DefaultCostMatrixSize;
    float *d = is_new_memory ? new float[m * n] : cost_matrix_;

    for (uint32_t i = 0; i < m; i++) {
        d[i * n] = i * opt.opr_cost[StrEditOpr::Deletion];
    }
    for (uint32_t j = 0; j < n; j++) {
        d[j] = j * opt.opr_cost[StrEditOpr::Insertion];
    }

    float del_cost = 0, ins_cost = 0, sub_cost = 0;
    for (uint32_t j = 1; j < n; j++)
    {
        for (uint32_t i = 1; i < m; i++)
        {
            if (str1[i - 1] == str2[j - 1])
            {
                d[i * n + j] = d[(i - 1) * n + (j - 1)];
            }
            else
            {
                del_cost = d[(i - 1) * n + j] + opt.opr_cost[StrEditOpr::Deletion];
                ins_cost = d[i * n + (j - 1)] + opt.opr_cost[StrEditOpr::Insertion];
                sub_cost = d[(i - 1) * n + (j - 1)] + opt.opr_cost[StrEditOpr::Substitution];
                d[i * n + j] = min(min(del_cost, ins_cost), sub_cost);
            }
        }
    }

    float dist = d[m * n - 1];
    if (is_new_memory) {
        delete[] d;
    }
    return dist;
}

} //end of namespace
