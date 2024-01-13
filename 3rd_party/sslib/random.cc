#include "random.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>
#include "macro.h"

using namespace std;

namespace sslib
{

int64_t Random::MULTIPLIER = 0x5DEECE66DL;
int64_t Random::ADDEND = 0xBL;
int64_t Random::MASK = 0x0FFFFFFFFFFFF;	//(1L << 48) - 1

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Random::Random()
{
    SetSeed((uint64_t)time(nullptr));
}

Random::Random(int seed)
{
	SetSeed(seed);
	have_next_next_gaussian_ = false;
	next_next_gaussian_ = 0.0;
}

Random::Random(uint64_t seed)
{
	SetSeed(seed);
	have_next_next_gaussian_ = false;
	next_next_gaussian_ = 0.0;
}

Random::~Random()
{
}

double Random::NextGaussian()
{
    if (have_next_next_gaussian_)
    {
        have_next_next_gaussian_ = false;
        return next_next_gaussian_;
    }
    else
    {
        double v1, v2, s;

        do
        {
            v1 = 2 * NextDouble() - 1; // between -1.0 and 1.0
            v2 = 2 * NextDouble() - 1; // between -1.0 and 1.0
            s = v1 * v1 + v2 * v2;
        } while (s >= 1 || s == 0);

        double multiplier = sqrt(-2 * log(s) / s);
        next_next_gaussian_ = v2 * multiplier;
        have_next_next_gaussian_ = true;
        return v1 * multiplier;
    }
}

struct Range4Sampling
{
    uint32_t id = 0;
    double base = 0, weight = 0;
};

void Random::RandomSampling(vector<IdWeight<float> > &output,
    const vector<IdWeight<float>> &input, uint32_t count)
{
    output.clear();
    output.reserve(count);

    vector<Range4Sampling> range_array;
    range_array.reserve(input.size());
    double upper_bound = 0;
    for (size_t range_idx = 0; range_idx < input.size(); range_idx++)
    {
        Range4Sampling range;
        range.id = input[range_idx].id;
        range.weight = input[range_idx].weight;
        range.base = upper_bound;
        range_array.push_back(range);

        upper_bound = range.base + std::max((double)0, range.weight);
    }

    uint32_t sampled_count = 0;
    uint32_t range_num = (uint32_t)range_array.size();
    for (; sampled_count < count && range_num > 0; sampled_count++)
    {
        double random_value = NextDouble(0, upper_bound);

        //lookup
        uint32_t begin_idx = 0, end_idx = range_num - 1, mid = begin_idx;
        while (begin_idx < end_idx)
        {
            mid = (end_idx + begin_idx) / 2;
            if (random_value < range_array[mid].base)
            {
                end_idx = mid;
            }
            else if (random_value > range_array[mid + 1].base)
            {
                begin_idx = mid + 1;
                mid = begin_idx; //!!!
            }
            else
            {
                break;
            }
        }

        const Range4Sampling &cur_range = range_array[mid];
        output.push_back(IdWeight<float>(cur_range.id, (float)cur_range.weight));

        //
        const auto &last_range = range_array[range_num - 1];
        if (mid + 1 < range_num && Macro_FloatEqual((float)cur_range.weight, (float)last_range.weight))
        {
            auto &target_range = range_array[mid];
            target_range.id = last_range.id;
            upper_bound -= last_range.weight;
        }
        else
        {
            upper_bound = range_array[mid].base;
            for (uint32_t range_idx = mid; range_idx + 1 < range_num; range_idx++)
            {
                range_array[range_idx].id = range_array[range_idx + 1].id;
                range_array[range_idx].weight = range_array[range_idx + 1].weight;
                range_array[range_idx].base = upper_bound;
                upper_bound = range_array[range_idx].base + std::max((double)0, range_array[range_idx].weight);
            }
        }

        range_num--;
    }
}

void Random::RandomSampling(vector<IdWeight<double>> &output,
    const vector<IdWeight<double>> &input, uint32_t count)
{
    output.clear();
    output.reserve(count);

    vector<Range4Sampling> range_array;
    range_array.reserve(input.size());
    double upper_bound = 0;
    for (size_t range_idx = 0; range_idx < input.size(); range_idx++)
    {
        Range4Sampling range;
        range.id = input[range_idx].id;
        range.weight = input[range_idx].weight;
        range.base = upper_bound;
        range_array.push_back(range);

        upper_bound = range.base + std::max((double)0, range.weight);
    }

    uint32_t sampled_count = 0;
    uint32_t range_num = (uint32_t)range_array.size();
    for (; sampled_count < count && range_num > 0; sampled_count++)
    {
        double random_value = NextDouble(0, upper_bound);

        //lookup
        uint32_t begin_idx = 0, end_idx = range_num - 1, mid = begin_idx;
        while (begin_idx < end_idx)
        {
            mid = (end_idx + begin_idx) / 2;
            if (random_value < range_array[mid].base) {
                end_idx = mid;
            }
            else if (random_value > range_array[mid + 1].base) {
                begin_idx = mid + 1;
                mid = begin_idx; //!!!
            }
            else {
                break;
            }
        }

        const Range4Sampling &cur_range = range_array[mid];
        output.push_back(IdWeight<double>(cur_range.id, cur_range.weight));

        //
        const auto &last_range = range_array[range_num - 1];
        if (mid + 1 < range_num && Macro_DoubleEqual(cur_range.weight, last_range.weight))
        {
            auto &target_range = range_array[mid];
            target_range.id = last_range.id;
            upper_bound -= last_range.weight;
        }
        else
        {
            upper_bound = range_array[mid].base;
            for (uint32_t range_idx = mid; range_idx + 1 < range_num; range_idx++)
            {
                range_array[range_idx].id = range_array[range_idx + 1].id;
                range_array[range_idx].weight = range_array[range_idx + 1].weight;
                range_array[range_idx].base = upper_bound;
                upper_bound = range_array[range_idx].base + std::max((double)0, range_array[range_idx].weight);
            }
        }

        range_num--;
    }
}

} //end of namespace
