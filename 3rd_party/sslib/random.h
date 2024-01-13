#pragma once

#include "prime_types.h"
#include <string>
#include <vector>

namespace sslib
{

using namespace std;

class Random
{
protected:
	uint64_t seed_ = 0;
	bool have_next_next_gaussian_ = false;
	double next_next_gaussian_ = 0.0;

	static int64_t MULTIPLIER;
	static int64_t ADDEND;
	static int64_t MASK;

public:
	Random();
	Random(int seed);
	Random(uint64_t seed);
	virtual ~Random();

	//Sets the seed of this random number generator
	inline void SetSeed(int seed)
    {
        SetSeed((uint64_t)seed);
    }
	inline void SetSeed(uint64_t seed)
    {
        seed_ = (seed ^ MULTIPLIER) & MASK;
        have_next_next_gaussian_ = false;
    }

    //
    inline int NextInt() {
        return (int)Next(8 * sizeof(int));
    }

    inline int NextInt(int to) {
        return (int)NextInt32(to);
    }

    inline int NextInt(int from, int to) {
        return (int)NextInt32(from, to);
    }

    inline int32_t NextInt32() {
        return Next(32);
    }

    inline int32_t NextInt32(int32_t to);
    inline int32_t NextInt32(int32_t from, int32_t to);

    inline int64_t NextInt64() {
        return ((int64_t)Next(32) << 32) + Next(32);
    }

    inline float NextFloat() {
        return Next(24) / ((float)(1 << 24));
    }

    inline float NextFloat(float from, float to);

    inline double NextDouble() {
        return (double)(((int64_t)Next(26) << 27) + Next(27)) / 0x20000000000000;
    }

    inline double NextDouble(double from, double to);

    double NextGaussian();

    void RandomSampling(vector<IdWeight<float>> &output, const vector<IdWeight<float>> &input, uint32_t count);
    void RandomSampling(vector<IdWeight<double>> &output, const vector<IdWeight<double>> &input, uint32_t count);

//Static methods
public:

protected:
	inline int32_t Next(int bits);
}; //class Random

int32_t Random::Next(int bits)
{
	seed_ = (seed_ * MULTIPLIER + ADDEND) & MASK;
	return (int32_t)(seed_ >> (48 - bits));
}

int32_t Random::NextInt32(int32_t to)
{
	int32_t bits, val;
	do {
		bits = Next(31);
		val = bits % to;
	}
	while(bits - val + (to-1) < 0);
	return val;
}

int32_t Random::NextInt32(int32_t from, int32_t to)
{
	if(from >= to) return from;
	return from + NextInt(to-from);
}

float Random::NextFloat(float from, float to)
{
	if(from >= to) return from;
	return from + NextFloat() * (to - from);
}

double Random::NextDouble(double from, double to)
{
	if(from >= to) return from;
	return from + NextDouble() * (to - from);
}

} //end of namespace
