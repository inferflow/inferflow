#include "sparse_vector.h"
#include <cmath>
#include <algorithm>

namespace sslib
{

////////////////////////////////////////////////////////////////////////////////
// RawSparseVector

double RawSparseVector::Norm() const
{
    double d = 0;
    for (uint32_t idx = 0; idx < this->size; idx++)
    {
        d += (this->elements[idx].weight * this->elements[idx].weight);
    }

    return sqrt(d);
}

double RawSparseVector::L1Norm() const
{
    double d = 0;
    for (uint32_t idx = 0; idx < this->size; idx++)
    {
        d += std::fabs(this->elements[idx].weight);
    }

    return d;
}

//static
double RawSparseVector::Norm(const vector<IdWeight<float>> &vec)
{
    double d = 0;
    for (uint32_t idx = 0; idx < (uint32_t)vec.size(); idx++)
    {
        d += (vec[idx].weight * vec[idx].weight);
    }

    return sqrt(d);
}

RawSparseVector& RawSparseVector::operator /= (float v)
{
    for (uint32_t idx = 0; idx < this->size; idx++) {
        elements[idx].weight /= v;
    }

    return *this;
}

double RawSparseVector::DotProduct(const RawSparseVector &v2) const
{
    double dp = 0;
    uint32_t idx1 = 0, idx2 = 0;
    while (idx1 < this->size && idx2 < v2.size)
    {
        if (this->elements[idx1].id < v2.elements[idx2].id)
        {
            idx1++;
        }
        else if (this->elements[idx1].id > v2.elements[idx2].id)
        {
            idx2++;
        }
        else
        {
            dp += (this->elements[idx1].weight * v2.elements[idx2].weight);
            idx1++;
            idx2++;
        }
    }

    return dp;
}

double RawSparseVector::CosineSimilarity(const RawSparseVector &v2, bool bAlreadyNormalized) const
{
    double u = this->DotProduct(v2);
    if (bAlreadyNormalized) {
        return u;
    }

    double n1 = this->Norm();
    double n2 = v2.Norm();
    if (Number::IsAlmostZero(n1) || Number::IsAlmostZero(n2)) {
        return 0.0;
    }

    return u / n1 / n2;
}

double RawSparseVector::JaccardSimilarity(const RawSparseVector &v2) const
{
    double u = 0, d = 0;
    uint32_t idx1 = 0, idx2 = 0;
    while (idx1 < this->size && idx2 < v2.size)
    {
        if (this->elements[idx1].id < v2.elements[idx2].id)
        {
            d = d + this->elements[idx1].weight;
            idx1++;
        }
        else if (this->elements[idx1].id > v2.elements[idx2].id)
        {
            d = d + v2.elements[idx2].weight;
            idx2++;
        }
        else
        {
            u = u + min(this->elements[idx1].weight, v2.elements[idx2].weight);
            d = d + max(this->elements[idx1].weight, v2.elements[idx2].weight);
            idx1++;
            idx2++;
        }
    }

    for (; idx1 < this->size; idx1++) {
        d += this->elements[idx1].weight;
    }
    for (; idx2 < v2.size; idx2++) {
        d += v2.elements[idx2].weight;
    }

    if (Number::IsAlmostZero(d)) {
        return 0.0;
    }

    return u / d;
}

SparseVectorRecord::~SparseVectorRecord()
{
    Clear();
}

void SparseVectorRecord::Clear()
{
    data.clear();
}

bool SparseVectorRecord::Read(IBinaryStream &reader, void *params)
{
    bool ret = true;
    Clear();
    (void)params;

    uint32_t n = 0;
    ret = reader.Read(n);
    for (uint32_t idx = 0; ret && idx < n; idx++)
    {
        IdWeight<float> item;
        ret = ret && reader.Read(item.id);
        ret = ret && reader.Read(item.weight);
        data.push_back(item);
    }

    return ret && reader.IsGood();
}

bool SparseVectorRecord::Write(IBinaryStream &writer, void *params) const
{
    bool ret = true;
    (void)params;
    uint32_t n = (uint32_t)data.size();
    ret = writer.Write(n);
    for (const auto &item : data)
    {
        ret = ret && writer.Write(item.id);
        ret = ret && writer.Write(item.weight);
    }

    return ret && writer.IsGood();
}

} //end of namespace
