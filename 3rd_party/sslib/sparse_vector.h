#pragma once

#include <vector>
#include "prime_types.h"
#include "number.h"
#include "serializable.h"

namespace sslib
{

using namespace std;

struct RawSparseVector
{
public:
    uint32_t size = 0;
    IdWeight<float> *elements = nullptr;

public:
    RawSparseVector(IdWeight<float> *p_elements = nullptr, uint32_t p_size = 0)
    {
        elements = p_elements;
        size = p_size;
    }

    void Set(IdWeight<float> *p_elements, uint32_t p_size)
    {
        elements = p_elements;
        size = p_size;
    }

    RawSparseVector& operator /= (float v);

    double Norm() const;
    double L1Norm() const;
    static double Norm(const vector<IdWeight<float>> &vec);

    double DotProduct(const RawSparseVector &v2) const;

    double CosineSimilarity(const RawSparseVector &v2, bool is_already_normalized = false) const;
    double JaccardSimilarity(const RawSparseVector &v2) const;
};

class SparseVectorRecord : public IBinSerializable
{
public:
    vector<IdWeight<float>> data;

public:
    SparseVectorRecord() {};
    virtual ~SparseVectorRecord();
    void Clear();

    virtual bool Read(IBinaryStream &reader, void *params = nullptr);
    virtual bool Write(IBinaryStream &writer, void *params = nullptr) const;
};

} //end of namespace
