#pragma once

#include <set>
#include <vector>
#include "prime_types.h"

namespace sslib
{

class TopKQueue
{
public:
    struct LessWeight
    {
        bool operator () (const IdWeight<float> &lhs, const IdWeight<float> &rhs) const
        {
            if (lhs.weight == rhs.weight) {
                return lhs.id > rhs.id;
            }
            return lhs.weight < rhs.weight;
        }
    };

public:
    TopKQueue(uint32_t k = 10);

    void Clear(uint32_t k = 0);

    uint32_t GetK() const {
        return k_;
    }

    uint32_t Size() const {
        return (uint32_t)data_.size();
    }

    const std::set<IdWeight<float>, LessWeight>& GetData() const {
        return data_;
    }

    void UpdateTopK(const IdWeight<float> &item);
    void UpdateTopK(uint32_t id, float weight);
    void UpdateTopK(const TopKQueue &topK);

    void GetList(std::vector<IdWeight<float>> &topList, uint32_t k = 0) const;
    IdWeight<float> GetTop() const;
    IdWeight<float> GetBottom() const;

protected:
    int k_ = 10;
    std::set<IdWeight<float>, LessWeight> data_;
};

} //end of namespace
