#include "top_k_queue.h"
#include "number.h"

namespace sslib
{

using namespace std;

TopKQueue::TopKQueue(uint32_t k)
{
    k_ = k;
}

void TopKQueue::Clear(uint32_t k)
{
    data_.clear();
    if (k > 0) {
        k_ = k;
    }
}

void TopKQueue::UpdateTopK(const IdWeight<float> &item)
{
    if (k_ <= 0) {
        return;
    }

    int current_size = (int)data_.size();
    if (current_size <= 0 || current_size < k_)
    {
        data_.insert(item);
    }
    else
    {
        const auto lowest = data_.begin();
        if (item.weight > lowest->weight) {
            data_.erase(lowest);
            data_.insert(item);
        }
    }
}

void TopKQueue::UpdateTopK(uint32_t id, float weight)
{
    UpdateTopK(IdWeight<float>(id, weight));
}

void TopKQueue::UpdateTopK(const TopKQueue &topK)
{
    for (auto iter = topK.data_.rbegin(); iter != topK.data_.rend(); iter++) {
        UpdateTopK(*iter);
    }
}

void TopKQueue::GetList(std::vector<IdWeight<float>> &top_list, uint32_t k) const
{
    top_list.clear();
    top_list.reserve(data_.size());
    uint32_t m = 0;
    for (auto iter = data_.rbegin(); iter != data_.rend(); iter++)
    {
        top_list.push_back(*iter);
        m++;
        if (k > 0 && m >= k) {
            break;
        }
    }
}

IdWeight<float> TopKQueue::GetTop() const
{
    IdWeight<float> top_item(UINT32_MAX, 0);
    if (!data_.empty()) {
        top_item = *data_.rbegin();
    }

    return top_item;
}

IdWeight<float> TopKQueue::GetBottom() const
{
    IdWeight<float> item(UINT32_MAX, 0);
    if (!data_.empty()) {
        item = *data_.begin();
    }

    return item;
}

} //end of namespace
