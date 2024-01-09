#pragma once

#include <cstring>
#include <list>
#include <map>
#include <algorithm>
#include "prime_types.h"
#include "hashtable.h"

namespace sslib
{

// A Least Recently Used (LRU) cache
template <class KeyType, class ValueType>
class LruCache
{
public:
    LruCache(int capacity = 0);
    virtual ~LruCache();

    void Clear(int new_capacity = 0);
    int Capacity() const;
    int Size() const;
    void Add(const KeyType &key, ValueType *value);
    void Touch(const KeyType &key);

    const KeyType* LatestKey() const;
    const ValueType* Get(const KeyType &key) const;
    ValueType* Get(const KeyType &key);

private:
    struct CacheItem
    {
        int64_t score = 0;
        ValueType *value = nullptr;
    };

private:
    int capacity_ = 0;
    std::map<int64_t, KeyType> list_;
    std::map<KeyType, CacheItem> map_;
    int64_t time_ = 0;
};

template <class KeyType, class ValueType>
LruCache<KeyType, ValueType>::LruCache(int capacity)
{
    capacity_ = capacity > 0 ? capacity : 1;
    time_ = 0;
}

template <class KeyType, class ValueType>
LruCache<KeyType, ValueType>::~LruCache()
{
    Clear();
}

template <class KeyType, class ValueType>
void LruCache<KeyType, ValueType>::Clear(int new_capacity)
{
    for (auto iter = map_.begin(); iter != map_.end(); iter++)
    {
        auto &item = iter->second;
        if (item.value != nullptr) {
            delete item.value ;
        }
    }
    map_.clear();
    list_.clear();

    if (new_capacity > 0) {
        capacity_ = new_capacity;
    }
    time_ = 0;
}

template <class KeyType, class ValueType>
int LruCache<KeyType, ValueType>::Capacity() const
{
    return capacity_;
}

template <class KeyType, class ValueType>
int LruCache<KeyType, ValueType>::Size() const
{
    return (int)map_.size();
}

template <class KeyType, class ValueType>
void LruCache<KeyType, ValueType>::Add(const KeyType &key, ValueType *value)
{
    time_++;
    auto iter = map_.find(key);
    if (iter == map_.end())
    {
        if ((int)list_.size() >= capacity_)
        {
            auto iter_least = list_.begin();
            auto iter_map = map_.find(iter_least->second);
            if (iter_map->second.value != nullptr) {
                delete iter_map->second.value;
                iter_map->second.value = nullptr;
            }
            map_.erase(iter_map);
            list_.erase(iter_least);
        }

        CacheItem new_item;
        new_item.score = time_;
        new_item.value = value;
        list_[new_item.score] = key;
        map_[key] = new_item;
    }
    else
    {
        auto &item = iter->second;
        if (item.value != nullptr)
        {
            delete item.value;
            item.value = nullptr;
        }
        item.value = value;

        auto iter_find = list_.find(item.score);
        if (iter_find != list_.end())
        {
            list_.erase(iter_find);
            item.score = time_;
            list_[time_] = key;
        }
    }
}

template <class KeyType, class ValueType>
void LruCache<KeyType, ValueType>::Touch(const KeyType &key)
{
    auto iter = map_.find(key);
    if (iter != map_.end())
    {
        time_++;
        auto iterTime = list_.find(iter->second.score);
        if (iterTime != list_.end()) {
            list_.erase(iterTime);
        }
        list_[time_] = key;
        iter->second.score = time_;
    }
}

template <class KeyType, class ValueType>
const KeyType* LruCache<KeyType, ValueType>::LatestKey() const
{
    auto iter = list_.rbegin();
    return iter == list_.rend() ? nullptr : &iter->second;
}

template <class KeyType, class ValueType>
const ValueType* LruCache<KeyType, ValueType>::Get(const KeyType &key) const
{
    auto iter = map_.find(key);
    if (iter != map_.end()) {
        return iter->second.m_value;
    }

    return nullptr;
}

template <class KeyType, class ValueType>
ValueType* LruCache<KeyType, ValueType>::Get(const KeyType &key)
{
    auto iter = map_.find(key);
    if (iter != map_.end()) {
        return iter->second.value;
    }

    return nullptr;
}

} //end of namespace
