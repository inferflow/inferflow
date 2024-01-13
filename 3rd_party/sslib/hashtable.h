#pragma once

#include <cstring>
#include <functional>
#include "prime_types.h"
#include "blocked_allocator.h"
#include "blocked_array.h"

namespace sslib
{

#define Large_Prime 14988613

template <class IterType>
inline uint64_t RangeHashValue(IterType begin_iter, IterType end_iter)
{
    uint64_t result = 0;
    while(begin_iter != end_iter) {
        result = Large_Prime * result + (uint64_t)(*begin_iter);
        begin_iter++;
    }
    return result;
}

inline uint64_t HashValue(uint64_t val)
{
    return (uint64_t)(Large_Prime * (val >> 32) + (val & 0xFFFFFFFF));
}

inline uint64_t HashValue(int64_t val)
{
    return (uint64_t)(Large_Prime * (val >> 32) + (val & 0xFFFFFFFF));
}

inline uint64_t HashValue(uint32_t val)
{
    return (uint64_t)val;
}

inline uint64_t HashValue(int32_t val)
{
    return (uint64_t)(UInt32)val;
}

inline uint64_t HashValue(uint16_t val)
{
    return (uint64_t)val;
}

inline uint64_t HashValue(int16_t val)
{
    return (uint64_t)(uint16_t)val;
}

inline uint64_t HashValue(const char *val)
{
    return RangeHashValue(val, val + ::strlen(val));
}

inline uint64_t HashValue(const std::string &val)
{
    return RangeHashValue(val.c_str(), val.c_str() + val.size());
}

inline uint64_t HashValue(const wchar_t *val)
{
    return RangeHashValue(val, val + ::wcslen(val));
}

inline uint64_t HashValue(const std::wstring &val)
{
    return RangeHashValue(val.c_str(), val.c_str() + val.size());
}

inline uint64_t HashValue(const PairUInt32 &val)
{
    return Large_Prime * val.first + val.second;
}

template <class EleType, class BinaryPred = std::less<EleType>>
class HashCompare
{
public:
    uint64_t operator()(const EleType &val) const {
        return HashValue(val);
    }

    int operator()(const EleType &lhs, const EleType &rhs) const {
        return pred_(lhs, rhs) ? -1 : (pred_(rhs, lhs) ? 1 : 0);
    }

protected:
    BinaryPred pred_;
};

class StrHashCompare
{
public:
    uint64_t operator()(const char *val) const {
        return HashValue(val);
    }

    uint64_t operator()(const std::string &val) const {
        return RangeHashValue(val.c_str(), val.c_str()+val.size());
    }

    uint64_t operator()(const wchar_t *val) const {
        return HashValue(val);
    }

    uint64_t operator()(const std::wstring &val) const {
        return RangeHashValue(val.c_str(), val.c_str() + val.size());
    }

    int operator()(const char *lhs, const char *rhs) const {
        return strcmp(lhs, rhs);
    }

    int operator()(const std::string &lhs, const std::string &rhs) const {
        return strcmp(lhs.c_str(), rhs.c_str());
    }

    int operator()(const wchar_t *lhs, const wchar_t *rhs) const {
        return wcscmp(lhs, rhs);
    }

    int operator()(const std::wstring &lhs, const std::wstring &rhs) const {
        return wcscmp(lhs.c_str(), rhs.c_str());
    }
};

////////////////////////////////////////////////////////////////////////////////
// class Hashtable
////////////////////////////////////////////////////////////////////////////////

#pragma pack(push, 1)

template <class EleType, class CompType = HashCompare<EleType>>
class Hashtable
{
protected:
    struct Node
    {
        EleType data;
        Node *next = nullptr;
    };

    struct Slot
    {
        uint32_t size = 0;
        Node *first_node = nullptr;
    };

    struct Node32
    {
        EleType data;
        Location32 next = Location32Null;
    };

    struct Slot32
    {
        uint32_t size = 0;
        Location32 first_node = Location32Null;
    };

public:
    const static uint64_t DefaultSlotCount = 8;
    const static uint64_t MinBlockSize = 32;
    const static uint64_t MaxBlockSize = 8192000;
    const static uint64_t DefaultBlockSize = 4096;

    struct Iterator
    {
    public:
        bool Next()
        {
            if (table_ptr_ == nullptr) {
                return false;
            }

            if (table_ptr_->use_location32_)
            {
                if (node32_ptr_->next != Location32Null)
                {
                    node32_ptr_ = table_ptr_->node32_heap_.Get(node32_ptr_->next);
                }
                else
                {
                    node32_ptr_ = nullptr;
                    while (++slot_idx_ < table_ptr_->slot_num_)
                    {
                        if (table_ptr_->slot32_array_[slot_idx_].first_node != Location32Null)
                        {
                            node32_ptr_ = table_ptr_->node32_heap_.Get(table_ptr_->slot32_array_[slot_idx_].first_node);
                            break;
                        }
                    }
                }
            }
            else
            {
                if (node_ptr_->next != nullptr)
                {
                    node_ptr_ = node_ptr_->next;
                }
                else
                {
                    node_ptr_ = nullptr;
                    while (++slot_idx_ < table_ptr_->slot_num_)
                    {
                        if (table_ptr_->slot_array_[slot_idx_].first_node != nullptr) {
                            node_ptr_ = table_ptr_->slot_array_[slot_idx_].first_node;
                            break;
                        }
                    }
                }
            }
            return true;
        }

        bool IsEnd() const {
            return table_ptr_ == nullptr || (node_ptr_ == nullptr && node32_ptr_ == nullptr);
        }

        EleType& operator * () {
            return table_ptr_->use_location32_ ? node32_ptr_->data : node_ptr_->data;
        }

        const EleType& operator * () const {
            return table_ptr_->use_location32_ ? node32_ptr_->data : node_ptr_->data;
        }

        Iterator(uint32_t slot_idx = 0, Node *nd = nullptr, Node32 *nd32 = nullptr)
        {
            table_ptr_ = nullptr;
            slot_idx_ = slot_idx;
            node_ptr_ = nd;
            node32_ptr_ = nd32;
        }

    protected:
        uint64_t slot_idx_ = 0;
        Node *node_ptr_ = nullptr;
        Node32 *node32_ptr_ = nullptr;
        Hashtable *table_ptr_ = nullptr;

        friend class Hashtable;
    };

    struct ConstIterator
    {
    public:
        bool Next()
        {
            if(table_ptr_ == nullptr) {
                return false;
            }

            if(table_ptr_->use_location32_)
            {
                if(node32_ptr_->next != Location32Null)
                {
                    node32_ptr_ = table_ptr_->node32_heap_.Get(node32_ptr_->next);
                }
                else
                {
                    node32_ptr_ = nullptr;
                    while(++slot_idx_ < table_ptr_->slot_num_)
                    {
                        if(table_ptr_->slot32_array_[slot_idx_].first_node != Location32Null)
                        {
                            node32_ptr_ = table_ptr_->node32_heap_.Get(
                                table_ptr_->slot32_array_[slot_idx_].first_node);
                            break;
                        }
                    }
                }
            }
            else
            {
                if(node_ptr_->next != nullptr)
                {
                    node_ptr_ = node_ptr_->next;
                }
                else
                {
                    node_ptr_ = nullptr;
                    while(++slot_idx_ < table_ptr_->slot_num_)
                    {
                        if(table_ptr_->slot_array_[slot_idx_].first_node != nullptr)
                        {
                            node_ptr_ = table_ptr_->slot_array_[slot_idx_].first_node;
                            break;
                        }
                    }
                }
            }
            return true;
        }

        bool IsEnd() const {
            return table_ptr_ == nullptr || (node_ptr_ == nullptr && node32_ptr_ == nullptr);
        }

        const EleType& operator * () const {
            return table_ptr_->use_location32_ ? node32_ptr_->data : node_ptr_->data;
        }

        ConstIterator(uint32_t slot_idx = 0, Node *nd = nullptr, Node32 *nd32 = nullptr)
        {
            table_ptr_ = nullptr;
            slot_idx_ = slot_idx;
            node_ptr_ = nd;
            node32_ptr_ = nd32;
        }

        ConstIterator(Iterator iter)
        {
            table_ptr_ = iter.table_ptr;
            slot_idx_ = iter.slot_idx;
            node_ptr_ = iter.node_ptr_;
            node32_ptr_ = iter.node32_ptr;
        }

        const ConstIterator& operator = (Iterator iter)
        {
            table_ptr_ = iter.table_ptr;
            slot_idx_ = iter.slot_idx;
            node_ptr_ = iter.node_ptr_;
            node32_ptr_ = iter.node32_ptr;
            return *this;
        }

    protected:
        uint64_t slot_idx_ = 0;
        const Node *node_ptr_ = nullptr;
        const Node32 *node32_ptr_ = nullptr;
        const Hashtable *table_ptr_ = nullptr;

        friend class Hashtable;
    };

public:
    explicit Hashtable(uint64_t slot_num = 0, float max_load_factor = 1.5f, bool use_location32 = false)
        : slot_array_(use_location32 ? 32 : DefaultBlockSize), slot32_array_(use_location32 ? DefaultBlockSize : 32)
        , node_heap_(use_location32 ? 32 : DefaultBlockSize), node32_heap_(use_location32 ? DefaultBlockSize : 32)
    {
        item_num_ = 0;
        Reinit(slot_num, max_load_factor, use_location32);
    }
    Hashtable(CompType cmp, uint64_t slot_num = 0, float max_load_factor = 1.5f, bool use_location32 = false)
        : comp_(cmp)
        , slot_array_(use_location32 ? 32 : DefaultBlockSize), slot32_array_(use_location32 ? DefaultBlockSize : 32)
        , node_heap_(use_location32 ? 32 : DefaultBlockSize), node32_heap_(use_location32 ? DefaultBlockSize : 32)
    {
        //comp_ = cmp;
        item_num_ = 0;
        Reinit(slot_num, max_load_factor, use_location32);
    }
    virtual ~Hashtable()
    {
        Clear();
    }

    /////////////////////////////////////////////////////////////////////
    /// init and clear

    bool Reinit(uint64_t slot_num = 0, float max_load_factor = 1.5f, bool use_location32 = false)
    {
        Clear();
        use_location32_ = use_location32;
        if(use_location32_) {
            node32_heap_.SetEnableLocationNew(use_location32_);
        }

        slot_num_ = slot_num > 0 ? slot_num : DefaultSlotCount;
        max_load_factor_ = max_load_factor < 0.25f ? 0.25f : max_load_factor;
        if(slot_num_ > 0)
        {
            UInt32 block_size = slot_num_ <= MinBlockSize ? (UInt32)MinBlockSize
                : (slot_num_ <= MaxBlockSize ? (UInt32)slot_num_ : (UInt32)MaxBlockSize);
            if(use_location32_)
            {
                node32_heap_.SetBlockSize(block_size);
                slot32_array_.SetBlockSize(block_size);
            }
            else
            {
                node_heap_.SetBlockSize(block_size);
                slot_array_.SetBlockSize(block_size);
            }
        }

        if(use_location32_) {
            slot32_array_.resize(slot_num_);
        }
        else {
            slot_array_.resize(slot_num_);
        }
        return true;
    }

    void Clear()
    {
        slot_array_.clear();
        slot32_array_.clear();
        slot_num_ = 0;
        item_num_ = 0;
        node_heap_.Reset();
        node32_heap_.Reset();
    }

    /////////////////////////////////////////////////////////////////////
    /// 

    uint64_t Size() const
    {
        return item_num_;
    }

    ConstIterator Find(const EleType &item) const
    {
        ConstIterator iter(0, nullptr, nullptr);
        iter.table_ptr_ = this;
        if(slot_num_ == 0) {
            return iter;
        }

        uint64_t slot_idx = GetSlotIdx(item);
        iter.slot_idx_ = slot_idx;

        if(use_location32_)
        {
            const Node32 *cur_node = node32_heap_.Get(slot32_array_[slot_idx].first_node);
            while(cur_node != nullptr)
            {
                if(comp_(cur_node->data, item) == 0) {
                    iter.node32_ptr_ = cur_node;
                    break;
                }
                cur_node = node32_heap_.Get(cur_node->next);
            }
        }
        else
        {
            Node *cur_node = slot_array_[slot_idx].first_node;
            while(cur_node != nullptr)
            {
                if(comp_(cur_node->data, item) == 0) {
                    iter.node_ptr_ = cur_node;
                    break;
                }
                cur_node = cur_node->next;
            }
        }

        return iter;
    }

    Iterator Find(const EleType &item)
    {
        Iterator iter(0, nullptr, nullptr);
        iter.table_ptr_ = this;
        if(slot_num_ == 0) {
            return iter;
        }

        uint64_t slot_idx = GetSlotIdx(item);
        iter.slot_idx_ = slot_idx;

        if(use_location32_)
        {
            Node32 *cur_node = node32_heap_.Get(slot32_array_[slot_idx].first_node);
            while(cur_node != nullptr)
            {
                if(comp_(cur_node->data, item) == 0) {
                    iter.node32_ptr_ = cur_node;
                    break;
                }
                cur_node = node32_heap_.Get(cur_node->next);
            }
        }
        else
        {
            Node *cur_node = slot_array_[slot_idx].first_node;
            while(cur_node != nullptr)
            {
                if(comp_(cur_node->data, item) == 0) {
                    iter.node_ptr_ = cur_node;
                    break;
                }
                cur_node = cur_node->next;
            }
        }

        return iter;
    }

    ConstIterator Begin() const
    {
        ConstIterator iter(0, nullptr, nullptr);
        iter.table_ptr_ = this;
        
        if(use_location32_)
        {
            for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
            {
                if(slot32_array_[slot_idx].first_node != Location32Null)
                {
                    iter.slot_idx_ = slot_idx;
                    iter.node32_ptr_ = node32_heap_.Get(slot32_array_[slot_idx].first_node);
                    break;
                }
            }
        }
        else
        {
            for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
            {
                if(slot_array_[slot_idx].first_node != nullptr)
                {
                    iter.slot_idx_ = slot_idx;
                    iter.node_ptr_ = slot_array_[slot_idx].first_node;
                    break;
                }
            }
        }

        return iter;
    }

    Iterator Begin()
    {
        Iterator iter(0, nullptr);
        iter.table_ptr = this;

        if(use_location32_)
        {
            for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
            {
                if(slot32_array_[slot_idx].first_node != Location32Null)
                {
                    iter.slot_idx = slot_idx;
                    iter.node32_ptr = node32_heap_.Get(slot32_array_[slot_idx].first_node);
                    break;
                }
            }
        }
        else
        {
            for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
            {
                if(slot_array_[slot_idx].first_node != nullptr)
                {
                    iter.slot_idx = slot_idx;
                    iter.node_ptr_ = slot_array_[slot_idx].first_node;
                    break;
                }
            }
        }

        return iter;
    }

    /////////////////////////////////////////////////////////////////////
    ///

    Iterator Insert(const EleType &item)
    {
        if((float)item_num_ >= max_load_factor_ * slot_num_) {
            AddSlots();
        }

        Iterator iter(0, nullptr);
        iter.table_ptr_ = this;
        if(slot_num_ == 0)
        {
            bool is_succ = Reinit();
            if(!is_succ) {
                return iter;
            }
        }

        uint64_t slot_idx = GetSlotIdx(item);
        iter.slot_idx_ = slot_idx;

        if(use_location32_)
        {
            Slot32 &cur_slot = slot32_array_[slot_idx];
            Node32 *first_node = node32_heap_.Get(cur_slot.first_node);
            if(first_node == nullptr)
            {
                first_node = node32_heap_.New(1, cur_slot.first_node);
                first_node->next = Location32Null;
                first_node->data = item;

                cur_slot.size = 1;
                item_num_++;
                iter.node32_ptr_ = first_node;
            }
            else
            {
                int cmp_ret = comp_(item, first_node->data);
                bool is_dup_item_detected = cmp_ret == 0;
                Node32 *pre_node = nullptr;
                Node32 *cur_node = first_node;
                Location32 cur_node_loc = cur_slot.first_node;
                if(cmp_ret > 0)
                {
                    pre_node = first_node;
                    cur_node_loc = first_node->next;
                    cur_node = node32_heap_.Get(pre_node->next);
                    while(cur_node != nullptr)
                    {
                        cmp_ret = comp_(item, cur_node->data);
                        if(cmp_ret < 0) {
                            break;
                        }
                        else if(cmp_ret == 0) {
                            is_dup_item_detected = true;
                        }

                        pre_node = node32_heap_.Get(pre_node->next);
                        cur_node_loc = cur_node->next;
                        cur_node = node32_heap_.Get(cur_node->next);
                    }
                }

                if(!is_dup_item_detected)
                {
                    Location32 new_node_loc;
                    Node32 *new_node_ptr = node32_heap_.New(1, new_node_loc);
                    new_node_ptr->data = item;
                    new_node_ptr->next = cur_node_loc;

                    if(pre_node == nullptr) {
                        cur_slot.first_node = new_node_loc;
                    }
                    else {
                        pre_node->next = new_node_loc;
                    }

                    cur_slot.size++;
                    item_num_++;
                    iter.node32_ptr_ = new_node_ptr;
                }
            }
        }
        else //!use_location32_
        {
            Slot &cur_slot = slot_array_[slot_idx];
            if(cur_slot.first_node == nullptr)
            {
                cur_slot.first_node = node_heap_.New(1);
                cur_slot.first_node->next = nullptr;
                cur_slot.first_node->data = item;

                cur_slot.size = 1;
                item_num_++;
                iter.node_ptr_ = cur_slot.first_node;
            }
            else
            {
                int cmp_ret = comp_(item, cur_slot.first_node->data);
                bool is_dup_item_detected = cmp_ret == 0;
                Node *pre_node = nullptr;
                Node *cur_node = cur_slot.first_node;
                if(cmp_ret > 0)
                {
                    pre_node = cur_slot.first_node;
                    cur_node = pre_node->next;
                    while(cur_node != nullptr)
                    {
                        cmp_ret = comp_(item, cur_node->data);
                        if(cmp_ret < 0) {
                            break;
                        }
                        else if(cmp_ret == 0) {
                            is_dup_item_detected = true;
                        }

                        pre_node = pre_node->next;
                        cur_node = cur_node->next;
                    }
                }

                if(!is_dup_item_detected)
                {
                    Node *new_node_ptr = node_heap_.New(1);
                    new_node_ptr->data = item;
                    new_node_ptr->next = cur_node;

                    if(pre_node == nullptr) {
                        cur_slot.first_node = new_node_ptr;
                    }
                    else {
                        pre_node->next = new_node_ptr;
                    }

                    cur_slot.size++;
                    item_num_++;
                    iter.node_ptr_ = new_node_ptr;
                }
            }
        }

        return iter;
    }

    /////////////////////////////////////////////////////////////////////
    /// Advanced member functions

    //get the maximal number of items in an slot
    UInt32 GetMaxSlotSize();

    bool CheckValid() const;

protected:
    uint64_t item_num_;
    uint64_t slot_num_;
    BlockedArray<Slot> slot_array_;
    BlockedArray<Slot32> slot32_array_;
    BlockedAllocator<Node> node_heap_;
    BlockedAllocator<Node32> node32_heap_;

    CompType comp_;

    //options
    float max_load_factor_;
    bool use_location32_;

protected:
    uint64_t GetSlotIdx(const EleType &item) const
    {
        uint64_t hash_val = comp_(item);
        return hash_val % slot_num_;
    }

    void AddSlots();    
};

#pragma pack(pop)

////////////////////////////////////////////////////////////////////////////////////////////////
// class HashMap
////////////////////////////////////////////////////////////////////////////////////////////////

#pragma pack(push, 1)

template <class KeyType, class ValueType, class CompType = HashCompare<KeyType>>
class HashMap
{
protected:
    struct Element
    {
        KeyType key;
        ValueType value;

        Element() {};
        Element(const KeyType &k) { key = k; }
        Element(const KeyType &k, const ValueType &v) { key = k; value = v; }
    };

    class ElementHashCompare
    {
    public:
        uint64_t operator()(const Element &val) const {
            return cmp_(val.key);
        }

        int operator()(const Element &lhs, const Element &rhs) const {
            return cmp_(lhs.key, rhs.key);
        }

    protected:
        CompType cmp_;
    };

    typedef Hashtable<Element, ElementHashCompare> ElementTable;
    typedef typename ElementTable::Iterator ElementTableIterator;
    typedef typename ElementTable::ConstIterator ElementTableConstIterator;

public:
    class Iterator
    {
    public:
        bool Next() {
            return iter_.Next();
        }
        bool IsEnd() const {
            return iter_.IsEnd();
        }
        const KeyType Key() const {
            return (*iter_).key;
        }
        ValueType& Value() {
            return (*iter_).value;
        }

    protected:
        ElementTableIterator iter_;
        friend class HashMap;
    };

    class ConstIterator
    {
    public:
        bool Next() {
            return iter_.Next();
        }
        bool IsEnd() const {
            return iter_.IsEnd();
        }
        const KeyType Key() const {
            return (*iter_).key;
        }
        const ValueType& Value() const {
            return (*iter_).value;
        }

        ConstIterator() {};
        ConstIterator(const Iterator rhs) {
            iter_ = rhs.iter_;
        }
        const ConstIterator& operator = (const Iterator rhs) {
            if(this != &rhs) {
                iter_ = rhs.iter_;
            }
        }

    protected:
        ElementTableConstIterator iter_;
        friend class HashMap;
    };

public:
    explicit HashMap(uint64_t slot_num = 0, float max_load_factor = 1.5f, bool use_location32 = false)
        : item_table_(slot_num, max_load_factor, use_location32)
    {
    }
    HashMap(CompType cmp, uint64_t slot_num = 0, float max_load_factor = 1.5f, bool use_location32 = false)
        : item_table_(cmp, slot_num, max_load_factor, use_location32)
    {
    }
    virtual ~HashMap()
    {
        Clear();
    }

    bool Reinit(uint64_t slot_num = 0, float max_load_factor = 1.5f, bool use_location32 = false) {
        return item_table_.Reinit(slot_num, max_load_factor, use_location32);
    }
    void Clear() {
        return item_table_.Clear();
    }

    uint64_t Size() const {
        return item_table_.Size();
    }

    ConstIterator Find(const KeyType &key) const
    {
        ConstIterator iter;
        Element item;
        item.key = key;
        iter.iter_ = item_table_.Find(item);
        return iter;
    }

    Iterator Find(const KeyType &key)
    {
        Iterator iter;
        Element item;
        item.key = key;
        iter.iter_ = item_table_.Find(item);
        return iter;
    }

    ConstIterator Begin() const
    {
        ConstIterator iter;
        iter.iter_ = item_table_.Begin();
        return iter;
    }

    Iterator Begin()
    {
        Iterator iter;
        iter.iter_ = item_table_.Begin();
        return iter;
    }

    Iterator Insert(const KeyType &key, const ValueType &value)
    {
        Element item(key, value);
        Iterator iter;
        iter.iter_ = item_table_.Insert(item);
        return iter;
    }

protected:
    ElementTable item_table_;
};

#pragma pack(pop)

typedef HashMap<UInt32, UInt32> HashMapUInt32;
typedef HashMap<std::string, std::string> HashMapString;

////////////////////////////////////////////////////////////////////////////////////////////////
// Hashtable Implementation
////////////////////////////////////////////////////////////////////////////////////////////////

template <class EleType, class CompType>
UInt32 Hashtable<EleType, CompType>::GetMaxSlotSize()
{
    uint32_t nMaxSlotSize = 0;
    if(use_location32_)
    {
        for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
        {
            if(nMaxSlotSize < slot32_array_[slot_idx].size) {
                nMaxSlotSize = slot32_array_[slot_idx].size;
            }
        }
    }
    else
    {
        for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
        {
            if(nMaxSlotSize < slot_array_[slot_idx].size) {
                nMaxSlotSize = slot_array_[slot_idx].size;
            }
        }
    }
    return nMaxSlotSize;
}

template <class EleType, class CompType>
bool Hashtable<EleType, CompType>::CheckValid() const
{
    uint64_t total_items = 0;
    if(use_location32_)
    {
        for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
        {
            const Slot32 &cur_slot = slot32_array_[slot_idx];
            uint32_t slot_size = 0;
            const Node32 *node_ptr = node32_heap_.Get(cur_slot.first_node);
            while(node_ptr != nullptr)
            {
                slot_size++;
                node_ptr = node32_heap_.Get(node_ptr->next);
                if(slot_size == cur_slot.size && node_ptr != nullptr) {
                    return false;
                }
            }

            if(slot_size != cur_slot.size) {
                return false;
            }

            total_items += cur_slot.size;
        }
    }
    else
    {
        for(uint64_t slot_idx = 0; slot_idx < slot_num_; slot_idx++)
        {
            const Slot &cur_slot = slot_array_[slot_idx];
            uint32_t slot_size = 0;
            const Node *node_ptr = cur_slot.first_node;
            while(node_ptr != nullptr)
            {
                slot_size++;
                node_ptr = node_ptr->next;
                if(slot_size == cur_slot.size && node_ptr != nullptr) {
                    return false;
                }
            }
            if(slot_size != cur_slot.size) {
                return false;
            }

            total_items += cur_slot.size;
        }
    }

    if(total_items != item_num_) {
        return false;
    }
    return true;
}

template <class EleType, class CompType>
void Hashtable<EleType, CompType>::AddSlots()
{
    uint64_t old_slot_num = slot_num_;
    slot_num_ *= 2;
    if(use_location32_)
    {
        slot32_array_.resize(slot_num_);
        for(uint64_t slot_idx = 0; slot_idx < old_slot_num; slot_idx++)
        {
            Slot32 &cur_slot = slot32_array_[slot_idx];
            Location32 loc_node = cur_slot.first_node, loc_node_tmp = Location32Null;
            Node32 *node_ptr = node32_heap_.Get(cur_slot.first_node);
            Node32 *node_pre = nullptr, *temp_node = nullptr;
            Location32 loc_nodes_to_move = Location32Null; //items are in descending order
            while(node_ptr != nullptr)
            {
                uint64_t new_slot_idx = GetSlotIdx(node_ptr->data);
                if(new_slot_idx == slot_idx)
                {
                    node_pre = node_ptr;
                    loc_node = node_ptr->next;
                    node_ptr = node32_heap_.Get(node_ptr->next);
                }
                else
                {
                    //remove from this slot
                    if(node_pre == nullptr) {
                        cur_slot.first_node = node_ptr->next;
                    }
                    else {
                        node_pre->next = node_ptr->next;
                    }
                    temp_node = node_ptr;
                    node_ptr = node32_heap_.Get(node_ptr->next);
                    cur_slot.size--;

                    loc_node_tmp = temp_node->next;
                    temp_node->next = loc_nodes_to_move;
                    loc_nodes_to_move = loc_node;
                    loc_node = loc_node_tmp;
                }
            }

            Node32 *nodes_to_move = node32_heap_.Get(loc_nodes_to_move);
            while(nodes_to_move != nullptr)
            {
                loc_node_tmp = loc_nodes_to_move;
                temp_node = nodes_to_move;
                loc_nodes_to_move = nodes_to_move->next;
                nodes_to_move = node32_heap_.Get(loc_nodes_to_move);

                //because the items pointed by nodes_to_move are descendingly ordered,
                //we will keep ascending order of the items by the following insertion way
                uint64_t new_slot_idx = GetSlotIdx(temp_node->data);
                Slot32 &new_slot = slot32_array_[new_slot_idx];
                temp_node->next = new_slot.first_node;
                new_slot.first_node = loc_node_tmp;
                new_slot.size++;
            }
        }
    }
    else //!use_location32_
    {
        slot_array_.resize(slot_num_);
        for(uint64_t slot_idx = 0; slot_idx < old_slot_num; slot_idx++)
        {
            Slot &cur_slot = slot_array_[slot_idx];
            Node *node_ptr = cur_slot.first_node;
            Node *node_pre = nullptr, *temp_node = nullptr;
            Node *nodes_to_move = nullptr; //items are in descending order
            while(node_ptr != nullptr)
            {
                uint64_t new_slot_idx = GetSlotIdx(node_ptr->data);
                if(new_slot_idx == slot_idx)
                {
                    node_pre = node_ptr;
                    node_ptr = node_ptr->next;
                }
                else
                {
                    //remove from this slot
                    if(node_pre == nullptr) {
                        cur_slot.first_node = node_ptr->next;
                    }
                    else {
                        node_pre->next = node_ptr->next;
                    }
                    temp_node = node_ptr;
                    node_ptr = node_ptr->next;
                    cur_slot.size--;

                    temp_node->next = nodes_to_move;
                    nodes_to_move = temp_node;
                }
            }

            while(nodes_to_move != nullptr)
            {
                temp_node = nodes_to_move;
                nodes_to_move = nodes_to_move->next;

                //because the items pointed by nodes_to_move are descendingly ordered,
                //we will keep ascending order of the items by the following insertion way
                uint64_t new_slot_idx = GetSlotIdx(temp_node->data);
                Slot &new_slot = slot_array_[new_slot_idx];
                temp_node->next = new_slot.first_node;
                new_slot.first_node = temp_node;
                new_slot.size++;
            }
        }
    }
}

} //end of namespace
