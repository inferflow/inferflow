#pragma once

#include <vector>
#include "prime_types.h"
#include "log.h"

namespace sslib
{

template <class EleType>
class BlockedAllocator
{
public:
    static const int MinBlockSize = 1;          //Minimal block size: 1 element per block
    static const int DefaultBlockSize = 4096;   //Default block size: 4K elements per block

public:
    explicit BlockedAllocator(int block_size = DefaultBlockSize, bool enable_location_new = false)
    {
        item_count_ = 0;
        cur_block_ = nullptr;
        cur_block_size_ = 0;
        cur_block_idx_ = -1;
        cur_position_ = 0;

        enable_location_new_ = enable_location_new;
        loc_low_bit_num_ = 0;
        SetBlockSize(block_size);
    }

    virtual ~BlockedAllocator();

    void SetEnableLocationNew(bool enable_location_new);

    // Set block size (number of elements). Old blocks are not affected.
    // @param block_size: New block size
    void SetBlockSize(int block_size);

    int GetBlockSize() {
        return block_size_;
    }

    int GetItemCount() {
        return item_count_;
    }

    inline EleType* New(uint32_t num);
    inline EleType* New(uint32_t num, Location32 &loc);
    inline EleType* Get(Location32 loc);
    inline const EleType* Get(Location32 loc) const;

    /** Clear all strings and go back to the initial status */
    void Reset(int reserved_block_count = 0);

    void Clear(uint32_t reserved_block_count = 0) {
        Reset((int)reserved_block_count);
    }

protected:
    std::vector<EleType*> block_list_; //all blocks
    std::vector<uint32_t> block_size_list_; //size of each block

    int32_t cur_block_idx_;		//The index of new memory block
    EleType *cur_block_;		//New memory block. New strings will try to add into this block
    uint32_t cur_block_size_;	//
    uint32_t cur_position_;		//New element position. New elements will try to add to this position

    uint32_t item_count_;

    uint32_t block_size_, loc_low_bit_num_;
    bool enable_location_new_; //enable get locations (as the second parameter) in calling New()

public:
    BlockedAllocator(const BlockedAllocator<EleType>&)
    {
        LogSevere("Please do not call my copy constructor");
    }

    BlockedAllocator& operator = (const BlockedAllocator<EleType>&)
    {
        LogSevere("Please do not call my assignment function");
        return *this;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////////////////////

template <class EleType>
BlockedAllocator<EleType>::~BlockedAllocator()
{
    int block_count = (int)block_list_.size();
    for(int i = 0; i < block_count; i++)
    {
        if(block_list_[i] != nullptr) {
            delete[] block_list_[i];
        }
    }
    block_list_.clear();

    cur_block_ = nullptr;
    cur_block_size_ = 0;
    cur_block_idx_ = -1;
    cur_position_ = 0;

    item_count_ = 0;
}

template <class EleType>
void BlockedAllocator<EleType>::SetEnableLocationNew(bool enable_location_new)
{
    if(enable_location_new && !block_list_.empty()) {
        return;
    }
    enable_location_new_ = enable_location_new;
}

template <class EleType>
void BlockedAllocator<EleType>::SetBlockSize(int block_size)
{
    if(enable_location_new_ && !block_list_.empty()) {
        return;
    }

    block_size_ = block_size < MinBlockSize ? MinBlockSize : block_size;
    if(enable_location_new_)
    {
        loc_low_bit_num_ = 0;
        uint32_t data = block_size_ - 1;
        while(data != 0)
        {
            loc_low_bit_num_++;
            data = data >> 1;
        }
    }
}

template <class EleType>
inline EleType* BlockedAllocator<EleType>::New(uint32_t num)
{
    if(cur_block_ == nullptr || cur_position_ + num > cur_block_size_)
    {
        if(cur_block_idx_ >= 0)
        {
            if(cur_block_ != nullptr)
            {
                cur_block_idx_++;
                if(cur_block_idx_ >= (int)block_list_.size())
                {
                    block_list_.push_back(NULL);
                    block_size_list_.push_back(0);
                }

                cur_block_ = block_list_[cur_block_idx_];
                cur_block_size_ = block_size_list_[cur_block_idx_];
            }

            if(cur_block_ != nullptr && cur_block_size_ < num)
            {
                delete[] cur_block_;
                cur_block_ = nullptr;
            }
        }
        else
        {
            cur_block_idx_ = 0;
            block_list_.push_back(nullptr);
            block_size_list_.push_back(0);
            cur_block_ = nullptr;
        }

        if(cur_block_ == nullptr)
        {
            cur_block_size_ = block_size_ > num ? block_size_ : num;
            cur_block_ = new EleType[cur_block_size_];
        }

        cur_position_ = 0;
        block_list_[cur_block_idx_] = cur_block_;
        block_size_list_[cur_block_idx_] = cur_block_ != nullptr ? cur_block_size_ : 0;

        if(cur_block_ == nullptr) {
            return nullptr;
        }
    }

    EleType *ret = cur_block_ + cur_position_;
    cur_position_ += num;

    item_count_ += num;

	return ret;
}

template <class EleType>
inline EleType* BlockedAllocator<EleType>::New(uint32_t num, Location32 &loc)
{
    if(!enable_location_new_)
    {
        loc = UINT32_MAX;
        return nullptr;
    }

    if(cur_block_ == nullptr || cur_position_ + num > cur_block_size_)
    {
        if(cur_block_idx_ >= 0)
        {
            if(cur_block_ != nullptr)
            {
                cur_block_idx_++;
                if(cur_block_idx_ >= (int)block_list_.size())
                {
                    block_list_.push_back(NULL);
                    block_size_list_.push_back(0);
                }

                cur_block_ = block_list_[cur_block_idx_];
                cur_block_size_ = block_size_list_[cur_block_idx_];
            }

            if(cur_block_ != nullptr && cur_block_size_ < num)
            {
                delete[] cur_block_;
                cur_block_ = nullptr;
            }
		}
		else
        {
            cur_block_idx_ = 0;
            block_list_.push_back(NULL);
            block_size_list_.push_back(0);
            cur_block_ = nullptr;
        }

		if(cur_block_ == nullptr)
        {
			cur_block_size_ = block_size_ > num ? block_size_ : num;
			cur_block_ = new EleType[cur_block_size_];
		}

		cur_position_ = 0;
		block_list_[cur_block_idx_] = cur_block_;
		block_size_list_[cur_block_idx_] = cur_block_ != nullptr ? cur_block_size_ : 0;

		if(cur_block_ == nullptr) {
            loc = UINT32_MAX;
			return nullptr;
		}
	}

    loc = ((cur_block_idx_ << loc_low_bit_num_) | cur_position_);
	EleType *ret = cur_block_ + cur_position_;
	cur_position_ += num;

	item_count_ += num;
	return ret;
}

template <class EleType>
inline EleType* BlockedAllocator<EleType>::Get(Location32 loc)
{
    uint32_t block_idx = (loc >> loc_low_bit_num_);
    EleType *block_ptr = block_idx < block_list_.size() ? block_list_[block_idx] : nullptr;
    if(block_ptr != nullptr)
    {
        uint32_t iInBlockOffset = (loc & ((1 << loc_low_bit_num_) - 1));
        return iInBlockOffset < block_size_ ? block_ptr + iInBlockOffset : nullptr;
    }
    return nullptr;
}

template <class EleType>
inline const EleType* BlockedAllocator<EleType>::Get(Location32 loc) const
{
    uint32_t block_idx = (loc >> loc_low_bit_num_);
    const EleType *block_ptr = block_idx < block_list_.size() ? block_list_[block_idx] : nullptr;
    if(block_ptr != nullptr)
    {
        uint32_t iInBlockOffset = (loc & ((1 << loc_low_bit_num_) - 1));
        return iInBlockOffset < block_size_ ? block_ptr + iInBlockOffset : nullptr;
    }
    return nullptr;
}

template <class EleType>
void BlockedAllocator<EleType>::Reset(int reserved_block_count)
{
	int block_size = (int)block_list_.size();
	if(reserved_block_count < 0) reserved_block_count = 0;
	if(reserved_block_count > block_size) reserved_block_count = block_size;

	for(int i = reserved_block_count; i < block_size; i++)
    {
		delete[] block_list_[i];
		block_list_[i] = nullptr;
	}
	block_list_.resize(reserved_block_count);
	block_size_list_.resize(reserved_block_count);

	cur_block_idx_ = reserved_block_count > 0 ? 0 : -1;
	cur_block_ = reserved_block_count > 0 ? block_list_[0] : nullptr;
	cur_block_size_ = reserved_block_count > 0 ? block_size_list_[0] : 0;
	cur_position_ = 0;

	item_count_ = 0;
}

} //end of namespace
