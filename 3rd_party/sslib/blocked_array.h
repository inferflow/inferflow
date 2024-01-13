#pragma once

#include <vector>
#include <cassert>
#include <algorithm>
#include "prime_types.h"

namespace sslib
{

template <class EleType>
class BlockedArray
{
public:
    static const int MinBlockSize = 1;
    static const int DefaultBlockSize = 10240;

public:
	explicit BlockedArray(uint32_t block_size = DefaultBlockSize)
    {
		block_size_ = block_size;
		block_list_.clear();

        cur_block_idx_ = UINT64_MAX;
		cur_block_ptr_ = nullptr;
		new_position_ = 0;
	}

	virtual ~BlockedArray(void) {
		Clear();
	}

	// Set block size (number of elements). Can take effect only when there is no elements in the MultiArray
	// @param block_size: New block size
	bool SetBlockSize(uint32_t block_size)
    {
		if(block_list_.size() == 0)
        {
			block_size_ = block_size < MinBlockSize ? MinBlockSize : block_size;
			return true;
		}
		return false;
	}

	uint32_t GetBlockSize() const
    {
		return block_size_;
	}

    uint64_t Size() const
    {
        return cur_block_idx_ != (uint64_t)(-1) ? (cur_block_idx_ * block_size_ + new_position_)
            : (uint64_t)new_position_;
    }

    uint64_t size() const
    {
        return cur_block_idx_ != (uint64_t)(-1) ? (cur_block_idx_ * block_size_ + new_position_)
            : (uint64_t)new_position_;
    }

	bool ReserveBlocks(uint64_t block_count)
    {
        uint64_t cur_size = (uint64_t)block_list_.size();
        uint64_t new_block_num = block_count >= cur_size ? block_count - cur_size : 0;
		for(uint64_t i = 0; i < new_block_num; i++)
        {
			EleType *p = new EleType[block_size_];
			if(p == nullptr) {
				return false;
			}
			block_list_.push_back(p);
		}

		return true;
	}

	const EleType& Get(uint64_t idx) const
    {
		assert((size_t)idx / block_size_ < block_list_.size());
		EleType *block_ptr = block_list_[(size_t)idx / block_size_];
		return block_ptr[(size_t)idx % block_size_];
	}

    EleType& Get(uint64_t idx)
    {
		assert((size_t)idx / block_size_ < block_list_.size());
		EleType *block_ptr = block_list_[(size_t)idx / block_size_];
		return block_ptr[(size_t)idx % block_size_];
	}

    const EleType& operator[](uint64_t idx) const
    {
        return Get(idx);
    }

    EleType& operator[](uint64_t idx)
    {
        return Get(idx);
    }

    void push_back(const EleType &item)
    {
        Append(item);
    }

	void Append(const EleType &item)
    {
		EleType *item_ptr = New();
		(*item_ptr) = item;
	}

	EleType* New()
    {
		if(cur_block_ptr_ == nullptr || new_position_ >= block_size_)
        {
			cur_block_idx_++;
			if(cur_block_idx_ < block_list_.size())
            {
				cur_block_ptr_ = block_list_[(size_t)cur_block_idx_];
			}
			else
            {
				cur_block_ptr_ = new EleType[block_size_];
				if(cur_block_ptr_ == nullptr) {
					return nullptr;
				}
				block_list_.push_back(cur_block_ptr_);
			}

			new_position_ = 0;
		}

		EleType *ret = cur_block_ptr_ + new_position_;
		new_position_++;

		return ret;
	}

    void clear(uint64_t reserved_block_count = 0)
    {
        Clear(reserved_block_count);
    }

    void Clear(uint64_t reserved_block_count = 0)
    {
        uint64_t block_count = (uint64_t)block_list_.size();
        if (reserved_block_count > block_count) {
            reserved_block_count = block_count;
        }

        for (size_t i = (size_t)reserved_block_count; i < (size_t)block_count; i++)
        {
            delete[] block_list_[i];
            block_list_[i] = nullptr;
        }
        block_list_.resize((size_t)reserved_block_count);

        cur_block_idx_ = reserved_block_count > 0 ? 0 : UINT64_MAX;
        cur_block_ptr_ = reserved_block_count > 0 ? block_list_[0] : nullptr;
        new_position_ = 0;
    }

    void resize(uint64_t new_size)
    {
        Resize(new_size);
    }

    void Resize(uint64_t new_size)
    {
        uint64_t old_block_count = (uint64_t)block_list_.size();
        uint64_t new_block_count = (new_size + block_size_) / block_size_;
        if(new_block_count < old_block_count)
        {
            for(size_t iBlock = (size_t)new_block_count; iBlock < (size_t)old_block_count; iBlock++)
            {
                if(block_list_[iBlock] != nullptr) {
                    delete[] block_list_[iBlock];
                    block_list_[iBlock] = nullptr;
                }
            }
            block_list_.resize((size_t)new_block_count);
        }
        else if(new_block_count > old_block_count)
        {
            for(size_t iBlock = (size_t)old_block_count; iBlock < (size_t)new_block_count; iBlock++)
            {
                EleType *block_ptr = new EleType[block_size_];
                if(block_ptr == nullptr) {
                    //TODO
                }
                block_list_.push_back(block_ptr);
            }
        }

        cur_block_idx_ = new_block_count - 1;
	    cur_block_ptr_ = cur_block_idx_ != (uint64_t)(-1) ? block_list_[(size_t)cur_block_idx_] : nullptr;
        new_position_ = new_size % block_size_;
    }

    void erase(uint64_t start_idx, uint64_t end_idx)
    {
        Erase(start_idx, end_idx);
    }

    void Erase(uint64_t start_idx, uint64_t end_idx)
    {
        uint64_t size = (uint64_t)Size();
        if(start_idx >= end_idx || end_idx > size) {
            return;
        }

        for(uint64_t item_idx = 0; end_idx + item_idx < size; item_idx++) {
            Get((int)start_idx + item_idx) = Get(end_idx + item_idx);
        }
        Resize(size + start_idx - end_idx);
    }

private:
    uint32_t block_size_ = DefaultBlockSize;
    std::vector<EleType*> block_list_;  //all blocks

    uint64_t cur_block_idx_; //The index of new block
    EleType *cur_block_ptr_ = nullptr; //New memory block. New items will be added into this block
    uint32_t new_position_ = 0; //New element position, where new elements will be added
};

} //end of namespace
