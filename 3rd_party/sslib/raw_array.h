#pragma once

#include "prime_types.h"
#include <string>
#include <cstring>

namespace sslib
{

#pragma pack(push, 1)
template <class EleType>
struct ConstRawArray
{
public:
    uint32_t size = 0;
    const EleType *data = nullptr;

public:
    explicit ConstRawArray(const EleType *data_ptr = nullptr, uint32_t len = 0)
    {
        size = len;
        data = data_ptr;
    }
    ~ConstRawArray() {};

    const EleType& operator [] (uint32_t idx) const
    {
        return data[idx];
    }

    void Set(const EleType *data_ptr, uint32_t len)
    {
        size = len;
        data = data_ptr;
    }

    void Clear()
    {
        size = 0;
        data = nullptr;
    }
};
#pragma pack(pop)

#pragma pack(push, 1)
template <class EleType>
struct RawArray
{
public:
    uint32_t size = 0;
    EleType *data = nullptr;

public:
    explicit RawArray(EleType *data_ptr = nullptr, uint32_t len = 0)
    {
        size = len;
        data = data_ptr;
    }
    ~RawArray(){};

    const EleType& operator [] (uint32_t idx) const
    {
        return data[idx];
    }

    EleType& operator [] (uint32_t idx)
    {
        return data[idx];
    }

    void Set(EleType *data_ptr, uint32_t len)
    {
        size = len;
        data = data_ptr;
    }

	void Renew(uint32_t len)
	{
		Delete();
		New(len);
	}

	void New(uint32_t len)
	{
		size = len;
		data = len > 0 ? new EleType[len] : nullptr;
	}

    void Clear()
    {
        size = 0;
        data = nullptr;
    }

	void Delete()
	{
		size = 0;
        if(data != nullptr)
        {
            delete[] data;
            data = nullptr;
        }
	}
};
#pragma pack(pop)

typedef ConstRawArray<char> ConstRawString;
typedef RawArray<char> RawString;
typedef ConstRawArray<wchar_t> ConstRawWString;
typedef RawArray<wchar_t> RawWString;
struct RawStringLess
{
    bool operator () (const RawString &lhs, const RawString &rhs) const {
        return strcmp(lhs.data, rhs.data) < 0;
    }

    bool operator () (const ConstRawString &lhs, const ConstRawString &rhs) const {
        return strcmp(lhs.data, rhs.data) < 0;
    }

    bool operator () (const RawWString &lhs, const RawWString &rhs) const {
        return wcscmp(lhs.data, rhs.data) < 0;
    }

    bool operator () (const ConstRawWString &lhs, const ConstRawWString &rhs) const {
        return wcscmp(lhs.data, rhs.data) < 0;
    }
};

struct RawWStrStream
{
    const wchar_t *data = nullptr;
    uint32_t size = 0;
    uint32_t offset = 0;

    RawWStrStream(const wchar_t *the_data = nullptr, uint32_t the_size = 0)
    {
        data = the_data;
        size = the_size;
        offset = 0;
    }
};

} //end of namespace
