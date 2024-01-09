#pragma once

#include <inttypes.h>
#include <cstddef>
#include <cstring>
#include <wchar.h>

#ifndef INT8
    typedef int8_t      INT8;
#endif
#ifndef INT16
    typedef int16_t     INT16;
#endif
#ifndef INT32
    typedef int32_t     INT32;
#endif
#ifndef INT64
	typedef int64_t     INT64;
#endif
#ifndef UINT8
    typedef uint8_t     UINT8;
#endif
#ifndef UINT16
    typedef uint16_t    UINT16;
#endif
#ifndef UINT32
    typedef uint32_t    UINT32;
#endif
#ifndef UINT64
	typedef uint64_t    UINT64;
#endif

#ifndef Int8
    typedef int8_t		Int8;
#endif
#ifndef Int16
    typedef int16_t     Int16;
#endif
#ifndef Int32
    typedef int32_t     Int32;
#endif
#ifndef Int64
    typedef int64_t     Int64;
#endif
#ifndef UInt8
    typedef uint8_t     UInt8;
#endif
#ifndef UInt16
    typedef uint16_t    UInt16;
#endif
#ifndef UInt32
    typedef uint32_t    UInt32;
#endif
#ifndef UInt64
    typedef uint64_t    UInt64;
#endif

typedef uint32_t Location32;
#define Location32Null UINT32_MAX

#ifdef _WIN32
#	define strcasecmp _stricmp
#	define wcscasecmp _wcsicmp
#   define strncasecmp _strnicmp
#   define wcsncasecmp _wcsnicmp
#	define atoll _atoi64
#endif

#define IN
#define OUT

namespace sslib
{

struct PairUInt32
{
    uint32_t first, second;

    PairUInt32(uint32_t f = 0, uint32_t s = 0)
    {
        first = f; second = s;
    }

    void Set(uint32_t f, uint32_t s)
    {
        first = f; second = s;
    }

    bool operator < (const PairUInt32 &rhs) const
    {
        return first == rhs.first ? second < rhs.second : first < rhs.first;
    }

    bool operator > (const PairUInt32 &rhs) const
    {
        return first == rhs.first ? second > rhs.second : first > rhs.first;
    }
};

#pragma pack(push, 1)
template <class WeightType>
struct IdWeight
{
    uint32_t id;
    WeightType weight;

    IdWeight(uint32_t d = 0, WeightType w = 0)
    {
        id = d; weight = w;
    }
    void Set(uint32_t d = 0, WeightType w = 0)
    {
        id = d; weight = w;
    }

    bool operator == (const IdWeight &rhs) const {
        return this->id == rhs.id;
    }
    bool operator != (const IdWeight &rhs) const
    {
        return this->id != rhs.id;
    }
    bool operator < (const IdWeight &rhs) const
    {
        return this->id < rhs.id;
    }
    bool operator > (const IdWeight &rhs) const
    {
        return this->id > rhs.id;
    }

    static bool LessId(const IdWeight &lhs, const IdWeight &rhs)
    {
        return lhs.id < rhs.id;
    }
    static bool GreaterId(const IdWeight &lhs, const IdWeight &rhs)
    {
        return lhs.id > rhs.id;
    }
    static bool GreaterWeight(const IdWeight &lhs, const IdWeight &rhs)
    {
        return lhs.weight > rhs.weight;
    }
    static bool LessWeight(const IdWeight &lhs, const IdWeight &rhs)
    {
        return lhs.weight < rhs.weight;
    }
};
#pragma pack(pop)

#pragma pack(push, 1)
template <class CharType>
struct IdName
{
    uint32_t id = 0;
    const CharType *name = 0;

    IdName(uint32_t p_id = 0, const CharType *p_name = nullptr)
    {
        id = p_id;
        name = p_name;
    }

    void Set(uint32_t p_id = 0, const CharType *p_name = nullptr)
    {
        id = p_id;
        name = p_name;
    }
};
#pragma pack(pop)

#pragma pack(push, 1)
template <class CharType, class ScoreType = float>
struct IdNameScore
{
    uint32_t id = 0;
    const CharType *name = 0;
    ScoreType score;

    IdNameScore(uint32_t p_id = 0, const CharType *p_name = nullptr, ScoreType p_score = 0)
    {
        id = p_id;
        name = p_name;
        score = p_score;
    }

    void Set(uint32_t p_id = 0, const CharType *p_name = nullptr, ScoreType p_score = 0)
    {
        id = p_id;
        name = p_name;
        score = p_score;
    }

    static bool GreaterScore(const IdNameScore &lhs, const IdNameScore &rhs)
    {
        return lhs.score > rhs.score;
    }

    static bool GreaterScoreLessName(const IdNameScore &lhs, const IdNameScore &rhs)
    {
        if (lhs.score == rhs.score) {
            return wcscasecmp(lhs.name, rhs.name) < 0;
        }
        return lhs.score > rhs.score;
    }

    static bool LessScore(const IdNameScore &lhs, const IdNameScore &rhs)
    {
        return lhs.score < rhs.score;
    }
};
#pragma pack(pop)

}
