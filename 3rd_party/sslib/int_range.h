#pragma once

#include "prime_types.h"

namespace sslib
{

struct UIntRange
{
    uint32_t start, end;

    explicit UIntRange(uint32_t s = 0, uint32_t e = 0)
    {
        start = s;
        end = e;
    }

    void Set(uint32_t s, uint32_t e)
    {
        start = s;
        end = e;
    }

    uint32_t Length() const
    {
        return end > start ? end - start : 0;
    }

    bool operator < (const UIntRange &rhs) const
    {
        return start == rhs.start ? end < rhs.end : start < rhs.start;
    }

    int Compare(const UIntRange &rhs) const
    {
        return start == rhs.start ? (end < rhs.end ? -1 : (end == rhs.end ? 0 : 1)) : (start < rhs.start ? -1 : 1);
    }

    bool Contains(const UIntRange &rhs) const
    {
        return start <= rhs.start && end >= rhs.end;
    }

    bool IsSub(const UIntRange &rhs) const {
        return start >= rhs.start && end <= rhs.end;
    }

    bool IsProperSub(const UIntRange &rhs) const {
        return (start >= rhs.start && end < rhs.end) || (start > rhs.start && end <= rhs.end);
    }

    bool IsOveralpping(const UIntRange &rhs) const
    {
        return start < rhs.end && rhs.start < end;
    }

    void ExpandStart(uint32_t s)
    {
        if (start > s) {
            start = s;
        }
    }

    void ExpandEnd(uint32_t e)
    {
        if (end < e) {
            end = e;
        }
    }

    void Expand(uint32_t s, uint32_t e)
    {
        if (start > s) {
            start = s;
        }
        if (end < e) {
            end = e;
        }
    }

    void Expand(const UIntRange &range)
    {
        if (start > range.start) {
            start = range.start;
        }
        if (end < range.end) {
            end = range.end;
        }
    }
};

typedef UIntRange OffsetRange;

} //end of namespace
