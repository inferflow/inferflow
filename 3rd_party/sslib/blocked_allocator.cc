#include "blocked_allocator.h"

namespace sslib
{

void BlockedAllocator_ForBuildingOnly()
{
    BlockedAllocator<uint32_t> the_allocator;
    the_allocator.SetBlockSize(10240);
    the_allocator.GetBlockSize();
    the_allocator.GetItemCount();
    the_allocator.New(32);
    uint32_t loc = 0;
    the_allocator.New(5, loc);
    the_allocator.Reset(2);
}

} //end of namespace
