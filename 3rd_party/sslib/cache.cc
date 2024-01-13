#include "cache.h"

namespace sslib
{

void Cache_ForCompilation()
{
    LruCache<int, int> cache;
    cache.Clear();
    cache.Capacity();
    cache.Size();
    int *value = new int(101);
    cache.Add(1, value);
    cache.Get(2);
}

} //end of namespace
