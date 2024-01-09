#include "string_vector.h"

namespace sslib
{

void BasicStringVector_ForCompilation()
{
    U16StringVector sv;
    sv.Add(u"str1");
    sv.Clear();
    sv.Get(0);
    sv.Load("");
    sv.Save("");
    sv.Size();
}

} //end of namespace
