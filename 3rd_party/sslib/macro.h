#pragma once

namespace sslib
{

#ifndef IN
#define IN
#endif //ifndef IN

#ifndef OUT
#define OUT
#endif //ifndef OUT

#define Macro_FloatEqual(a, b) (fabs((a) - (b)) < 0.00001f)
#define Macro_FloatLarger(a, b) ((a) - (b) > 0.00001f)
#define Macro_FloatSmaller(a, b) ((a) - (b) < -0.00001f)
#define Macro_DoubleEqual(a, b) (fabs(a - b) < 0.0000000001)
#define Macro_DoubleLarger(a, b) ((a) - (b) > 0.0000000001)
#define Macro_DoubleSmaller(a, b) ((a) - (b) < 0.0000000001)

#define Macro_RetIf(value, condition) if(condition) { return value; }
#define Macro_RetTrueIf(condition) if(condition) { return true; }
#define Macro_RetFalseIf(condition) if(condition) { return false; }
#define Macro_RetNullptrIf(condition) if(condition) { return nullptr; }
#define Macro_RetVoidIf(condition) if(condition) { return; }
#define Macro_RetxIf(value, condition, clause) if(condition) { clause; return value; }
#define Macro_RetxTrueIf(condition, clause) if(condition) { clause; return true; }
#define Macro_RetxFalseIf(condition, clause) if(condition) { clause; return false; }
#define Macro_RetxNullptrIf(condition, clause) if(condition) { clause; return nullptr; }
#define Macro_RetxVoidIf(condition, clause) if(condition) { clause; return; }

#ifdef _WIN32
#   define HiddenAttribute
#   define DefaultAttribute
#else
#   if __GNUC__ >= 4
#       define HiddenAttribute __attribute__((visibility("hidden")))
#       define DefaultAttribute __attribute__((visibility("default")))
#   else
#       define HiddenAttribute
#       define DefaultAttribute
#   endif
#endif

} //end of namespace
