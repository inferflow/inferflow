#pragma once

#include <string>
#include <map>
#include "sslib/string.h"
#include "common/quant_types.h"
#include "common/data_types.h"

INFER_FLOW_BEGIN

using std::string;
using std::map;
using sslib::StrLessNoCase;

enum class ElementType
{
    F32 = 0,    //float
    F16,        //float16
    BF16,       //bfloat16 (brain floating point 16 bits)
    Q8_GL,      //8-bit quantization: global linear
    Q8_LOG,     //8-bit quantization: global logarithmic
    Q8_B32T1,   //8-bit quantization: block linear (block capacity 32, type-1)
    Q8_B32T2,   //8-bit quantization: block linear (block capacity 32, type-2)
    Q6_B64T1,   //6-bit quantization: block linear (block capaticy 64, type-1)
    Q5_B64T1,
    Q5_B32T1,
    Q4_B16,
    Q4_B32T1A,
    Q4_B32T1B,
    Q4_B32T2,
    Q4_B32P8,
    Q4_B64T1,
    Q3H_B64T1,
    Q3_B32T1A,
    Q3_B32T1B,
    Q2_B32T1A,
    Q2_B32T1B,
    Invalid,
    Auto = 99
};

typedef map<string, ElementType, StrLessNoCase> ElementTypeMap;
typedef map<ElementType, string> ElementTypeNameMap;

enum class TensorNormAlg
{
    STD,
    RMS,
    LINEAR
};

enum class ActivationFn
{
    SIGMOID,
    ELU,            //Exponential Linear Unit
    RELU,           //Rectified linear unit
    GELU,           //Gaussian error linear unit
    SILU,           //Sigmoid Linear Units (another name: SwiGLU)
    GLU_SIGMOID,    //GLU (Gated Linear Units) with SIGMOID as the sigma function
    GLU_ELU,        //GLU with ELU as the sigma function
    GLU_RELU,       //GLU with RELU as the sigma function
    GLU_GELU,       //GLU with GELU as the sigma function
    GLU_SILU        //GLU with SILU as the sigma function
};

enum class PositionEmbeddingAlg
{
    EMPTY,
    ROPE,
    ALIBI,
    SINUSOIDAL,
    SINUSOIDAL2
};

enum class MatrixMulAlg
{
    Auto = 0,
    Alg1,
    Alg2,
    Alg3,
    Cublas,
    Cutlass,
    Bruce
};

enum class VectorMatrixMulAlg
{
    Auto = 0,
    Alg1,
    Alg2,
    Alg3,
    Alg4,
    Cublas
};

enum class RowAggregationType
{
    Sum = 0, Max, Min
};

enum class TensorPartitionType
{
    DUP = 0, //duplicate
    BY_ROW,
    BY_COL
};

#pragma pack(push, 1)
struct SparseMatrixCell
{
    uint16_t row = 0;
    uint16_t col = 0;
    float score = 0;

    SparseMatrixCell(uint16_t p_row = 0, uint16_t p_col = 0, float p_score = 0)
    {
        this->row = p_row;
        this->col = p_col;
        this->score = p_score;
    }

    bool operator < (const SparseMatrixCell &rhs)
    {
        return this->row == rhs.row ? this->col < rhs.col : this->row < rhs.row;
    }
};
#pragma pack(pop)

class TensorCommon
{
public:
    static bool IsQuantType(ElementType etype);

    //Bytes per element
    static int ElementSize(ElementType etype);

    //Bytes per block
    static int BlockSize(ElementType etype);

    //Number of elements per block
    static int BlockCapacity(ElementType etype);

    //Return the bytes of num elements
    static uint64_t ByteCount(ElementType etype, uint64_t element_num);

    static void InitElementTypeMap(ElementTypeMap &type_map);
    static void InitElementTypeMap(ElementTypeNameMap &type_name_map);
};

INFER_FLOW_END
