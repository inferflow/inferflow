#pragma once

#include "sslib/log.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "namespace.inc"

INFER_FLOW_BEGIN

bool CutlassSgemm_RowMajor_Float(int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc)
{
    using AType = float;
    using BType = float;
    using CType = float;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;
    using RowMajor = cutlass::layout::RowMajor;
    // whether you want to use tensor cores or regular SIMT cores on GPU SM:
    // {cutlass::arch::OpClassTensorOp, cutlass::arch::OpClassSimt}
    using MMAOp = cutlass::arch::OpClassSimt;
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

    // This code section describes the epilogue part of the kernel
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        CType,                                     // <- data type of output matrix
        128 / cutlass::sizeof_bits<CType>::value,  // <- the number of elements per vectorized
                                                   // memory access. For a byte, it's 16
                                                   // elements. This becomes the vector width of
                                                   // math instructions in the epilogue too
        ElementAccumulator,                            // <- data type of accumulator
        ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

    // Number of pipelines you want to use
    //constexpr int NumStages = 2;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        AType, RowMajor,  //data type and layout of matrix A
        BType, RowMajor,  //data type and layout of matrix B
        CType, RowMajor,  //data type and layout of matrix C
        ElementAccumulator,
        MMAOp,
        cutlass::arch::Sm70/*,
        ShapeMMAThreadBlock,
        ShapeMMAWarp,
        ShapeMMAOp,
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages*/>;

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({ M , N, K },  // Gemm problem dimensions
        { A, lda },    // Tensor-ref for source matrix A
        { B, ldb },    // Tensor-ref for source matrix B
        { C, ldc },    // Tensor-ref for source matrix C
        { C, ldc },    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        { alpha, beta }); // Scalars used in the Epilogue

    // Launch the CUTLASS GEMM kernel.
    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
        LogError("Cutlass error: %d", status);
        return false;
    }

    return true;
}

bool CutlassSgemm_RowMajor_Half(int M, int N, int K,
    float alpha,
    half const *A, int lda,
    half const *B, int ldb,
    float beta,
    half *C, int ldc)
{
    using AType = half;
    using BType = half;
    using CType = half;
    using ElementAccumulator = float;
    using RowMajor = cutlass::layout::RowMajor;
    // whether you want to use tensor cores or regular SIMT cores on GPU SM:
    // {cutlass::arch::OpClassTensorOp, cutlass::arch::OpClassSimt}
    using MMAOp = cutlass::arch::OpClassSimt;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        AType, RowMajor,  //data type and layout of matrix A
        BType, RowMajor,  //data type and layout of matrix B
        CType, RowMajor,  //data type and layout of matrix C
        ElementAccumulator,
        MMAOp,
        cutlass::arch::Sm70>;

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({ M , N, K },  // Gemm problem dimensions
        { A, lda },    // Tensor-ref for source matrix A
        { B, ldb },    // Tensor-ref for source matrix B
        { C, ldc },    // Tensor-ref for source matrix C
        { C, ldc },    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        { alpha, beta }); // Scalars used in the Epilogue

    // Launch the CUTLASS GEMM kernel.
    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
        LogError("Cutlass error: %d", status);
        return false;
    }

    return true;
}

bool CutlassSgemm_ColMajor(int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc)
{
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
        ColumnMajor,  // Layout of A matrix
        float,        // Data-type of B matrix
        ColumnMajor,  // Layout of B matrix
        float,        // Data-type of C matrix
        ColumnMajor>; // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({ M , N, K },  // Gemm Problem dimensions
        { A, lda },    // Tensor-ref for source matrix A
        { B, ldb },    // Tensor-ref for source matrix B
        { C, ldc },    // Tensor-ref for source matrix C
        { C, ldc },    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        { alpha, beta }); // Scalars used in the Epilogue

    // Launch the CUTLASS GEMM kernel.
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        LogError("Cutlass error: %d", status);
        return false;
    }

    return true;
}

bool CutlassSgemm_RowColMajor(int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb, //cutlass::gemm::GemmCoord coord_b,
    float beta,
    float *C, int ldc)
{
    using AType = float;
    using BType = float;
    using CType = float;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;
    using RowMajor = cutlass::layout::RowMajor;
    // whether you want to use tensor cores or regular SIMT cores on GPU SM:
    // {cutlass::arch::OpClassTensorOp, cutlass::arch::OpClassSimt}
    using MMAOp = cutlass::arch::OpClassSimt;

    using RowMajor = cutlass::layout::RowMajor;
    using ColMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        AType, RowMajor,  //data type and layout of matrix A
        BType, ColMajor,  //data type and layout of matrix B
        CType, RowMajor,  //data type and layout of matrix C
        ElementAccumulator,
        MMAOp,
        cutlass::arch::Sm70
        >;

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({ M , N, K },  // Gemm problem dimensions
        { A, lda },    // Tensor-ref for source matrix A
        { B, ldb },    // Tensor-ref for source matrix B
        { C, ldc },    // Tensor-ref for source matrix C
        { C, ldc },    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        { alpha, beta }); // Scalars used in the Epilogue

    // Launch the CUTLASS GEMM kernel.
    cutlass::Status status = gemm_operator(args);
    if (status != cutlass::Status::kSuccess) {
        LogError("Cutlass error: %d", status);
        return false;
    }

    return true;
}

INFER_FLOW_END

