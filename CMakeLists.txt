cmake_minimum_required(VERSION 3.12)
project("InferFlow" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../../bin)

#
# Option list
#

# general
option(STATIC_LINKING                   "enable static linking"                         OFF)
option(ENABLE_NATIVE                    "enable -march=native flag"                     OFF)
option(ENABLE_LTO                       "Enable link time optimization"                 OFF)

# debug
option(ENABLE_ALL_WARNINGS              "Enable all compiler warnings"                  ON)
option(ENABLE_ALL_WARNINGS_3RD_PARTY    "Enable compiler warnings in 3rd party libs"    OFF)

# instruction set specific
option(ENABLE_AVX                       "Enable AVX"                                    ON)
option(ENABLE_AVX2                      "Enable AVX2"                                   ON)
option(ENABLE_AVX512                    "Enable AVX512"                                 OFF)
option(ENABLE_AVX512_VBMI               "Enable AVX512-VBMI"                            OFF)
option(ENABLE_AVX512_VNNI               "Enable AVX512-VNNI"                            OFF)
option(ENABLE_FMA                       "Enable FMA"                                    ON)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(ENABLE_F16C                  "Enable F16C"                                   ON)
endif()

# 3rd party libs
option(USE_CUDA                         "Use cuda (GPU support)"                        ON)
option(ENABLE_ACCELERATE                "Enable Accelerate framework"                   ON)
option(USE_BLAS                         "Use BLAS"                                      OFF)
option(USE_CUTLASS                      "Use cutlass"                                   OFF)
option(USE_CLBLAST                      "Use CLBlast"                                   OFF)
set(BLAS_VENDOR "Generic" CACHE STRING  "BLAS library vendor")

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(build_type "release")
else()
    #set(build_type ${CMAKE_BUILD_TYPE})
    string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#SET(CMAKE_INSTALL_RPATH "$ORIGIN")
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}")

message(STATUS "build_type: ${build_type}")
message(STATUS "CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")

if(${STATIC_LINKING})
    message("Static linking is enabled!")
    if(APPLE)
        #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_CONFIG_H -fPIC -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
if(WIN32)
else()
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/bin/${build_type})

    #set(OPENSSL_LIBRARIES)
    #find_library(temp libssl.a PATHS "${CMAKE_SOURCE_DIR}/3rd_party/openssl/lib/" )
    #list(APPEND OPENSSL_LIBRARIES ${temp})
    #unset(temp CACHE)

    #find_library(temp libcrypto.a PATHS "${CMAKE_SOURCE_DIR}/3rd_party/openssl/lib/" )
    #list(APPEND OPENSSL_LIBRARIES ${temp})
    #unset(temp CACHE)
endif()

#if(${STATIC_LINKING})
#	set(openssl_lib "OpenSSL::SSL")
#	set(crypto_lib "OpenSSL::Crypto")
#else()
#	set(openssl_lib "ssl")
#	set(crypto_lib "crypto")
#endif()

if (APPLE AND ENABLE_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(INFERFLOW_EXTRA_LIBS ${INFERFLOW_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (USE_BLAS)
    if (STATIC_LINKING)
        set(BLA_STATIC ON)
    endif()
    if ($(CMAKE_VERSION) VERSION_GREATER_EQUAL 3.22)
        set(BLA_SIZEOF_INTEGER 8)
    endif()
    set(BLA_VENDOR ${BLAS_VENDOR})
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

        add_compile_options(${BLAS_LINKER_FLAGS})
        add_compile_definitions(GGML_USE_OPENBLAS)
        set(INFERFLOW_EXTRA_LIBS ${INFERFLOW_EXTRA_LIBS} ${BLAS_LIBRARIES})

        message("${BLAS_LIBRARIES} ${BLAS_INCLUDE_DIRS}")
        include_directories(${BLAS_INCLUDE_DIRS})
    else()
        message(WARNING "BLAS not found, please refer to "
        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
        " to set correct BLAS_VENDOR")
    endif()
endif()

if (USE_CUDA)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit REQUIRED)
    find_package(CUDA REQUIRED)

    set(INFER_FLOW_NVCC_ARCHS_SUPPORTED "")
    if (NOT CUDA_VERSION VERSION_LESS 9.0)
        list(APPEND INFER_FLOW_NVCC_ARCHS_SUPPORTED 70)
    endif()
    if (NOT CUDA_VERSION VERSION_LESS 9.2)
        list(APPEND INFER_FLOW_NVCC_ARCHS_SUPPORTED 72)
    endif()
    if (NOT CUDA_VERSION VERSION_LESS 10.0)
        list(APPEND INFER_FLOW_NVCC_ARCHS_SUPPORTED 75)
    endif()
    if (NOT CUDA_VERSION VERSION_LESS 11.0)
        list(APPEND INFER_FLOW_NVCC_ARCHS_SUPPORTED 80)
    endif()

    message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
    set(CUDA_ARCHITECTURES, ${INFER_FLOW_NVCC_ARCHS_SUPPORTED})
    message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")
    message(STATUS "INFER_FLOW_NVCC_ARCHS_SUPPORTED: ${INFER_FLOW_NVCC_ARCHS_SUPPORTED}")

    if (CUDAToolkit_FOUND)
        message(STATUS "cuda_toolkit found")

        enable_language(CUDA)

        include_directories("${CUDA_INCLUDE_DIRS}")

        add_compile_definitions(USE_CUDA)
        if (USE_CUTLASS)
            add_compile_definitions(USE_CUTLASS)
        endif()

        if (USE_CUDA)
            if (STATIC_LINKING)
                set(INFERFLOW_EXTRA_LIBS ${INFERFLOW_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            else()
                set(INFERFLOW_EXTRA_LIBS ${INFERFLOW_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
            endif()
        else()
            if (STATIC_LINKING)
                set(INFERFLOW_EXTRA_LIBS ${INFERFLOW_EXTRA_LIBS} CUDA::cudart_static)
            else()
                set(INFERFLOW_EXTRA_LIBS ${INFERFLOW_EXTRA_LIBS} CUDA::cudart)
            endif()
        endif()
    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (USE_CLBLAST)
    find_package(CLBlast)
    if (CLBlast_FOUND)
        message(STATUS "CLBlast found")

        add_compile_definitions(GGML_USE_CLBLAST)

        set(INFERFLOW_EXTRA_LIBS ${INFERFLOW_EXTRA_LIBS} clblast)
    else()
        message(WARNING "CLBlast not found")
    endif()
endif()

if (ENABLE_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
        )
    else()
    endif()

    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (STATIC_LINKING)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (ENABLE_GPROF)
        add_compile_options(-pg)
    endif()
    if (ENABLE_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            # Apple M1, M2, etc.
            # Raspberry Pi 3, 4, Zero 2 (64-bit)
            add_compile_options(-mcpu=native)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            # Raspberry Pi 2
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (ENABLE_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (ENABLE_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (ENABLE_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (ENABLE_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (ENABLE_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (ENABLE_F16C)
            add_compile_options(-mf16c)
        endif()
        if (ENABLE_FMA)
            add_compile_options(-mfma)
        endif()
        if (ENABLE_AVX)
            add_compile_options(-mavx)
        endif()
        if (ENABLE_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (ENABLE_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
        if (ENABLE_AVX512_VBMI)
            add_compile_options(-mavx512vbmi)
        endif()
        if (ENABLE_AVX512_VNNI)
            add_compile_options(-mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    add_compile_options(-mcpu=native -mtune=native)
    #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
else()
    message(STATUS "Unknown architecture")
endif()

if (USE_CUTLASS)
    LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rd_party/cutlass/lib64/)
endif()

message("INFERFLOW_EXTRA_LIBS: ${INFERFLOW_EXTRA_LIBS}")

include_directories("src/")
include_directories("3rd_party/")
#include_directories("3rd_party/openssl/include/")
if (USE_CUTLASS)
    include_directories("3rd_party/cutlass/include/")
endif()

set(bin_dir ${PROJECT_BINARY_DIR})
message("bin_dir: ${bin_dir}")

add_subdirectory(3rd_party/sslib ${bin_dir}/sslib)

#
# Build libraries
#

add_library(ggml OBJECT
    3rd_party/ggml/ggml.c
    3rd_party/ggml/ggml.h)
target_include_directories(ggml PUBLIC .)
target_compile_features(ggml PUBLIC c_std_11)
target_link_libraries(ggml PUBLIC Threads::Threads ${INFERFLOW_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(src)
