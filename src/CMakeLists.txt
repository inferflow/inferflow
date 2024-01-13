
include_directories(".")

if (USE_CUDA)
    set(GPU_SRC_FILES
        common/cuda_util.cu
        common/quant_cuda.cu
        tensor/cublas_engine.cu
        tensor/device_tensor.cu
        tensor/device_tensor_builder.cu
        tensor/device_memory_heap.cu
        tensor/device_tensor_util.cu
        tensor/tensor_opr.cu
        tensor/tensor_mul.cu
        tensor/parallel_gemm.cu
        transformer/gpu_inf_global_data.cu
        transformer/kv_cache.cc
        transformer/inference_worker.cc
    )
else()
    set(GPU_SRC_FILES)
endif()

add_library(inferflow STATIC
    common/pickle_reader.cc
    common/quantization.cc
    common/std_vocabulary.cc
    common/text_tokenizer.cc
    tensor/host_float_buffer.cc
    tensor/host_tensor.cc
    tensor/host_tensor_opr.cc
    tensor/tensor_common.cc
    tensor/tensor_util.cc
    transformer/inference_engine.cc
    transformer/model.cc
    transformer/model_reader.cc
    transformer/model_writer.cc
    transformer/network_builder.cc
    transformer/network_structure.cc
    transformer/query_state_table.cc
    transformer/sampling_strategy.cc
    transformer/decoding_strategies.cc
    transformer/cpu_inference_worker.cc
    transformer/cpu_kv_cache.cc
    transformer/service_data.cc
    ${GPU_SRC_FILES}
    )
target_link_libraries(inferflow sslib ggml ${CMAKE_THREAD_LIBS_INIT} ${INFERFLOW_EXTRA_LIBS})
target_compile_features(inferflow PRIVATE cxx_std_11)
if (USE_CUDA)
    target_compile_features(inferflow PUBLIC cuda_std_11)
    set_target_properties(inferflow PROPERTIES CUDA_ARCHITECTURES "70;72;75;80")
endif()
set_property(TARGET inferflow PROPERTY CUDA_ARCHITECTURES ${INFER_FLOW_NVCC_ARCHS_SUPPORTED})

add_subdirectory(service)
add_subdirectory(tools)
#add_subdirectory(examples)
#add_subdirectory(unittest)
