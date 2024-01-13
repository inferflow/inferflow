
add_executable(
    inferflow_client
    inferflow_client.cc
)
target_link_libraries(inferflow_client PRIVATE inferflow ${CMAKE_THREAD_LIBS_INIT})

add_executable(
    llm_inference
    llm_inference.cc
)
target_link_libraries(llm_inference PRIVATE inferflow ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llm_inference PRIVATE cxx_std_11)
if (USE_CUDA)
    set_property(TARGET llm_inference PROPERTY CUDA_ARCHITECTURES ${LLAMA_NVCC_ARCHS_SUPPORTED})
endif()

add_executable(
    convert_tokenizer_data
    convert_tokenizer_data.cc
)
target_link_libraries(convert_tokenizer_data PRIVATE inferflow ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(convert_tokenizer_data PRIVATE cxx_std_11)
if (USE_CUDA)
    set_property(TARGET convert_tokenizer_data PROPERTY CUDA_ARCHITECTURES ${LLAMA_NVCC_ARCHS_SUPPORTED})
endif()

set(executables_for_install
    inferflow_client
    llm_inference
    convert_tokenizer_data
)

install(TARGETS ${executables_for_install} DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT inferflow)
