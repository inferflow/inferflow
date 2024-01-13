
add_executable(inferflow_service
    inferflow_service.cc
    inferflow_service_main.cc
    )
target_link_libraries(inferflow_service PRIVATE inferflow ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(inferflow_service PRIVATE cxx_std_11)
if (USE_CUDA)
    target_compile_features(inferflow_service PUBLIC cuda_std_11)
    set_property(TARGET inferflow_service PROPERTY CUDA_ARCHITECTURES ${INFER_FLOW_NVCC_ARCHS_SUPPORTED})
endif()

set(executables_for_install
    inferflow_service
)
install(TARGETS ${executables_for_install} DESTINATION ${CMAKE_INSTALL_PREFIX} COMPONENT inferflow)

