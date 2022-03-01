add_executable(lib-exe main.cu)

#
target_link_libraries(lib-exe PRIVATE my-libs)

set_target_properties(
        lib-exe
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)