
add_library(
        my-libs
        src/my-libs/functions.cu
        src/my-libs/functions.cuh
        src/my-libs/Vector.cu src/my-libs/Vector.cuh src/my-libs/Matrix.cu src/my-libs/Matrix.cuh)
#Public means anyone consumes this libary should include this src into
#their include directories, such that they can directly use the modules
#within the this src which will be indicated by my-libs
target_include_directories(my-libs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

#set a preprocessor definition and PUBLIC will propagates this constant
target_compile_definitions(my-libs PUBLIC MY_LIBS_VERSION=1)
set_target_properties(
        my-libs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)