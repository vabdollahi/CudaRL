
cmake_minimum_required(VERSION 3.18)
project(CudaRL LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
option(USE_CUDA "Enable CUDA backend" ON)

include_directories(include)

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
endif()

file(GLOB_RECURSE CORE_SRC src/core/*.cpp src/core/*.cu)
file(GLOB_RECURSE BACKEND_SRC src/backends/*.cpp src/backends/*.cu)
add_library(cudarl ${CORE_SRC} ${BACKEND_SRC})

if(USE_CUDA)
    target_compile_definitions(cudarl PRIVATE USE_CUDA)
    target_link_libraries(cudarl PRIVATE CUDA::cublas)
endif()

file(GLOB_RECURSE TEST_SRC tests/*.cpp)
add_executable(test_cudarl ${TEST_SRC})
target_link_libraries(test_cudarl PRIVATE cudarl)

enable_testing()
add_test(NAME test_all COMMAND test_cudarl)
