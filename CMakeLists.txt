cmake_minimum_required(VERSION 3.18)

# Optional CUDA language enabling
option(USE_CUDA "Enable CUDA backend" ON)

if(APPLE)
    message(STATUS "🖥 macOS detected: CUDA is not supported. Forcing USE_CUDA=OFF.")
    set(USE_CUDA OFF)
endif()

# Conditionally enable CUDA language
if(USE_CUDA)
    project(CudaRL LANGUAGES CXX CUDA)
else()
    project(CudaRL LANGUAGES CXX)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(USE_CUDA "Enable CUDA backend" ON)

include_directories(include)

# Detect platform
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "🖥 macOS detected: CUDA will be disabled.")
    set(USE_CUDA OFF)
endif()

# Add source files
file(GLOB_RECURSE CORE_SRC src/core/*.cpp)
file(GLOB_RECURSE BACKEND_CPU_SRC src/backends/*_cpu.cpp)

set(SOURCE_FILES ${CORE_SRC} ${BACKEND_CPU_SRC})

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    file(GLOB_RECURSE BACKEND_CUDA_SRC src/backends/*_cuda.cu)
    list(APPEND SOURCE_FILES ${BACKEND_CUDA_SRC})
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
endif()

add_library(cudarl ${SOURCE_FILES})

if(USE_CUDA)
    target_link_libraries(cudarl PRIVATE CUDA::cublas)
endif()

# Test executable
file(GLOB_RECURSE TEST_SRC tests/*.cpp)
add_executable(test_cudarl ${TEST_SRC})
target_link_libraries(test_cudarl PRIVATE cudarl)

enable_testing()
add_test(NAME test_all COMMAND test_cudarl)
