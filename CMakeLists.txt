cmake_minimum_required(VERSION 3.15)
project(dot_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_OPENMP "Enable OpenMP parallel version" ON)

if (ENABLE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP, enabling parallel version")
        add_compile_definitions(_OPENMP)
        set(OpenMP_FLAGS "${OpenMP_CXX_FLAGS}")
    else()
        message(WARNING "OpenMP requested but not found; falling back to single-threaded")
    endif()
endif()

add_executable(dot_bench dot_bench.cpp)

if (ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(dot_bench PRIVATE OpenMP::OpenMP_CXX)
endif()
