cmake_minimum_required(VERSION 3.18)
project(ms_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------
# pybind11
# -----------------------------------------------------------------------
find_package(pybind11 QUIET)
if (NOT pybind11_FOUND)
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 REQUIRED)
endif()

# -----------------------------------------------------------------------
# OpenMP (required)
# -----------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# -----------------------------------------------------------------------
# Optional CUDA support  (-DWITH_CUDA=ON)
#
# GPU architecture: -DCMAKE_CUDA_ARCHITECTURES=<cc>, default 80 (A100).
# Examples: 70 (V100), 80 (A100), 86 (RTX 30xx), 89 (L40/RTX 40xx), 90 (H100).
# -----------------------------------------------------------------------
option(WITH_CUDA "Build with CUDA/GPU support" OFF)

if(WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Build: CPU + GPU (OpenMP + CUDA, sm_${CMAKE_CUDA_ARCHITECTURES})")

    pybind11_add_module(ms_ops
        src/ms_ops.cpp
        src/ms_ops_cuda.cu
    )
    target_compile_definitions(ms_ops PRIVATE WITH_CUDA)
    # no --use_fast_math: the phases need full double precision
    target_compile_options(ms_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    target_link_libraries(ms_ops PRIVATE
        OpenMP::OpenMP_CXX
        CUDA::cudart_static
    )
else()
    message(STATUS "Build: CPU only (OpenMP)")
    pybind11_add_module(ms_ops src/ms_ops.cpp)
    target_compile_options(ms_ops PRIVATE -O3)
    target_link_libraries(ms_ops PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(ms_ops PRIVATE ${CMAKE_SOURCE_DIR}/src)

# -----------------------------------------------------------------------
# Install the compiled module into the vista package directory
# -----------------------------------------------------------------------
install(TARGETS ms_ops
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/vista
)
