
cmake_minimum_required(VERSION 3.18)
project(SgemmPortfolio LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)

target_include_directories(sgemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set(SOURCES
    sgemm.cu
)

add_executable(sgemm ${SOURCES})

# --generate-line-info — correct for NCU source correlation
# -Xptxas=-v and -warn-spills — shows profiling awareness
target_compile_options(sgemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)

set_source_files_properties(sgemm.cu
    PROPERTIES COMPILE_OPTIONS
    "-Xptxas=-v;-Xptxas=-warn-spills"
)

target_link_libraries(sgemm PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)



