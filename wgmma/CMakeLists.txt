<pre>
cmake_minimum_required(VERSION 3.18)
project(SgemmPortfolio LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90a")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/tools/util/include)

set(SOURCES
    wgmma_sm90.cu
)

add_executable(wgmma_sm90 ${SOURCES})

target_compile_options(wgmma_sm90 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --generate-line-info
        -O3
        --resource-usage
        -UCUTLASS_ENABLE_SYNCLOG
    >
)

target_compile_definitions(wgmma_sm90 PRIVATE
    CUTLASS_ARCH_MMA_SM90_SUPPORTED
)

set_source_files_properties(wgmma_sm90.cu
    PROPERTIES COMPILE_OPTIONS
    "-Xptxas=-v;-Xptxas=-warn-spills"
)

#set_target_properties(wgmma_sm90 PROPERTIES CUDA_ARCHITECTURES "90a-real;90a-virtual")


set_property(TARGET wgmma_sm90 PROPERTY CUDA_ARCHITECTURES 90a)

target_link_libraries(wgmma_sm90 PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    OpenMP::OpenMP_CXX
)


</pre>
