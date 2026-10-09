cmake_minimum_required(VERSION 3.28)
project(WgmmaSm90 LANGUAGES CXX CUDA)

# compiling with release mode
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "90a")


find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/tools/util/include)




set(SOURCES
    wgmma_sm90.cu
)

add_executable(wgmma_sm90 wgmma_sm90.cu)

# Hopper with arch-specific features (wgmma requires sm_90a)
set_property(TARGET wgmma_sm90 PROPERTY CUDA_ARCHITECTURES 90a)


target_include_directories(wgmma_sm90 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include            # remove if unused
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/tools/util/include
)

# The only source is a .cu file, so these flags apply to CUDA compilation.
target_compile_options(wgmma_sm90 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --generate-line-info
        -Xptxas=-v 
        -Xptxas=-warn-spills
        -O3
        --resource-usage
        -UCUTLASS_ENABLE_SYNCLOG
    >
)

set_source_files_properties(wgmma_sm90.cu
    PROPERTIES COMPILE_OPTIONS
    "-Xptxas=-v;-Xptxas=-warn-spills"
)


set_property(TARGET wgmma_sm90 PROPERTY CUDA_ARCHITECTURES 90a)

target_link_libraries(wgmma_sm90 PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
#    CUDA::nvToolsExt
    OpenMP::OpenMP_CXX
)

if(CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Darwin" OR UNIX)
    add_custom_target(analyze
        # 1. Clear out any previous sass files to ensure fresh results
        COMMAND ${CMAKE_COMMAND} -E rm -f "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass"
        
        # 2. Extract assembly using the exact location of the target binary
        COMMAND cuobjdump -sass $<TARGET_FILE:wgmma_sm90> > "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass"
        
        # 3. Print analysis metrics safely
        COMMAND echo "--- HGMMA Count ---"
        COMMAND grep -c "HGMMA" "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass" || true
        
        # 4. Print Warpgroup context
        COMMAND echo "--- WARPGROUP Lines ---"
        COMMAND grep -n "WARPGROUP" "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass" | head -n 10 || true
        
        # 5. Print Calls
        COMMAND echo "--- CALL Lines ---"
        COMMAND grep -n "CALL" "${CMAKE_CURRENT_BINARY_DIR}/wgmma_release.sass" | head -n 10 || true
        
        # 6. Extract PTX calls and checks
        COMMAND echo "--- PTX Calls/Asserts ---"
        COMMAND cuobjdump -ptx $<TARGET_FILE:wgmma_sm90> | grep -E "call|__assertfail" | head -n 10 || true
        
        DEPENDS wgmma_sm90
        COMMENT "Inspecting SASS and PTX outputs for Hopper SM90 instructions..."
        VERBATIM
    )
endif()

