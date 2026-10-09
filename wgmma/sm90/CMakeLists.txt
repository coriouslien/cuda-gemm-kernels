cmake_minimum_required(VERSION 3.28)
project(WgmmaSm90 LANGUAGES CXX CUDA)

# compiling with release mode
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)     # print every full compile/link command
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_ARCHITECTURES "90a")


find_package(CUDAToolkit REQUIRED)

add_executable(wgmma_sm90 wgmma_sm90.cu)

# Hopper with arch-specific features (wgmma requires sm_90a).
# "90a" embeds both PTX (compute_90a) and SASS (sm_90a).
set_property(TARGET wgmma_sm90 PROPERTY CUDA_ARCHITECTURES 90a)

target_include_directories(wgmma_sm90 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/tools/util/include
)


# The only source is a .cu file, so these flags apply to CUDA compilation.
# -O3 and -DNDEBUG come from CMAKE_BUILD_TYPE=Release.
target_compile_options(wgmma_sm90 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --generate-line-info
        -Xptxas=-v 
        -Xptxas=-warn-spills
        --resource-usage
        -UCUTLASS_ENABLE_SYNCLOG
    >
)



if(UNIX)
    set(SASS_FILE "${CMAKE_CURRENT_BINARY_DIR}/wgmma_${CMAKE_BUILD_TYPE}.sass")
    set(CUOBJDUMP "${CUDAToolkit_BIN_DIR}/cuobjdump")

    add_custom_target(analyze
        COMMAND ${CMAKE_COMMAND} -E rm -f "${SASS_FILE}"
        COMMAND ${CUOBJDUMP} -sass $<TARGET_FILE:wgmma_sm90> > "${SASS_FILE}"
        COMMAND echo "--- HGMMA Count (all kernels) ---"
        COMMAND grep -c "HGMMA" "${SASS_FILE}" || true
        COMMAND echo "--- WARPGROUP Lines ---"
        COMMAND grep -n "WARPGROUP" "${SASS_FILE}" | head -n 10 || true
        COMMAND echo "--- CALL Lines ---"
        COMMAND grep -n "CALL" "${SASS_FILE}" | head -n 10 || true
        COMMAND echo "--- PTX Calls/Asserts ---"
        COMMAND ${CUOBJDUMP} -ptx $<TARGET_FILE:wgmma_sm90> | grep -E "call|__assertfail" | head -n 10 || true
        COMMENT "Inspecting SASS and PTX outputs for Hopper SM90 instructions..."
        VERBATIM
    )
    add_dependencies(analyze wgmma_sm90)
endif()
