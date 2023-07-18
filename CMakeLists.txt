cmake_minimum_required(VERSION 3.12...3.18)
project(fake_custom_call_ext LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -ldl")
set(USING_WMMA False)
set(SM_SETS 75 80 86)
foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")
    set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
  endif()
endforeach()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
  gpu_ops
${CMAKE_CURRENT_LIST_DIR}/lib/fake_custom_call.cu
${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc)
target_link_libraries(gpu_ops PUBLIC -lcudart -lcublas -lcublasLt)
install(TARGETS gpu_ops DESTINATION fake_custom_call_ext)
