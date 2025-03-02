cmake_minimum_required(VERSION 3.20)
project(RapidAlign LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# CPU-only implementation for testing/debugging core algorithm
add_executable(CPUTest main.cpp)

# CPU-based graph visualization test
add_executable(GraphAlignTest visualize_graph_alignment.cpp)

# Only build CUDA project if CUDA is available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  
  # Single-instance CUDA implementation
  add_executable(MyCudaProject pointcloud_alignment.cu)
  
  # Batched CUDA implementation
  add_executable(BatchedAlign batch_alignment.cu)
  
  # Test program with graph visualization
  add_executable(TestBatchAlign test_batch_alignment.cu)
  
  # You can set additional CUDA compilation flags
  set_target_properties(BatchedAlign PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"  # Specify target architectures
  )
  
  set_target_properties(TestBatchAlign PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"  # Specify target architectures
  )
else()
  message(STATUS "CUDA not found - skipping CUDA implementation")
endif()
