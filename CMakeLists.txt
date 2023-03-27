# CmakeLists for building python bindings
cmake_minimum_required(VERSION 3.0)

project(llamacpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GGML_USE_ACCELERATE 1)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(vendor/llama.cpp)
pybind11_add_module(llamacpp MODULE src/llama2.cpp src/llama_wrapper.cpp src/llama_wrapper.h)
target_include_directories(llamacpp PRIVATE vendor/llama.cpp)
target_link_libraries(llamacpp PRIVATE pybind11::module pybind11::lto pybind11::windows_extras llama)
add_link_options(-no_fixup_chains)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(llamacpp)
endif()

set_target_properties(llamacpp PROPERTIES CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden")

install(TARGETS llamacpp DESTINATION llamacpp)
