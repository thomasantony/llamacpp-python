# CmakeLists for building python bindings
cmake_minimum_required(VERSION 3.0)

project(llamacpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GGML_USE_ACCELERATE 1)
find_package(pybind11 REQUIRED)

add_library(llamacpp MODULE src/PyLlama.cpp ../llama.cpp ../ggml.c ../utils.cpp)

target_include_directories(llamacpp PRIVATE ../)

target_link_libraries(llamacpp PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
add_link_options(-no_fixup_chains)
pybind11_extension(llamacpp)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(llamacpp)
endif()

set_target_properties(llamacpp PROPERTIES CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden")
