# CmakeLists for building python bindings
cmake_minimum_required(VERSION 3.0)

project(llamacpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GGML_USE_ACCELERATE 1)
find_package(pybind11 REQUIRED)

add_subdirectory(vendor/llama.cpp)
add_library(llamacpp MODULE src/PyLlama.cpp)

target_include_directories(llamacpp PRIVATE vendor/llama.cpp)
target_link_libraries(llamacpp PRIVATE pybind11::module pybind11::lto pybind11::windows_extras llama)
add_link_options(-no_fixup_chains)
pybind11_extension(llamacpp)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(llamacpp)
endif()

set_target_properties(llamacpp PROPERTIES CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden")

install(TARGETS llamacpp LIBRARY DESTINATION llamacpp)
