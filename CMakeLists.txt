cmake_minimum_required(VERSION 3.16)
project(hanzo-bot-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Vendored BLAKE3 reference implementation. Wallet addresses are
# content-addressable; every brain runtime hashes with BLAKE3 so this
# C++ port produces byte-equivalent output with TS / Python / Go / Rust.
set(BLAKE3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blake3)
set(BLAKE3_SRC
    ${BLAKE3_DIR}/blake3.c
    ${BLAKE3_DIR}/blake3_dispatch.c
    ${BLAKE3_DIR}/blake3_portable.c
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    list(APPEND BLAKE3_SRC ${BLAKE3_DIR}/blake3_neon.c)
endif()
add_library(blake3 STATIC ${BLAKE3_SRC})
target_include_directories(blake3 PUBLIC ${BLAKE3_DIR})
# Reference impl uses some macro patterns clang flags as pedantic.
target_compile_options(blake3 PRIVATE -Wno-unused-parameter -Wno-pedantic)

# Header-only brain algorithm library.
add_library(hanzo_brain INTERFACE)
target_include_directories(hanzo_brain INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(hanzo_brain INTERFACE blake3)

# Tests.
enable_testing()
add_executable(test_algorithms tests/test_algorithms.cpp)
target_link_libraries(test_algorithms PRIVATE hanzo_brain)
add_test(NAME algorithms COMMAND test_algorithms)

install(DIRECTORY include/hanzo DESTINATION include)
install(DIRECTORY third_party/blake3 DESTINATION include/hanzo_brain_deps
    FILES_MATCHING PATTERN "*.h")
