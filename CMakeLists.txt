cmake_minimum_required(VERSION 3.20)
project(hanzo-bot-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# BLAKE3 from the canonical Lux C++ crypto stack. Pull straight from
# luxcpp/blake3-reference — same source luxcpp/crypto uses, no vendored
# copy. One way to do it.
include(FetchContent)
FetchContent_Declare(
    luxcpp_blake3
    GIT_REPOSITORY https://github.com/luxcpp/blake3-reference.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(luxcpp_blake3)

# blake3-reference provides a `blake3` CMake target via its own
# CMakeLists.txt (under /c/). If for some reason the upstream target name
# changes, this is the single point of failure.
if(NOT TARGET blake3)
    # Fallback: build the portable + NEON sources directly.
    set(BLAKE3_DIR ${luxcpp_blake3_SOURCE_DIR}/c)
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
    target_compile_options(blake3 PRIVATE -Wno-unused-parameter -Wno-pedantic)
endif()

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
