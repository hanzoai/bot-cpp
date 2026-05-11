cmake_minimum_required(VERSION 3.16)
project(hanzo-bot-cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Header-only library.
add_library(hanzo_brain INTERFACE)
target_include_directories(hanzo_brain INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Tests.
enable_testing()
add_executable(test_algorithms tests/test_algorithms.cpp)
target_link_libraries(test_algorithms PRIVATE hanzo_brain)
add_test(NAME algorithms COMMAND test_algorithms)

install(DIRECTORY include/hanzo DESTINATION include)
