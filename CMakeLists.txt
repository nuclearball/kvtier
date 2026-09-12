cmake_minimum_required(VERSION 3.20)
project(solidcacher_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SC_BUILD_TESTS "Build tests" ON)
option(SC_BUILD_EXAMPLES "Build examples and tools" ON)
option(SC_BUILD_SHARED "Build shared library too" ON)
option(SC_ENABLE_ASAN "Build with ASan/UBSan" OFF)
option(SC_CUCKOO "Enable cuckoo-filter tombstones" OFF)

find_package(Threads REQUIRED)

add_compile_options(-Wall -Wextra)
if(SC_ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

file(GLOB_RECURSE SC_CORE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER SC_CORE_SOURCES EXCLUDE REGEX ".*/model_profile\\.cpp$")

add_library(solidcacher STATIC ${SC_CORE_SOURCES})
target_include_directories(solidcacher PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(solidcacher PUBLIC Threads::Threads)
if(SC_CUCKOO)
  target_compile_definitions(solidcacher PUBLIC KV_USE_CUCKOO_TOMB)
endif()

if(SC_BUILD_SHARED)
  add_library(solidcacher_shared SHARED ${SC_CORE_SOURCES})
  target_include_directories(solidcacher_shared PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(solidcacher_shared PUBLIC Threads::Threads)
  set_target_properties(solidcacher_shared PROPERTIES OUTPUT_NAME solidcacher)
  if(SC_CUCKOO)
    target_compile_definitions(solidcacher_shared PUBLIC KV_USE_CUCKOO_TOMB)
  endif()
endif()

# model_profile is an adapter-side, model-aware module: not part of the core.
add_library(solidcacher_mp STATIC src/model_profile.cpp)
target_include_directories(solidcacher_mp PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(solidcacher_mp PUBLIC solidcacher)

if(SC_BUILD_TESTS)
  enable_testing()
  file(GLOB SC_TEST_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
  foreach(testsrc ${SC_TEST_SOURCES})
    get_filename_component(testname ${testsrc} NAME_WE)
    add_executable(${testname} ${testsrc})
    target_link_libraries(${testname} PRIVATE solidcacher_mp)
    add_test(NAME ${testname} COMMAND ${testname})
  endforeach()
endif()

if(SC_BUILD_EXAMPLES)
  add_executable(demo examples/demo.cpp)
  target_link_libraries(demo PRIVATE solidcacher)

  add_executable(bench examples/bench.cpp src/model_profile.cpp)
  target_link_libraries(bench PRIVATE solidcacher)

  add_executable(kv-profile tools/kv_profile.cpp src/model_profile.cpp)
  target_link_libraries(kv-profile PRIVATE solidcacher)
endif()
