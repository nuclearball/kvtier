cmake_minimum_required(VERSION 3.20)
project(kvtier VERSION 0.1.0 LANGUAGES C CXX)

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
include(GNUInstallDirs)

add_compile_options(-Wall -Wextra)
if(SC_ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

file(GLOB_RECURSE SC_CORE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER SC_CORE_SOURCES EXCLUDE REGEX ".*/model_profile\\.cpp$")

add_library(kvtier STATIC ${SC_CORE_SOURCES})
target_include_directories(kvtier PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(kvtier PUBLIC Threads::Threads)
if(SC_CUCKOO)
  target_compile_definitions(kvtier PUBLIC KV_USE_CUCKOO_TOMB)
endif()

if(SC_BUILD_SHARED)
  add_library(kvtier_shared SHARED ${SC_CORE_SOURCES})
  target_include_directories(kvtier_shared PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(kvtier_shared PUBLIC Threads::Threads)
  set_target_properties(kvtier_shared PROPERTIES OUTPUT_NAME kvtier)
  if(SC_CUCKOO)
    target_compile_definitions(kvtier_shared PUBLIC KV_USE_CUCKOO_TOMB)
  endif()
endif()

# model_profile is an adapter-side, model-aware module: not part of the core.
add_library(kvtier_mp STATIC src/model_profile.cpp)
target_include_directories(kvtier_mp PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(kvtier_mp PUBLIC kvtier)

if(SC_BUILD_TESTS)
  enable_testing()
  file(GLOB SC_TEST_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
  foreach(testsrc ${SC_TEST_SOURCES})
    get_filename_component(testname ${testsrc} NAME_WE)
    add_executable(${testname} ${testsrc})
    target_link_libraries(${testname} PRIVATE kvtier_mp)
    add_test(NAME ${testname} COMMAND ${testname})
  endforeach()
endif()

if(SC_BUILD_EXAMPLES)
  add_executable(demo examples/demo.cpp)
  target_link_libraries(demo PRIVATE kvtier)

  add_executable(bench examples/bench.cpp src/model_profile.cpp)
  target_link_libraries(bench PRIVATE kvtier)

  add_executable(kv-profile tools/kv_profile.cpp src/model_profile.cpp)
  target_link_libraries(kv-profile PRIVATE kvtier)
endif()

# ---------------------------------------------------------------------------
# install + package config (find_package(kvtier))
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

set(SC_INSTALL_TARGETS kvtier kvtier_mp)
if(SC_BUILD_SHARED)
  list(APPEND SC_INSTALL_TARGETS kvtier_shared)
endif()

install(TARGETS ${SC_INSTALL_TARGETS}
        EXPORT kvtierTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/kvtier.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/model_profile.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/kv_config_gen.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/kv_config_defaults.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT kvtierTargets
        FILE kvtierTargets.cmake
        NAMESPACE kvtier::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvtier)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kvtierConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kvtierConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvtier)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kvtierConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/kvtierConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/kvtierConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kvtier)
