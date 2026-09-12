cmake_minimum_required(VERSION 3.20)
project(llama-kvcache LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

# Options
option(LLAMAKVCACHE_BUILD_EXAMPLES "Build examples" ON)
option(LLAMAKVCACHE_SHARED         "Build as shared library" OFF)

# Root of the kvtier core.  Defaults to the sibling repository root
# (this file lives in <repo>/adapters/llama).
set(KVTier_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE PATH "Root of the kvtier tree")

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# llama.cpp (llama.h / libllama / ggml.h)
# ---------------------------------------------------------------------------
find_path(LLAMA_INCLUDE_DIR llama.h
    HINTS /usr/local/include /opt/homebrew/include
)
find_library(LLAMA_LIBRARY llama
    HINTS /usr/local/lib /opt/homebrew/lib
)

if(NOT LLAMA_INCLUDE_DIR)
    message(FATAL_ERROR "llama.h not found. Install llama.cpp or set -DLLAMA_INCLUDE_DIR=")
endif()
if(NOT LLAMA_LIBRARY)
    message(FATAL_ERROR "libllama not found. Install llama.cpp or set -DLLAMA_LIBRARY=")
endif()

# ggml.h usually ships next to llama.h
find_path(GGML_INCLUDE_DIR ggml.h
    HINTS ${LLAMA_INCLUDE_DIR} /usr/local/include /opt/homebrew/include
)

# ---------------------------------------------------------------------------
# kvtier core: prefer the sibling kvtier tree
# ---------------------------------------------------------------------------
find_path(KVTier_INCLUDE_DIR kvtier.h
    HINTS ${KVTier_ROOT}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../kvtier/include
          ${CMAKE_CURRENT_SOURCE_DIR}/../../kvtier/include
          /usr/local/include
          /opt/homebrew/include
)

if(NOT KVTier_INCLUDE_DIR)
    message(FATAL_ERROR
        "kvtier.h not found. Build the core, or set "
        "-DKVTier_ROOT=/path/to/kvtier (or -DKVTier_INCLUDE_DIR=)")
endif()

# If the core is pulled in via add_subdirectory, use its CMake target.
# Otherwise locate the prebuilt library (build/libkvtier.* -> kvcache).
if(TARGET kvtier)
    set(KVTier_LINK kvtier)
else()
    find_library(KVTier_LIBRARY
        NAMES kvtier kvcache
        HINTS ${KVTier_ROOT}/build
              ${CMAKE_CURRENT_SOURCE_DIR}/../kvtier/build
              ${CMAKE_CURRENT_SOURCE_DIR}/../kvtier
              /usr/local/lib
              /opt/homebrew/lib
    )
    if(NOT KVTier_LIBRARY)
        message(FATAL_ERROR
            "libkvtier not found under ${KVTier_ROOT}/build. "
            "Build the core first or set -DKVTier_ROOT= / -DKVTier_LIBRARY=")
    endif()
    set(KVTier_LINK ${KVTier_LIBRARY})
endif()

# ---------------------------------------------------------------------------
# llama-kvcache library (static or shared)
# ---------------------------------------------------------------------------
set(LIB_SOURCES src/llama-kvcache.cpp)

if(LLAMAKVCACHE_SHARED)
    add_library(llama-kvcache SHARED ${LIB_SOURCES})
else()
    add_library(llama-kvcache STATIC ${LIB_SOURCES})
endif()

target_include_directories(llama-kvcache
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${LLAMA_INCLUDE_DIR}
        ${GGML_INCLUDE_DIR}
        ${KVTier_INCLUDE_DIR}
)

target_link_libraries(llama-kvcache
    PRIVATE
        ${LLAMA_LIBRARY}
        ${KVTier_LINK}
)

# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------
if(LLAMAKVCACHE_BUILD_EXAMPLES)
    add_executable(example_save_restore examples/example_save_restore.cpp)
    target_include_directories(example_save_restore
        PRIVATE
            ${LLAMA_INCLUDE_DIR}
            ${GGML_INCLUDE_DIR}
            ${KVTier_INCLUDE_DIR}
    )
    target_link_libraries(example_save_restore
        PRIVATE
            llama-kvcache
            ${LLAMA_LIBRARY}
    )

    add_executable(example_noise_mixed examples/example_noise_mixed.cpp)
    target_include_directories(example_noise_mixed
        PRIVATE
            ${LLAMA_INCLUDE_DIR}
            ${GGML_INCLUDE_DIR}
            ${KVTier_INCLUDE_DIR}
    )
    target_link_libraries(example_noise_mixed
        PRIVATE
            llama-kvcache
            ${LLAMA_LIBRARY}
            Threads::Threads
    )
endif()

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
install(TARGETS llama-kvcache
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/llama-kvcache.h DESTINATION include)
