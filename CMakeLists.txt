cmake_minimum_required(VERSION 3.10)
project(NBodySimulation)

# --- General Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_INIT "-O3 -march=native -ffast-math -mavx2 -fopenmp")

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- SFML Configuration ---
# Set SFML_ROOT if not already set
if(NOT SFML_ROOT)
    set(SFML_ROOT "C:/SFML 2.6.1/SFML-2.6.1")
endif()

message(STATUS "SFML_ROOT: ${SFML_ROOT}")

# Find SFML
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED HINTS ${SFML_ROOT})

# --- Main SFML Application ---
add_executable(n-body-simulation "main.cpp")

target_compile_options(n-body-simulation PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -ffast-math -mavx2 -fopenmp -DNDEBUG>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -ffast-math -mavx2 -fopenmp -DNDEBUG>
)

# Find and link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(n-body-simulation PRIVATE OpenMP::OpenMP_CXX)

# Link SFML
target_link_libraries(n-body-simulation PRIVATE 
    sfml-graphics 
    sfml-window 
    sfml-system
)

target_include_directories(n-body-simulation PRIVATE 
    "${SFML_ROOT}/include"
)

# --- Benchmark Application (Console-only, no SFML) ---
add_executable(benchmark "benchmark.cpp")

target_compile_options(benchmark PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -ffast-math -mavx2 -fopenmp -DNDEBUG>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -ffast-math -mavx2 -fopenmp -DNDEBUG>
)

find_package(OpenMP REQUIRED)
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX pthread)

target_include_directories(benchmark PRIVATE 
    . 
)

# --- Print Build Summary ---
message(STATUS "========== Build Configuration ==========")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SFML Root: ${SFML_ROOT}")
message(STATUS "Output Directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "=========================================")
