# N-Body CMake Build Setup - FIXED

## Folder Structure

```
N-Body/
├── CMakeLists.txt              (ROOT - Main build configuration)
├── build/                      (Build directory - created by cmake)
│   ├── CMakeFiles/
│   ├── bin/
│   │   ├── sfml-app.exe        (SFML rendering app)
│   │   └── benchmark.exe       (Console-only benchmark)
│   └── ...
├── bin/                        (Original output - can be deleted)
├── N-Body Simulation/
│   ├── main.cpp                (SFML rendering app source)
│   ├── benchmark.cpp           (Console benchmark source)
│   ├── config.json             (Configuration file)
│   ├── Body.h
│   ├── Simulation.h
│   ├── Vec2.h
│   └── ...
└── .git/
```

## Build Instructions

### Initial Setup (First time)
```bash
cd "F:\ETF PROJECTS\N-Body"
mkdir build
cd build
cmake -G "MinGW Makefiles" -DSFML_ROOT="C:/SFML 2.6.1/SFML-2.6.1" -DCMAKE_BUILD_TYPE=Release ..
```

### Build
```bash
cd "F:\ETF PROJECTS\N-Body\build"
cmake --build . --config Release
```

### Run Applications
```bash
# Run SFML app
"F:\ETF PROJECTS\N-Body\build\bin\n-body-simulation.exe"

# Run benchmark
"F:\ETF PROJECTS\N-Body\build\bin\benchmark.exe"
```

## CMakeLists.txt Features

✅ **Correct Compiler Detection**
- Detects GCC/GNU compiler
- Uses proper C++17 standard
- Sets all optimization flags: `-O3 -march=native -ffast-math -mavx2 -fopenmp -DNDEBUG`

✅ **Dual Target Support**
- `sfml-app`: Full SFML rendering application with graphics
- `benchmark`: Console-only physics benchmark (no SFML dependency)

✅ **Automatic SFML Detection**
- Uses `find_package()` with SFML_ROOT hint
- Properly links graphics, window, system libraries

✅ **OpenMP Integration**
- Finds and links OpenMP for parallel physics computation
- Both targets use `-fopenmp` flag

## Troubleshooting

### If cmake fails to find SFML:
```bash
cmake -G "MinGW Makefiles" -DSFML_ROOT="C:/SFML 2.6.1/SFML-2.6.1" -DCMAKE_BUILD_TYPE=Release ..
```

### To clean and rebuild:
```bash
cd "F:\ETF PROJECTS\N-Body\build"
rm -r CMakeCache.txt CMakeFiles
cmake -G "MinGW Makefiles" -DSFML_ROOT="C:/SFML 2.6.1/SFML-2.6.1" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## What Was Fixed

| Issue | Before | After |
|-------|--------|-------|
| CMakeLists.txt location | In `N-Body Simulation/` | In root `N-Body/` |
| Project keyword syntax | `project(NBodySimulation CXX)` ❌ | `project(NBodySimulation)` ✅ |
| File paths | Relative paths ❌ | Proper quoted paths ✅ |
| Build output | Not specified | `${CMAKE_BINARY_DIR}/bin` ✅ |
| Compiler detection | Missing | Proper GNU/GCC detection ✅ |
| SFML_ROOT requirement | Hard error if missing | Sensible default provided ✅ |
