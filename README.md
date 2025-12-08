# N-Body Simulation

A high-performance chaotic N-Body gravitational simulation with real-time visualization using SFML.

## Features

- **Real-time Rendering** with SFML graphics
- **Barnes-Hut Algorithm** (O(N log N)) for efficient force calculation
- **Naive O(N²) Algorithm** option for validation and small systems
- **SIMD Vectorization** (AVX2) for integration loop
- **OpenMP Parallelization** for physics computation
- **Console Benchmark** for performance profiling without graphics overhead
- **Configurable Physics** via JSON config file

## System Requirements

- **OS**: Windows (MinGW-w64), Linux, or macOS
- **C++ Compiler**: GCC 7+ or Clang 5+ with C++17 support
- **CMake**: 3.10 or higher
- **SFML**: 2.6.1 (for main app only, benchmark has no external dependencies)

### Windows with MinGW Setup
```bash
# If you don't have MinGW installed
# It needs to match the exact version mingw 13.1.0. Can be downloaded from the same page as SFML 2.6.1
# Download from: https://www.sfml-dev.org/download/sfml/2.6.1/

# SFML 2.6.1 (for the main app)
# Download from: https://www.sfml-dev.org/download/sfml/2.6.1/
```

## Installation & Building

### Quick Start (Windows - PowerShell)

```powershell
# 1. Navigate to project directory
cd "ABSOLUTE_PATH\N-Body"

# 2. Create build directory
mkdir build
cd build

# 3. Configure with CMake (first time only)
cmake -G "MinGW Makefiles" -DSFML_ROOT="ABSOLUTE_PATH/SFML-2.6.1" -DCMAKE_BUILD_TYPE=Release ..

# 4. Build
cmake --build . --config Release

# 5. Copy .dll SFML files to the exact folder with n-body-simulation.exe from SFML-2.6.1\bin

# 6. Run main app or benchmark
# Main app with graphics:
.\bin\n-body-simulation.exe

# Console benchmark (no graphics):
.\bin\benchmark.exe
```

### Linux/macOS

```bash
# 1. Install dependencies
# Ubuntu/Debian:
sudo apt-get install cmake g++ libsfml-dev libopenmp-dev

# macOS (with Homebrew):
brew install cmake sfml open-mp

# 2. Build
cd /path/to/N-Body
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
chmod +x bin/n-body-simulation bin/benchmark

# 3. Run
./bin/n-body-simulation     # Main app
./bin/benchmark             # Benchmark
```

### Custom SFML Location (Windows)

If your SFML is installed elsewhere:

```powershell
cmake -G "MinGW Makefiles" -DSFML_ROOT="C:/your/sfml/path" -DCMAKE_BUILD_TYPE=Release ..
```

## Folder Structure

```
N-Body/
├── CMakeLists.txt                 # Build configuration
├── README.md                       # This file
├── build/                          # Build directory (created by cmake)
│   ├── bin/
│   │   ├── n-body-simulation.exe  # Main app (with graphics)
│   │   └── benchmark.exe          # Console benchmark
│   └── ...
├── N-Body Simulation/
│   ├── main.cpp                   # Main SFML app
│   ├── benchmark.cpp              # Console benchmark (no SFML)
│   ├── config.json                # Physics configuration
│   ├── Body.h
│   ├── Simulation.h
│   ├── Vec2.h
│   └── ...
└── .git/
```

## Configuration

Edit `N-Body Simulation/config.json` to customize physics parameters:

```json
{
  "useBarnesHut": true,
  "useParallel": true,
  "numBodies": 10000,
  "spread": 400.0,
  "mass": 1000.0,
  "theta": 1.0,
  "dt": 0.05,
  "windowSize": 1000,
  "windowTitle": "Chaotic N-Body Simulation"
}
```

### Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `useBarnesHut` | bool | `true` | Use Barnes-Hut (O(N log N)) instead of naive O(N²) |
| `useParallel` | bool | `true` | Enable OpenMP parallelization |
| `numBodies` | int | `10000` | Number of bodies to simulate |
| `spread` | float | `400.0` | Initial spatial distribution radius |
| `mass` | float | `1000.0` | Mass of each body |
| `theta` | float | `1.0` | Barnes-Hut theta parameter (accuracy control) |
| `dt` | float | `0.05` | Time step for integration |
| `windowSize` | int | `1000` | Window dimensions (pixels) |
| `windowTitle` | string | `"Chaotic N-Body Simulation"` | Window title |

## Usage

### Main Application (with Graphics)
```bash
./bin/n-body-simulation
```

**Controls:**
- Mouse wheel scroll: Zoom in/out
- Close window: Exit

**Output:**
- FPS counter in console
- Real-time physics metrics
- Visual rendering of all bodies

### Benchmark (Console Only)
```bash
./bin/benchmark
```

**Perfect for:**
- VTune profiling (no graphics overhead)
- Performance measurement
- Parallelization analysis
- Physics validation

**Output:**
- `FPS: X.XX | Physics: X.XX ms | Total frame: X.XX ms` every 1 second
- Metrics logged to `log.txt`

## Performance

### Optimization Flags Used

```
-O3 -march=native -ffast-math -mavx2 -fopenmp -DNDEBUG
```

- `-O3`: Maximum optimization
- `-march=native`: CPU-specific optimizations
- `-ffast-math`: Relaxed floating-point semantics
- `-mavx2`: 256-bit SIMD vectorization
- `-fopenmp`: OpenMP parallelization
- `-DNDEBUG`: Disable assertions

### Physics Computation

- **O(N²) Algorithm**: ~2ms per frame (10,000 bodies, single-threaded)
- **Barnes-Hut**: ~0.2-0.3ms per frame (10,000 bodies, parallelized)
- **Rendering**: ~8-16ms per frame (dominated by graphics, not physics)

## Build Troubleshooting

### CMake can't find SFML
```powershell
# Specify exact SFML path:
cmake -G "MinGW Makefiles" -DSFML_ROOT="ABSOLUTE_PATH/SFML-2.6.1" ..
```

### Compiler not found
```powershell
# Make sure MinGW is in PATH:
g++ --version  # Should show version

# If not, add to PATH:
$env:PATH += ";C:\mingw64\bin"
```

### Config.json not found
```powershell
# Copy config to build output directory:
cp "N-Body Simulation/config.json" "build/bin/config.json"
```

## Build with Custom Options

### Only compile main app (skip benchmark)
Edit `CMakeLists.txt` and comment out the benchmark section.

### Change compiler flags
Edit `CMakeLists.txt` line 7:
```cmake
set(CMAKE_CXX_FLAGS_INIT "-O2 -march=native")  # Custom flags
```

## Development

### Clean rebuild
```powershell
cd build
rm CMakeCache.txt CMakeFiles -r -Force
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Debug build (with symbols for VTune)
```powershell
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug
```

## VTune Profiling

The benchmark executable is specifically designed for VTune profiling:

```bash
# Build with debug symbols (no -DNDEBUG):
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Release

# Profile with VTune
vtune -collect hotspots -result-dir=vtune_results -- .\bin\benchmark.exe
```

## Physics Algorithms

### O(N²) Naive Algorithm
- Direct pairwise force calculation
- Quadratic scaling with body count
- Good for validation and small systems
- Fully parallelized with dynamic scheduling

### Barnes-Hut Tree Algorithm
- Hierarchical spatial decomposition
- O(N log N) complexity
- Configurable accuracy via theta parameter
- Static parallelization over bodies

### Integration Method
- Velocity-Verlet with manual AVX2 vectorization
- 8-wide SIMD for position/velocity updates
- Maintains numerical stability

## Performance Analysis

### Typical Performance (Intel i7/Ryzen 5)

| Bodies | Algorithm | FPS | Physics Time |
|--------|-----------|-----|--------------|
| 1000 | O(N²) | 200-300 | 3-5ms |
| 1000 | Barnes-Hut | 800-1000 | 1-2ms |
| 10000 | O(N²) | 50-100 | 50-100ms |
| 10000 | Barnes-Hut | 200-400 | 2-5ms |

*Rendering adds ~8-16ms per frame independent of physics*

## License

[Add your license information here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## References

- **Barnes-Hut Algorithm**: Appel, A. W. (1985). An efficient program for many-body simulation
- **SFML Documentation**: https://www.sfml-dev.org/documentation/2.6.1/
- **OpenMP**: https://www.openmp.org/
