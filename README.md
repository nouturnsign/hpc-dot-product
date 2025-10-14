# HPC Dot Product OpenMP Benchmark

A small demo of dot product implementations.

## Build

```sh
mkdir -p build
cd build
# the following depends on where you installed OpenMP:
# - on Mac, make sure to export OpenMP_ROOT=$(brew --prefix)/opt/libomp
cmake -DENABLE_OPENMP=ON ..
# if you did not select the correct path for OpenMP_ROOT, CMake will warn
cmake --build .
./dot_bench
```
