# Standalone example: build against an INSTALLED epblas-parallel.
#
#   cmake -S example -B build-example \
#         -DCMAKE_PREFIX_PATH=/path/to/install
#   cmake --build build-example
#   ./build-example/scale
#
# It is deliberately NOT wired into the top-level build (it consumes the
# installed package, not the in-tree targets).
cmake_minimum_required(VERSION 3.25)
project(epblas-parallel-example LANGUAGES C)

find_package(epblas-parallel REQUIRED)

add_executable(scale scale.c)
target_link_libraries(scale PRIVATE epblas-parallel::eyblas)
