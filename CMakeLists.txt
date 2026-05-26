# epblas — hand-written extended-precision BLAS overlays.
#
# Depends on `fortran-migrator` being installed; finds its per-target
# packages on CMAKE_PREFIX_PATH. Builds overlays + composites for
# every target whose migrator install is discoverable. See CONTEXT.md.

cmake_minimum_required(VERSION 3.25)

project(epblas
    VERSION 0.1.0
    LANGUAGES C CXX)

# Discovery and subdir wiring land in step 2.
