cmake_minimum_required(VERSION 3.13)

# To properly handle VERSION directive in PROJECT
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0023 NEW)

# To export dependencies
cmake_policy(SET CMP0022 NEW)

# for find_package with Foo_ROOT
cmake_policy(SET CMP0074 NEW)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

project(ALIEN_HYPRE
        LANGUAGES C CXX
        VERSION 1.0)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(BUILD_SHARED_LIBS True)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})

find_package(Arccore REQUIRED)
find_package(ALIEN REQUIRED)
find_package(Hypre REQUIRED)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_subdirectory(src)

enable_testing()

add_subdirectory(test)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
