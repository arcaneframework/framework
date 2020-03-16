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

project(ALIEN_HYPRE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(BUILD_SHARED_LIBS True)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})

find_package(Arccore REQUIRED)

# mode plugin ? ALIEN should be defined by parent project
# (and could not be found because not yet installed)
# otherwise we need to find ALIEN
if(NOT TARGET alien_core)
  find_package(Alien REQUIRED)
endif()

find_package(Hypre REQUIRED)
find_package(MPI REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_subdirectory(src)

enable_testing()

add_subdirectory(test)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
