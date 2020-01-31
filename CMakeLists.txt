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
find_package(Arccore REQUIRED)
find_package(ALIEN REQUIRED)
find_package(Hypre REQUIRED)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(alien_hypre
        src/hypre_vector.cpp
        src/hypre_matrix.cpp
        src/internal/hypre_internal.cpp
        src/internal/hypre_internal_linear_algebra.cpp
        src/internal/hypre_internal_linear_solver.cpp
        src/converters/hypre_to_simplecsr_vector.cpp
        src/converters/simplecsr_to_hypre_vector.cpp
        src/converters/simplecsr_to_hypre_matrix.cpp
        src/hypre_backend.h)

target_link_libraries(alien_hypre PUBLIC
        arccore_trace
        arccore_collections
        arccore_base
        arccore_message_passing_mpi
        Alien::ALIEN
        arcconpkg_Hypre)

target_include_directories(alien_hypre PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/public
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/src
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ALIEN/hypre>)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

include(GenerateExportHeader)

generate_export_header(alien_hypre
        EXPORT_FILE_NAME ALIEN/hypre/export.h)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

enable_testing()

add_subdirectory(test)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
