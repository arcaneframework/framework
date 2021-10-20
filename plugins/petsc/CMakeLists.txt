# Copyright 2020 IFPEN-CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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

project(AlienPETSc
        VERSION 1.0)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


option(ALIEN_PETSC_UNIT_TESTS "Enable Hypre wrapper tests" ${ALIEN_UNIT_TESTS})

set(BUILD_SHARED_LIBS True)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})

find_package(Arccore REQUIRED)

find_package(Alien REQUIRED)
find_package(MPI REQUIRED)

find_package(PETSc REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(alien_petsc
        include/alien/petsc/backend.h
        src/matrix.h
        src/matrix.cpp
        src/vector.h
        src/vector.cpp
        src/petsc_linear_algebra.cpp
        src/petsc_linear_solver.cpp
        src/converters/simplecsr_to_petsc_matrix.cpp
        src/converters/simplecsr_to_petsc_vector.cpp
        src/converters/petsc_to_simplecsr_vector.cpp
        src/petsc_instance.cpp src/petsc_instance.h)

target_link_libraries(alien_petsc PUBLIC
        Alien::alien_core
        )

target_link_libraries(alien_petsc PRIVATE
        MPI::MPI_CXX
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi
        arccon::PETSc
        )

target_include_directories(alien_petsc PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

add_library(Alien::alien_petsc ALIAS alien_petsc)

include(GenerateExportHeader)

generate_export_header(alien_petsc
        EXPORT_FILE_NAME include/alien/petsc/export.h
        )

# Tests
if (ALIEN_PETSC_UNIT_TESTS)
    enable_testing()
    add_subdirectory(test)
endif (ALIEN_PETSC_UNIT_TESTS)

# Install

include(ArcconSetInstallDirs)

set(ALIENPETSC_EXPORT_TARGET ${PROJECT_NAME}Targets)

install(DIRECTORY include/alien/petsc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
        )

install(TARGETS alien_petsc EXPORT ${ALIENPETSC_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/include/alien/petsc/export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/petsc
)

set(alien_install_lib "${CMAKE_INSTALL_LIBDIR}/alien")
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(EXPORT ${ALIENPETSC_EXPORT_TARGET}
        NAMESPACE Alien::
        DESTINATION ${config_install_dir})

include(CMakePackageConfigHelpers)

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")

configure_package_config_file(
        "${PROJECT_NAME}Config.cmake.in"
        "${project_config}"
        INSTALL_DESTINATION "${config_install_dir}"
)

install(FILES "${project_config}"
        DESTINATION "${config_install_dir}"
        )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
