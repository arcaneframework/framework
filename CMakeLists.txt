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

set(BUILD_SHARED_LIBS True)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})

find_package(Arccore REQUIRED)

if(NOT TARGET Alien::alien_core)
  find_package(Alien REQUIRED)
endif()
find_package(PETSc REQUIRED)
find_package(Boost REQUIRED) # For boost::timer.

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(alien_petsc
  include/alien/petsc/backend.h
  src/matrix.h
  src/matrix.cpp
  src/vector.h
  src/vector.cpp
  )

target_link_libraries(alien_petsc PUBLIC
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi
        Alien::alien_core
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

#enable_testing()

#add_subdirectory(test)

# Install

include(ArcconSetInstallDirs)

set(ALIEN_EXPORT_TARGET ${PROJECT_NAME}Targets)

install(DIRECTORY include/alien/petsc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
        )

install(TARGETS alien_petsc EXPORT ${ALIEN_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/include/alien/petsc/export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/petsc
)

set(alien_install_lib "${CMAKE_INSTALL_LIBDIR}/alien")
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(EXPORT ${ALIEN_EXPORT_TARGET}
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
