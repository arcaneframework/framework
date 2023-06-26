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

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0079 NEW)

# MUST be done before call to 'project'
get_cmake_property(vars CACHE_VARIABLES)
foreach (var ${vars})
    get_property(currentHelpString CACHE "${var}" PROPERTY HELPSTRING)
    if ("${currentHelpString}" MATCHES "No help, variable specified on the command line." OR "${currentHelpString}" STREQUAL "")
        # message("${var} = [${${var}}]  --  ${currentHelpString}") # uncomment to see the variables being processed
        list(APPEND CL_ARGS "-D${var}=${${var}}")
    endif ()
endforeach ()

file(READ "version" ALIEN_VERSION_STR_FULL)
string(REPLACE "_dev" "" ALIEN_VERSION ${ALIEN_VERSION_STR_FULL})
string(STRIP "${ALIEN_VERSION}" ALIEN_VERSION)  # In case of \n
message(STATUS "AlienVersion = ${ALIEN_VERSION}")

# ----------------------------------------------------------------------------
# Indique que le projet n'a pas de language
# Cela est indispensable pour que CMake ne recherche pas les
# compilateurs avant d'avoir inclus les fichiers de configuration.
Project(Alien
        LANGUAGES C CXX
        VERSION ${ALIEN_VERSION})

option(ALIEN_COMPONENT_RefSemantic "Whether or not to compile RefSemantic package" ON)
option(ALIEN_COMPONENT_MoveSemantic "Whether or not to compile MoveSemantic package" ON)
option(ALIEN_EXAMPLES "Whether or not to compile examples" ON)
option(ALIEN_BENCHMARKS "Benchmark driver for Alien" ON)
option(ALIEN_UNIT_TESTS "Whether or not to enable unit tests" OFF)
option(ALIEN_EXPAND_TESTS "Run unit tests one by one" OFF)

option(ALIEN_USE_HDF5 "Enable HDF5 export" OFF)
option(ALIEN_USE_LIBXML2 "Enable xml export" OFF)
option(ALIEN_USE_SYCL "Whether or not to compile SYCL backend" OFF)
option(ALIEN_USE_PERF_TIMER "Whether or not to enable perf timer" OFF)
option(ALIEN_WANT_AVX "Whether or not to enable avx flags" OFF)
option(ALIEN_WANT_AVX2 "Whether or not to enable avx2 flags" OFF)
option(ALIEN_WANT_AVX512 "Whether or not to enable avx512 flags" OFF)
option(ALIEN_DEFAULT_OPTIONS "Enable best effort to find optional dependencies" ON)

option(ALIEN_GENERATE_DOCUMENTATION "Generate API documentation" OFF)
option(ALIEN_GENERATE_TUTORIAL "Compile examples of Alien" ON)

option(ALIEN_PLUGIN_HYPRE "Whether or not to compile Hypre backend" OFF)
option(ALIEN_PLUGIN_PETSC "Whether or not to compile PETSc backend" OFF)
option(ALIEN_PLUGIN_TRILINOS "Whether or not to compile Trilinos backend" OFF)

option(ALIEN_PLUGIN_GINKGO "Whether or not to compile Ginkgo backend" OFF)
option(ALIEN_PLUGIN_GINKGO_REF "Whether or not to compile Ginkgo REFERENCE backend" OFF)
option(ALIEN_PLUGIN_GINKGO_OMP "Whether or not to compile Ginkgo OPENMP backend" OFF)
option(ALIEN_PLUGIN_GINKGO_CUDA "Whether or not to compile Ginkgo CUDA backend" OFF)
option(ALIEN_PLUGIN_GINKGO_HIP "Whether or not to compile Ginkgo HIP backend" OFF)
option(ALIEN_PLUGIN_GINKGO_DPCPP "Whether or not to compile DPCPP Ginkgo backend" OFF)

option(ALIENDEV_EMBEDDED "[dev] forbid to compile Arcon and Arccore" OFF)

# Make this conditional.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# For shared libs, needed for Arccore
set(BUILD_SHARED_LIBS ON)

include(GNUInstallDirs)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(ALIEN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

include(sanitizers)

# First we try to use external arccon and arccore
if (ALIENDEV_EMBEDDED)
    if (POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)  # DOWNLOAD_EXTRACT_TIMESTAMP instead of archive timestamp
    endif ()
    include(FetchContent)

    FetchContent_Declare(
            Arccon
            URL https://github.com/arcaneframework/framework/releases/download/arccon-v1.5.0/arccon-1.5.0.src.tar.gz
    )

    FetchContent_Declare(
            Arccore
            URL https://github.com/arcaneframework/framework/releases/download/arccore-v2.2.0.0/arccore-2.2.0.0.src.tar.gz
    )
    set(FRAMEWORK_EXPORT_NAME AlienFrameworkTargets)

    get_filename_component(_fp_framework "CMake/fake-framework" REALPATH)
    if (EXISTS ${_fp_framework})
        list(APPEND CMAKE_MODULE_PATH ${_fp_framework})
    endif ()

    FetchContent_MakeAvailable(Arccon Arccore)
endif ()
find_package(Arccon REQUIRED)
# Only use find package with Arccon
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})

find_package(Arccore REQUIRED)

find_package(LibArchive)

if (LibArchive_FOUND)
    add_definitions(-DALIEN_USE_LIBARCHIVE)
endif ()

include(GNUInstallDirs)

if (ALIEN_DEFAULT_OPTIONS)
    find_package(HDF5 QUIET)
    set(ALIEN_USE_HDF5 ${HDF5_FOUND})

    find_package(LibXml2 QUIET)
    set(ALIEN_USE_LIBXML2 ${LibXml2_FOUND})

    find_package(GTest QUIET)
    set(ALIEN_UNIT_TESTS ${GTest_FOUND})
endif ()

enable_testing()

if (ALIEN_GENERATE_DOCUMENTATION)
    find_package(Doxygen REQUIRED)

    # doxygen settings can be set here, prefixed with "DOXYGEN_"
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

    # For sphinx
    set(DOXYGEN_GENERATE_HTML NO)
    set(DOXYGEN_GENERATE_XML YES)
endif (ALIEN_GENERATE_DOCUMENTATION)


# ----------------------------------------------------------------------------
# Configure installation layout.
# Layout. This works for all platforms:
#   * <prefix>/lib*/cmake/<PROJECT-NAME>
#   * <prefix>/lib*/
#   * <prefix>/include/
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(namespace "Alien::")

if (ALIEN_USE_PERF_TIMER)
    add_definitions(-DALIEN_USE_PERF_TIMER)
endif ()

add_subdirectory(src)

if (ALIEN_BENCHMARKS)
    add_subdirectory(benchmark)
endif ()

if (ALIEN_PLUGIN_HYPRE)
    add_subdirectory(plugins/hypre)
endif (ALIEN_PLUGIN_HYPRE)

if (ALIEN_PLUGIN_PETSC)
    add_subdirectory(plugins/petsc)
endif (ALIEN_PLUGIN_PETSC)

if (ALIEN_PLUGIN_TRILINOS)
    add_subdirectory(plugins/trilinos)
endif (ALIEN_PLUGIN_TRILINOS)

if (ALIEN_PLUGIN_GINKGO)
    add_subdirectory(plugins/ginkgo)
endif (ALIEN_PLUGIN_GINKGO)

if (ALIEN_EXAMPLES)
    add_subdirectory(examples)
endif ()

add_subdirectory(docs)
# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
