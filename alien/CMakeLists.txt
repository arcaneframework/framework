# Copyright 2023 IFPEN-CEA
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

Project(AlienRepo LANGUAGES C CXX VERSION 1.0)

if (NOT ALIEN_BUILD_COMPONENT)
    set (ALIEN_BUILD_COMPONENT all)
endif()

if (ALIEN_BUILD_COMPONENT STREQUAL all)
    # Compile Arcane et ses composantes en même temps
    set(BUILD_ARCANE_INTERFACE TRUE)
elseif (ALIEN_BUILD_COMPONENT STREQUAL alien_standalone)
    # Compile uniquement Alien/standalone.
    set(BUILD_ARCANE_INTERFACE FALSE)
else()
    message(FATAL_ERROR "Invalid value '${ALIEN_BUILD_COMPONENT}' for ALIEN_BUILD_COMPONENT"
            "Valid values are: { 'all', 'alien_standalone'}")
endif()

# Common stuff
if (NOT ARCFRAMEWORK_ROOT)
    set(ARCFRAMEWORK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif ()

macro(arcane_framework_set_if_unset var_name var_value var_type description)
    if (NOT DEFINED ${var_name})
        set(${var_name} ${var_value})
    endif()
    set(${var_name} ${${var_name}} CACHE ${var_type} "${description}" FORCE)
    option(${var_name} "${description}")
endmacro()

arcane_framework_set_if_unset(Arccon_ROOT ${ARCFRAMEWORK_ROOT}/arccon PATH "Arccon root path")
arcane_framework_set_if_unset(Arccore_ROOT ${ARCFRAMEWORK_ROOT}/arccore PATH "Arccore root")
arcane_framework_set_if_unset(AXLSTAR_ROOT ${ARCFRAMEWORK_ROOT}/axlstar PATH "Axlstar root")
arcane_framework_set_if_unset(ARCDEPENDENCIES_ROOT ${ARCFRAMEWORK_ROOT}/dependencies PATH "Arcdependencies root")
arcane_framework_set_if_unset(Arcane_ROOT ${ARCFRAMEWORK_ROOT}/arcane PATH "Arcane root")
arcane_framework_set_if_unset(Arctools_ROOT ${ARCFRAMEWORK_ROOT}/arctools PATH "Arctools root")


list(APPEND CMAKE_MODULE_PATH ${ARCFRAMEWORK_ROOT}/_common/build_all/cmake)


set(BLA_F95 OFF)
#find blas can be override by -DXXX_ROOT=xx or -DBLA_VENDOR=xx
#-DXXX_ROOT=xx override -DBLA_VENDOR=xx
if(MKL_ROOT)
    set(BLA_VENDOR Intel10_lp64_seq)
elseif(FLEXIBLAS_ROOT)
    set(BLA_VENDOR FlexiBLAS)
elseif(OPENBLAS_ROOT)
    set(BLA_VENDOR OpenBLAS)
endif()


find_package(BLAS REQUIRED)
message(STATUS "[ Alien ] BLAS LIBRARIES: ${BLAS_LIBRARIES}")

set(CBLAS_INCLUDE_DIR CBLAS_INCLUDE_DIR-NOTFOUND)
if(BLAS_LIBRARIES MATCHES mkl OR BLA_VENDOR MATCHES Intel)
    # relative path for mkl_cblas.h change with mkl versions
    if(MKL_ROOT)
        find_path(CBLAS_INCLUDE_DIR mkl_cblas.h PATHS MKL_ROOT PATH_SUFFIXES include mkl/include NO_DEFAULT_PATH REQUIRED)
    else()
        find_path(CBLAS_INCLUDE_DIR mkl_cblas.h PATH_SUFFIXES include mkl/include REQUIRED)
    endif()
    add_definitions(-DALIEN_USE_MKL)
    message(STATUS "[ Alien ] MKL BLAS detected")
    message(STATUS "[ Alien ] mkl_cblas.h found in ${CBLAS_INCLUDE_DIR}")
elseif(BLAS_LIBRARIES MATCHES FlexiBLAS OR BLAS_LIBRARIES MATCHES flexiblas OR BLA_VENDOR MATCHES FlexiBLAS)
    # Flexiblas include may be installed in include/flexiblas dir from root install
    if(FLEXIBLAS_ROOT)
        find_path(CBLAS_INCLUDE_DIR cblas.h PATHS FLEXIBLAS_ROOT PATH_SUFFIXES include include/flexiblas NO_DEFAULT_PATH REQUIRED)
    else()
        find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
    endif()
    add_definitions(-DALIEN_USE_FLEXIBLAS)
    message(STATUS "[ Alien ] FlexiBLAS BLAS detected")
    message(STATUS "[ Alien ] cblas.h found in ${CBLAS_INCLUDE_DIR}")
elseif(BLAS_LIBRARIES MATCHES OpenBLAS OR BLAS_LIBRARIES MATCHES openblas OR BLA_VENDOR MATCHES OpenBLAS)

    if(OPENBLAS_ROOT)
        find_path(CBLAS_INCLUDE_DIR cblas.h HINTS ENV OPENBLAS_ROOT NO_DEFAULT_PATH REQUIRED)
    else()
        find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
    endif()
    add_definitions(-DALIEN_USE_OPENBLAS)
    message(STATUS "[ Alien ] OpenBLAS BLAS detected")
    message(STATUS "[ Alien ] cblas.h found in ${CBLAS_INCLUDE_DIR}")
else()
    # supposes that other BLAS packages provide a cblas.h file
    # we must be sure that the cblas found
    find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
    message(STATUS "[ Alien ] cblas.h found in ${CBLAS_INCLUDE_DIR}")
endif()

add_subdirectory(standalone)

set(ALIEN_FOUND YES)
set(FRAMEWORK_INSTALL YES)

if (WIN32) # Not yet ready...
endif()

if(BUILD_ARCANE_INTERFACE)
    add_subdirectory(ArcaneInterface)
endif ()

