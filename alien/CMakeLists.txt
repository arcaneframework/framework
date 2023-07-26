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

# Add Alien only if blas and boost found
find_package(BLAS)
find_package(Boost COMPONENTS program_options)
if (BLAS_FOUND AND Boost_FOUND)
    add_subdirectory(standalone)
    set(ALIEN_FOUND YES)
    set(FRAMEWORK_INSTALL YES)
    add_subdirectory(ArcaneInterface)
endif()
