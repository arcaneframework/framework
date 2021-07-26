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

cmake_minimum_required(VERSION 3.11)

# MUST be done before call to 'project'
get_cmake_property(vars CACHE_VARIABLES)
foreach (var ${vars})
    get_property(currentHelpString CACHE "${var}" PROPERTY HELPSTRING)
    if ("${currentHelpString}" MATCHES "No help, variable specified on the command line." OR "${currentHelpString}" STREQUAL "")
        # message("${var} = [${${var}}]  --  ${currentHelpString}") # uncomment to see the variables being processed
        list(APPEND CL_ARGS "-D${var}=${${var}}")
    endif ()
endforeach ()

# ----------------------------------------------------------------------------
# Indique que le projet n'a pas de language
# Cela est indispensable pour que CMake ne recherche pas les
# compilateurs avant d'avoir inclus les fichiers de configuration.
Project(Alien
        LANGUAGES C CXX
        VERSION 1.0.3)

option(ALIEN_COMPONENT_RefSemantic "Whether or not to compile RefSemantic package" ON)
option(ALIEN_COMPONENT_MoveSemantic "Whether or not to compile MoveSemantic package" ON)
option(ALIEN_UNIT_TESTS "Whether or not to enable unit tests" ON)

option(ALIEN_USE_HDF5 "Enable HDF5 export" OFF)
option(ALIEN_USE_LIBXML2 "Enable xml export" ON)

option(ALIEN_GENERATE_DOCUMENTATION "Generate API documentation" OFF)
option(ALIEN_GENERATE_TUTORIAL "Compile examples of Alien" ON)

option(ALIEN_PLUGIN_HYPRE "Whether or not to compile Hypre backend" OFF)
option(ALIEN_PLUGIN_PETSC "Whether or not to compile PETSc backend" OFF)
option(ALIEN_PLUGIN_SUPERLU "Whether or not to compile SuperLU backend" OFF)
option(ALIEN_PLUGIN_TRILINOS "Whether or not to compile Trilinos backend" OFF)

# Make this conditional.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# For shared libs, needed for Arccore
set(BUILD_SHARED_LIBS ON)

# ----------------------------------------------------------------------------
# Macro pour positionner la variable 'var_name' dans le cache
# avec sa valeur actuelle ou la valeur 'var_value' si elle n'a
# pas de valeur.
macro(alienframework_set_if_unset var_name var_value var_type description)
    if (NOT DEFINED ${var_name})
        set(${var_name} ${var_value})
    endif ()
    set(${var_name} ${${var_name}} CACHE BOOL "${description}" FORCE)
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

alienframework_set_if_unset(ARCCON_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/framework/arccon PATH "Arccon root path")
alienframework_set_if_unset(ARCCORE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/framework/arccore PATH "Arccore root")

set(ALIEN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

set(FRAMEWORK_EXPORT_NAME AlienFrameworkTargets)

add_subdirectory(docs)

add_subdirectory(src)

if (ALIEN_PLUGIN_HYPRE)
    add_subdirectory(plugins/hypre)
endif (ALIEN_PLUGIN_HYPRE)
if (ALIEN_PLUGIN_PETSC)
    add_subdirectory(plugins/petsc)
endif (ALIEN_PLUGIN_PETSC)
if (ALIEN_PLUGIN_SUPERLU)
    add_subdirectory(plugins/superlu)
endif (ALIEN_PLUGIN_SUPERLU)
if (ALIEN_PLUGIN_TRILINOS)
    add_subdirectory(plugins/trilinos)
endif (ALIEN_PLUGIN_TRILINOS)


# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
