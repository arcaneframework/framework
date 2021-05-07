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

cmake_minimum_required(VERSION 3.7.2)

# To properly handle VERSION directive in PROJECT
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0023 NEW)

# To export dependencies
cmake_policy(SET CMP0022 NEW)

# for find_package with Foo_ROOT
cmake_policy(SET CMP0074 NEW)

# target_link_libraries for out of directory targets.
cmake_policy(SET CMP0079 NEW)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

project(Alien
        LANGUAGES C CXX
        VERSION 1.0)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

option(ALIEN_COMPONENT_RefSemantic "Whether or not to compile RefSemantic package" ON)
option(ALIEN_COMPONENT_MoveSemantic "Whether or not to compile MoveSemantic package" ON)
option(ALIEN_UNIT_TESTS "Whether or not to enable unit tests" ON)

option(ALIEN_USE_HDF5 "Enable HDF5 export" OFF)
option(ALIEN_USE_LIBXML2 "Enable xml export" ON)

option(ALIEN_GENERATE_DOCUMENTATION "Generate API documentation" OFF)
option(ALIEN_GENERATE_TUTORIAL "Compile examples of Alien" ON)

# Make this conditional.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Use Arccon functions and macros
find_package(Arccon REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})

# use arccon commands otherwise a
include(${ARCCON_CMAKE_COMMANDS})
include(ArcconSetInstallDirs)


include(GenerateExportHeader)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# We also need 'Arccore'.

find_package(Arccore 2 REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

include(LoadAlienConfigFile)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if (NOT ALIEN_EXPORT_TARGET)
    set(ALIEN_EXPORT_TARGET ${PROJECT_NAME}Targets)
endif ()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_subdirectory(modules/core)
if (ALIEN_COMPONENT_RefSemantic)
    add_subdirectory(modules/refsemantic)
endif ()
if (ALIEN_COMPONENT_MoveSemantic)
    add_subdirectory(modules/movesemantic)
endif ()

if (ALIEN_UNIT_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

if (ALIEN_GENERATE_DOCUMENTATION)
    add_subdirectory(docs)
endif(ALIEN_GENERATE_DOCUMENTATION)

#-----------------------------------------------------------
# Tutorial
#-----------------------------------------------------------
if (ALIEN_GENERATE_TUTORIAL)
    add_subdirectory(tutorial)
endif(ALIEN_GENERATE_TUTORIAL)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# plugins
if (PLUGIN_DIR)
    message(STATUS "plugin directory is ${PLUGIN_DIR}")
    file(GLOB plugins RELATIVE ${PLUGIN_DIR} ${PLUGIN_DIR}/*)
    foreach (plugin ${plugins})
        message(STATUS "** plugin detected : ${plugin}")
        add_subdirectory(${PLUGIN_DIR}/${plugin} ${plugin})
    endforeach ()
else ()
    message(STATUS "no plugin detected")
endif ()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Use:
#   * PROJECT_VERSION
write_basic_package_version_file(
        "${version_config}" COMPATIBILITY SameMajorVersion
)

set(ALIEN_MODULE_PATH "${config_install_dir}")

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
configure_package_config_file(
        "AlienConfig.cmake.in"
        "${project_config}"
        INSTALL_DESTINATION "${config_install_dir}"
        PATH_VARS ALIEN_MODULE_PATH
)

set(Alien_project_config "${project_config}" CACHE FILEPATH "Alien project config" FORCE)
get_filename_component(Alien_config_build_dir "${project_config}" DIRECTORY CACHE)
set(Alien_config_install_dir "${config_install_dir}" CACHE STRING "Alien config install dir" FORCE)
set(Alien_VERSION "${Alien_VERSION}" CACHE STRING "Alien version" FORCE)

install(FILES "${project_config}" "${version_config}"
        DESTINATION "${config_install_dir}"
        )

install(EXPORT ${ALIEN_EXPORT_TARGET}
        NAMESPACE ${namespace}
        DESTINATION ${config_install_dir})

install(FILES cmake/LoadAlienTest.cmake
        DESTINATION ${config_install_dir})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
