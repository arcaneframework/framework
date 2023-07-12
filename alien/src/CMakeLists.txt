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

include(GenerateExportHeader)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# We also need 'Arccore'.

find_package(Arccore 2 REQUIRED)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if (NOT ALIEN_EXPORT_TARGET)
    set(ALIEN_EXPORT_TARGET ${PROJECT_NAME}Targets)
endif ()

configure_file(
        AlienConfig.h.in
        ${PROJECT_BINARY_DIR}/alien/AlienConfig.h
)

install(
        FILES ${PROJECT_BINARY_DIR}/alien/AlienConfig.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
)

add_library(alien_headers INTERFACE)
target_include_directories(alien_headers INTERFACE
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/core>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_subdirectory(core)

if (ALIEN_COMPONENT_RefSemantic)
    add_subdirectory(refsemantic)
endif ()

if (ALIEN_COMPONENT_MoveSemantic)
    add_subdirectory(movesemantic)
endif ()

if (ALIEN_UNIT_TESTS)
    enable_testing()
    add_subdirectory(test_framework)
endif ()


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

install(FILES ../CMake/LoadAlienTest.cmake
        DESTINATION ${config_install_dir})


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
