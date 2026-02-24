cmake_minimum_required(VERSION 3.13)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

option(ALIEN_LOAD_DEFAULT_COMPILER_OPTIONS "Whether or not we change default compiler flags (deprecated)" OFF)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

file(READ "version" ALIEN_VERSION_STR_FULL)
string(REPLACE "_dev" "" ALIEN_VERSION ${ALIEN_VERSION_STR_FULL})
string(STRIP "${ALIEN_VERSION}" ALIEN_VERSION)  # In case of \n
message(STATUS "AlienLegacyVersion = ${ALIEN_VERSION}")

project(AlienPlugins
        VERSION ${ALIEN_VERSION})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "app_buildsystem_tools")
set(INFRA_BUILDSYSTEM_PATH ${ARCFRAMEWORK_ROOT}/app_buildsystem_tools)

# Check if the directory 'packages' of build system is present.
# If not, this is probably because a submodule is missing.
if (NOT IS_DIRECTORY "${INFRA_BUILDSYSTEM_PATH}/packages")
  message(FATAL_ERROR "The needed directory '${INFRA_BUILDSYSTEM_PATH}/packages' is missing."
    " You may need to run the following command to get it"
    "\n   cd ${CMAKE_CURRENT_SOURCE_DIR} && git submodule update"
    )
endif()

set(ALIEN_ARCANE_INTERFACE_DIR alien/ArcaneInterface)

set(USE_AXLSTAR TRUE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
set(USE_ARCCON TRUE)
set(USE_PROJECT_CONFIG_HEADER TRUE)
set(REQUIRE_INSTALL_PROJECTTARGETS TRUE)
set(BUILDSYSTEM_INSTALL_TARGETS TRUE)
set(BUILDSYSTEM_INSTALL_EXPORT TRUE)

set(Arccon_USE_CMAKE_CONFIG TRUE)
set(Arccore_USE_CMAKE_CONFIG TRUE)
set(Axlstar_USE_CMAKE_CONFIG TRUE)
set(Arcane_USE_CMAKE_CONFIG TRUE)
#set(Alien_USE_CMAKE_CONFIG TRUE)
#set(PETSc_USE_CMAKE_CONFIG TRUE)
#set(PETSc_USE_PKGCONFIG TRUE)
set(Hypre_USE_CMAKE_CONFIG TRUE)
#set(MTL4_USE_CMAKE_CONFIG TRUE)
#set(SuperLU_USE_CMAKE_CONFIG TRUE)
set(composyx_USE_CMAKE_CONFIG TRUE)
set(GTest_USE_CMAKE_CONFIG TRUE)
set(TBB_USE_CMAKE_CONFIG TRUE)
set(MPI_USE_CMAKE_CONFIG TRUE)
set(OpenBLAS_USE_CMAKE_CONFIG TRUE)
set(Boost_USE_CMAKE_CONFIG TRUE)
#set(Glib_USE_CMAKE_CONFIG TRUE)
set(HDF5_USE_CMAKE_CONFIG TRUE)

# Path for dll copy
set(BUILDSYSTEM_DLL_COPY_DIRECTORY ${ARCANEFRAMEWORK_BINARY_DIR}/lib)

# Path to configuration files for Alien
set(ALIEN_CMAKE_CONFIG_PATH "${INFRA_BUILDSYSTEM_PATH}/alien")

find_package(Arccon REQUIRED)
include(${ARCCON_CMAKE_COMMANDS}) # needed when working with 'inline' (ie not installed) Arccon

include(${INFRA_BUILDSYSTEM_PATH}/alien/Functions.cmake)

include(${INFRA_BUILDSYSTEM_PATH}/tools/LoadBuildSystem.cmake)

# Default options (verbose, cxx11)
if (ALIEN_LOAD_DEFAULT_COMPILER_OPTIONS)
  include(${INFRA_BUILDSYSTEM_PATH}/tools/LoadDefaultOptions.cmake)
endif()

# root file for packages (if defined)
include(${INFRA_BUILDSYSTEM_PATH}/tools/LoadDefaultPackageFile.cmake)

# default metas (win32/linux)
include(${INFRA_BUILDSYSTEM_PATH}/tools/LoadDefaultMetas.cmake)

if (ALIEN_LOAD_DEFAULT_COMPILER_OPTIONS)
  # default compilation options
  include(${INFRA_BUILDSYSTEM_PATH}/tools/LoadDefaultCompilationFlags.cmake)
endif()

# default packages (mono et glib)
include(${INFRA_BUILDSYSTEM_PATH}/tools/LoadDefaultPackages.cmake)

# default languages  (axl)
# NB: after packages
# Incompatible avec axlstar !!
#include(LoadDefaultLanguages)

# Alien Options
include(${INFRA_BUILDSYSTEM_PATH}/alien/LoadAlienOptions.cmake)

# Alien's external dependencies
include(${INFRA_BUILDSYSTEM_PATH}/alien/LoadAlienPackages.cmake)

# Include Arcane cmake functions
include(${ARCANE_CMAKE_COMMANDS})

if (ENABLE_CUDA)
  # Pour le support du C++20 avec NVCC, il faut au moins cmake 3.26
  message(STATUS "CMake 3.26 is required for CUDA C++20 support in Arcane. Checking it")
  cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

  # A partir de CMake 3.18, il faut spécifier une architecture GPU pour CUDA
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50 60 70 80 90)
  endif()
  enable_language(CUDA)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Print out informations
message_separator()
logStatus("Load options from alien :")

printOptionInformations()

message_separator()
logStatus("Load languages from alien :")

printLanguageInformations()

message_separator()
logStatus("Load packages from alien :")

printPackageInformations()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message_separator()

logStatus("Load alien specifics tools :")

include(${INFRA_BUILDSYSTEM_PATH}/alien/LoadAlienLegacyConfigFile.cmake)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

enable_testing()

message_separator()
logStatus("Loading components...")

#IF(ALIEN_USE_INTELSYCL)
#  set(CMAKE_CXX_COMPILER ${ONEAPI_CXX_COMPILER})
#ENDIF()

add_subdirectory(modules/options)
add_subdirectory(modules/external_packages)
add_subdirectory(modules/trilinos)
add_subdirectory(modules/hpddm)
add_subdirectory(modules/composyx)
add_subdirectory(modules/ifpen_solvers)
add_subdirectory(modules/core_solvers)
add_subdirectory(modules/arcane_tools)
if(NOT ALIEN_USE_SYCL)
add_subdirectory(modules/interface_c)
endif()
message_separator()
logStatus("Loading tests...")

add_subdirectory(test)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# plugins
if(PLUGIN_DIR)
    logStatus("plugin directory is ${PLUGIN_DIR}")
    file(GLOB plugins RELATIVE ${PLUGIN_DIR} ${PLUGIN_DIR}/*)
    foreach(plugin ${plugins})
        logStatus("** plugin detected : ${plugin}")
        add_subdirectory(${PLUGIN_DIR}/${plugin} ${plugin})
    endforeach ()
else ()
    logStatus("no plugin detected")
endif ()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_subdirectory(modules/plugins)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

GenerateCMakeConfig()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

include(${INFRA_BUILDSYSTEM_PATH}/tools/FinishBuildSystem.cmake)

message_separator()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
