cmake_minimum_required(VERSION 3.13)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


file(READ "version" ALIEN_VERSION_STR_FULL)
string(REPLACE "_dev" "" ALIEN_VERSION ${ALIEN_VERSION_STR_FULL})
string(STRIP "${ALIEN_VERSION}" ALIEN_VERSION)  # In case of \n
message(STATUS "AlienLegacyVersion = ${ALIEN_VERSION}")

project(ALIEN 
        VERSION ${ALIEN_VERSION}) # Le nom est important car les axl sont générés dans <PROJECT_NAME>/axl

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(USE_BUILDSYSTEM_GIT_SUBMODULE)
  message(status "USE NEW BUILDSYSTEM")
  set(INFRA_BUILDSYSTEM_PATH ${PROJECT_SOURCE_DIR}/app-cmake-buildsystem)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/app-cmake-buildsystem/tools)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/app-cmake-buildsystem/alien)
else()
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/build-system)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
endif()
if(DESACTIVATE_AXLSTAR)
  set(USE_AXLSTAR FALSE)
  set(AXL_HEADERS_PATH ${CMAKE_SOURCE_DIR}/modules/axl)
else()
  set(USE_AXLSTAR TRUE)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
set(USE_ARCCON TRUE)
set(USE_PROJECT_CONFIG_HEADER TRUE)
set(REQUIRE_INSTALL_PROJECTTARGETS TRUE)
set(Arccon_USE_CMAKE_CONFIG TRUE)
set(Arccore_USE_CMAKE_CONFIG TRUE)
set(Axlstar_USE_CMAKE_CONFIG TRUE)
set(Alien_USE_CMAKE_CONFIG TRUE)
set(Arcane_USE_CMAKE_CONFIG TRUE)
#set(PETSc_USE_CMAKE_CONFIG TRUE)
#set(PETSc_USE_PKGCONFIG TRUE)
#set(Hypre_USE_CMAKE_CONFIG TRUE)
#set(MTL4_USE_CMAKE_CONFIG TRUE)
#set(SuperLU_USE_CMAKE_CONFIG TRUE)
set(GTest_USE_CMAKE_CONFIG TRUE)
set(TBB_USE_CMAKE_CONFIG TRUE)
set(MPI_USE_CMAKE_CONFIG TRUE)
set(OpenBLAS_USE_CMAKE_CONFIG TRUE)
set(Boost_USE_CMAKE_CONFIG TRUE)
set(Glib_USE_CMAKE_CONFIG TRUE)
set(HDF5_USE_CMAKE_CONFIG TRUE)

include(LoadBuildSystem)

# Default options (verbose, cxx11)
include(LoadDefaultOptions)

# root file for packages (if defined)
include(LoadDefaultPackageFile)

# default metas (win32/linux)
include(LoadDefaultMetas)

# default compilation options
include(LoadDefaultCompilationFlags)

# default packages (mono et glib)
include(LoadDefaultPackages)

# default languages  (axl)
# NB: after packages
# Incompatible avec axlstar !!
#include(LoadDefaultLanguages)

# Alien Options
include(LoadAlienOptions)

# Alien's external dependencies
include(LoadAlienPackages)

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

include(LoadAlienLegacyConfigFile)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

enable_testing()

message_separator()
logStatus("Loading components...")

add_subdirectory(modules/options)
add_subdirectory(modules/external_packages)
add_subdirectory(modules/trilinos)
add_subdirectory(modules/hpddm)
add_subdirectory(modules/ifpen_solvers)
add_subdirectory(modules/arcane_tools)
add_subdirectory(modules/interface_c)

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

include(FinishBuildSystem)

message_separator()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
