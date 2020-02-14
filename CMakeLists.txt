cmake_minimum_required(VERSION 3.13)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

project(ALIEN) # Le nom est important car les axl sont générés dans <PROJECT_NAME>/axl

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/build-system)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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

include(LoadAlienConfigFile)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

enable_testing()

message_separator()
logStatus("Loading components...")

add_subdirectory(modules/AlienOptions)
add_subdirectory(modules/AlienExternalPackages)
add_subdirectory(modules/AlienTrilinos)
add_subdirectory(modules/AlienIFPENSolvers)
add_subdirectory(modules/AlienArcaneTools)

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
    endforeach()
else()
    logStatus("no plugin detected")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

GenerateCMakeConfig()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

include(FinishBuildSystem)

message_separator()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
