#
# Find the LibXml2 includes and library
#
include(${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)

arccon_return_if_package_found(LibXml2)

# Supprime temporairement CMAKE_MODULE_PATH pour éviter une récursion
# infinie.

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(LibXml2)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

# Normalement cette cible est toujours définie depuis la version 3.12
# de CMake si LibXml2 est trouvé
if (NOT ARCCON_USE_LEGACY_FIND)
  if (TARGET LibXml2::LibXml2)
    arccon_register_cmake_config_target(LibXml2 CONFIG_TARGET_NAME LibXml2::LibXml2)
    return()
  endif()
endif()

# Ancienne version si on ne trouve pas la cible LibXml2::LibXml2
set(LibXml2_FOUND FALSE)
if (LIBXML2_FOUND)
  set(LibXml2_FOUND TRUE)
  set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})
endif()

arccon_register_package_library(LibXml2 LIBXML2)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
