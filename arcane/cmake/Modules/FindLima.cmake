#
# Find the Lima include and libraries
#
# This module defines
# LIMA_INCLUDE_DIRS, where to find headers
# LIMA_LIBRARIES, the libraries to link against to use LM tools
# LIMA_FOUND, If false, do not try to use LM tools.
 
arccon_return_if_package_found(Lima)

find_package(HDF5 COMPONENTS C CXX)

set(Lima_FOUND FALSE)

# Recherche d'abord avec le fichier de configuration CMake fourni par Lima
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(Lima CONFIG QUIET)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

message(STATUS "Lima version=${Lima_VERSION}")
message(STATUS "Lima Lima_INCLUDE_DIRS=${Lima_INCLUDE_DIRS}")
message(STATUS "Lima Lima_LIBRARIES=${Lima_LIBRARIES}")
message(STATUS "Lima found? ${Lima_FOUND}")

set(ARCANE_LIMA_HAS_MLI FALSE)
set(ARCANE_LIMA_HAS_MLI2 FALSE)

# Certains CMake de Arcane utilisent encore lima avec le nom tout en majuscule.
# On faire donc la correspondance si besoin
if (TARGET Lima::Lima)
  message(STATUS "Using target Lima::Lima")
  set(Lima_FOUND TRUE CACHE BOOL "Is Lima found" FORCE)

  # Il n'y a pas de moyens direct actuellement (novembre 2020) pour savoir
  # si Lima a été compilé avec le support des fichiers MLI et/ou MLI2.
  # Le seule moyen accessible est de récupérer les options de compilation et
  # de regarder si elle contient des '-D' pour '__INTERNE_MALIPP' et '__INTERNE_MALIPP2'
  get_target_property(_ARCANE_LIMA_COMPILE_DEFINITIONS Lima::Lima INTERFACE_COMPILE_DEFINITIONS)
  message(STATUS "Lima CompileDefinition=${_ARCANE_LIMA_COMPILE_DEFINITIONS}")
  if ("__INTERNE_MALIPP" IN_LIST _ARCANE_LIMA_COMPILE_DEFINITIONS)
    set(ARCANE_LIMA_HAS_MLI TRUE)
    message(STATUS "Lima has 'mli' support")
  endif()
  if ("__INTERNE_MALIPP2" IN_LIST _ARCANE_LIMA_COMPILE_DEFINITIONS)
    set(ARCANE_LIMA_HAS_MLI2 TRUE)
    message(STATUS "Lima has 'mli2' support")
  endif()
  arccon_register_cmake_config_target(Liam CONFIG_TARGET_NAME Lima::Lima PACKAGE_NAME Lima)
endif()

set(ARCANE_LIMA_HAS_MLI ${ARCANE_LIMA_HAS_MLI} CACHE BOOL "true if Lima has 'mli' support" FORCE)
set(ARCANE_LIMA_HAS_MLI2 ${ARCANE_LIMA_HAS_MLI2} CACHE BOOL "true if Lima has 'mli2' support" FORCE)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
