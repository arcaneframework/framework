#
# Find the Lima include and libraries
#
# This module defines
# LIMA_INCLUDE_DIRS, where to find headers
# LIMA_LIBRARIES, the libraries to link against to use LM tools
# LIMA_FOUND, If false, do not try to use LM tools.
 
if(TARGET arcconpkg_Lima)
  return()
endif()

# Recherche d'abord avec le fichier de configuration CMake fourni par Lima
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})                                                                                                                              
unset(CMAKE_MODULE_PATH)                                                                                                                                                        
find_package(HDF5 COMPONENTS CXX)
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
  set(LIMA_FOUND TRUE CACHE BOOL "Is Lima found" FORCE)
  add_library(arcanepkg_lima INTERFACE)
  target_link_libraries(arcanepkg_lima INTERFACE Lima::Lima)

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

else()

  # On arrive ici si on n'utilise pas le fichier de configuration de Lima.
  if (NOT LIMA_LIB_NAME)
    return()
  endif()
  # Lima a besoin de LM.
  if (NOT TARGET arcane::lm)
    message(STATUS "Disabling Lima because package 'LM' is not found")
    return()
  endif()

  find_library(LIMA_LIBRARY ${LIMA_LIB_NAME} PATHS ${LIMA_LIB_ROOT})
  find_path(LIMA_INCLUDE_DIR Lima/lima++.h PATHS ${LIMA_INCLUDE_ROOT})
  find_path(MACHINE_TYPES_INCLUDE_DIR machine_types.h)

  message(STATUS "LIMA_LIBRARY = ${LIMA_LIBRARY}")
  message(STATUS "LIMA_INCLUDE_DIR = ${LIMA_INCLUDE_DIR}")
  message(STATUS "MACHINE_TYPES_INCLUDE_DIR = ${MACHINE_TYPES_INCLUDE_DIR}")

  set(LIMA_FOUND "NO")
  if (LIMA_LIBRARY AND LIMA_INCLUDE_DIR AND MACHINE_TYPES_INCLUDE_DIR)
    message(STATUS "Found legacy Lima installation")
    # Les versions 'historiques' ont toujours le support du format MLI
    set(ARCANE_LIMA_HAS_MLI TRUE)
    set(Lima_FOUND TRUE CACHE BOOL "Is Lima found" FORCE)
    set(LIMA_FOUND TRUE CACHE BOOL "Is Lima found" FORCE)
    set(LIMA_LIBRARIES ${LIMA_LIBRARY} ${LIMAHDF_LIB_ROOT}/libhdf145_cpp.so ${LIMAHDF_LIB_ROOT}/libhdf145.so)
    set(LIMA_INCLUDE_DIRS ${LIMA_INCLUDE_DIR} ${LIMAHDF_INCLUDE_ROOT} ${MACHINE_TYPES_INCLUDE_DIR})
    arcane_add_package_library(lima LIMA)
    target_link_libraries(arcanepkg_lima INTERFACE arcane::lm)
    add_library(arcanepkg_Lima ALIAS arcanepkg_lima)
  endif()
endif()

set(ARCANE_LIMA_HAS_MLI ${ARCANE_LIMA_HAS_MLI} CACHE BOOL "true if Lima has 'mli' support" FORCE)
set(ARCANE_LIMA_HAS_MLI2 ${ARCANE_LIMA_HAS_MLI2} CACHE BOOL "true if Lima has 'mli2' support" FORCE)
