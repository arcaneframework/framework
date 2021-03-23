# Recherche d'abord avec le fichier de configuration CMake fourni par 'sloop'
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(Sloop CONFIG QUIET)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})
message(STATUS "Sloop_FOUND=? ${Sloop_FOUND}")

if (Sloop_FOUND)
  message(STATUS "Found Sloop via CMake config file")
  # Créé une cible interface 'arccon::sloop' qui sera
  # celle référencée par les autres packages. Cela permet d'être indépendant
  # du nom du package sloop.
  set(SLOOP_FOUND TRUE)
  add_library(arcanepkg_sloop INTERFACE)
  target_link_libraries(arcanepkg_sloop INTERFACE sloop)
  add_library(arccon::sloop ALIAS arcanepkg_sloop)
  set(SLOOP_FOUND TRUE CACHE BOOL "Is Package 'SLOOP' Found" FORCE)
  return()
endif()

message(STATUS "Infos for searching Sloop (manual configuration):")
message(STATUS "ARCANE_PACKAGE_SLOOP_LIB_NAME = ${ARCANE_PACKAGE_SLOOP_LIB_NAME}")
message(STATUS "ARCANE_PACKAGE_SLOOP_INCLUDE_PATH = ${ARCANE_PACKAGE_SLOOP_INCLUDE_PATH}")
message(STATUS "ARCANE_PACKAGE_SLOOP_LIBRARY_PATH = ${ARCANE_PACKAGE_SLOOP_LIBRARY_PATH}")

if (ARCANE_PACKAGE_SLOOP_LIB_NAME)
  find_library(SLOOP_LIBRARIES NAMES ${ARCANE_PACKAGE_SLOOP_LIB_NAME} PATHS ${ARCANE_PACKAGE_SLOOP_LIBRARY_PATH})
endif ()

if (ARCANE_PACKAGE_SLOOP_INCLUDE_PATH)
  find_path(SLOOP_INCLUDE_DIRS NAMES SLOOP.h PATHS ${ARCANE_PACKAGE_SLOOP_INCLUDE_PATH})
endif ()

message(STATUS "SLOOP_LIBRARIES = ${SLOOP_LIBRARIES}")
message(STATUS "SLOOP_INCLUDE_DIRS = ${SLOOP_INCLUDE_DIRS}")

set( SLOOP_FOUND "NO" )
if(SLOOP_INCLUDE_DIRS AND SLOOP_LIBRARIES)
  set( SLOOP_FOUND "YES" )
endif()

# Sloop a besoin de SuperLU (sauf sur ARM) donc si ce dernier n'est pas trouvé, désactive sloop
if (SLOOP_FOUND)
  if (NOT IS_TERA1K_ARM)
    if (NOT SUPERLU_FOUND)
      set(SLOOP_FOUND NO)
      message(STATUS "Warning: remove sloop packages because SuperLU is not available")
    endif ()
  endif ()
endif ()

if (SLOOP_FOUND)
  arcane_add_package_library(sloop SLOOP)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
