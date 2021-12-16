#
# Find the 'glib' includes and library
#
# This module defines
# Glib_INCLUDE_DIRS, where to find headers,
# Glib_LIBRARIES, the libraries to link against to use glib.
# Glib_FOUND, If false, do not try to use glib.

include(${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)
arccon_return_if_package_found(Glib)

find_package(PkgConfig)
pkg_check_modules(PKG_GLIB glib-2.0 gthread-2.0 gmodule-2.0)

message(STATUS "Infos from pkg_check_modules")
message(STATUS "PKG_GLIB_INCLUDE_DIRS       = ${PKG_GLIB_INCLUDE_DIRS}")
message(STATUS "PKG_GLIB_LIBRARIES          = ${PKG_GLIB_LIBRARIES}")
message(STATUS "PKG_GLIB_LIBRARY_DIRS       = ${PKG_GLIB_LIBRARY_DIRS}")

find_library(GLIB_GLIB_LIBRARIES NAMES glib-2.0 HINTS ${PKG_GLIB_LIBDIR} ${PKG_GLIB_LIBRARY_DIRS})
find_library(GLIB_GTHREAD_LIBRARIES NAMES gthread-2.0 HINTS ${PKG_GLIB_LIBDIR} ${PKG_GLIB_LIBRARY_DIRS})
find_library(GLIB_GMODULE_LIBRARIES NAMES gmodule-2.0 HINTS ${PKG_GLIB_LIBDIR} ${PKG_GLIB_LIBRARY_DIRS})

# Si on ne trouve pas via 'pkg-config' (par exemple parce que ce dernier n'est pas disponible),
# alors il faut ajouter directement le répertoire contenant la bibliothèque 'glib-2.0' car
# le fichier 'glibconfig.h' se trouve en général dans ce répertoire.
if (GLIB_GLIB_LIBRARIES)
  get_filename_component(_GLIB_LIB_DIR ${GLIB_GLIB_LIBRARIES} DIRECTORY)
endif()

message(STATUS "GLIB_GLIB_LIBRARIES     = ${GLIB_GLIB_LIBRARIES}")
message(STATUS "GLIB_GTHREAD_LIBRARIES  = ${GLIB_GTHREAD_LIBRARIES}")
message(STATUS "GLIB_GMODULE_LIBRARIES  = ${GLIB_GMODULE_LIBRARIES}")
message(STATUS "_GLIB_LIB_DIR           = ${_GLIB_LIB_DIR}")

# On a besoin de 'glib.h' et 'glibconfig.h' qui peuvent être dans des répertoires différents
find_path(GLIBCONFIG_INCLUDE_DIR
  NAMES glibconfig.h
  HINTS ${PKG_GLIB_LIBDIR} ${PKG_GLIB_LIBRARY_DIRS} ${PKG_GLIB_INCLUDE_DIRS} ${_GLIB_LIB_DIR}
  PATH_SUFFIXES glib-2.0/include
)

find_path(GLIB_INCLUDE_DIR
  NAMES glib.h
  HINTS ${PKG_GLIB_INCLUDEDIR} ${PKG_GLIB_INCLUDE_DIRS}
  PATH_SUFFIXES glib-2.0
)

set(Glib_INCLUDE_DIRS ${GLIB_INCLUDE_DIR} ${GLIBCONFIG_INCLUDE_DIR})

set(Glib_LIBRARIES ${GLIB_GLIB_LIBRARIES} ${GLIB_GTHREAD_LIBRARIES} ${GLIB_GMODULE_LIBRARIES})

message(STATUS "Glib_LIBRARIES          = ${Glib_LIBRARIES}")
message(STATUS "Glib_INCLUDE_DIRS       = ${Glib_INCLUDE_DIRS}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Glib REQUIRED_VARS Glib_INCLUDE_DIRS Glib_LIBRARIES)
if (Glib_FOUND)
  arccon_register_package_library(Glib Glib)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
