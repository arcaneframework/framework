#
# Find the 'glib' includes and library
#
# This module defines
# GLIB_INCLUDE_DIR, where to find headers,
# GLIB_LIBRARIES, the libraries to link against to use glib.
# GLIB_FOUND, If false, do not try to use glib.
# Glib_FOUND, If false, do not try to use glib.
 
# Utilise pkg_check_modules pour chercher les modules.
# pkg-config ne spécifie pas le chemin complet des bibliothèques
# donc on recherche ensuite spécifiquement chaque bibliothèque
# et fichier par un find_library() ou find_path()
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
if(PKG_GLIB_INCLUDE_DIRS)
  set(Glib_INCLUDE_DIRS ${PKG_GLIB_INCLUDE_DIRS})
else()
  find_path(Glib_INCLUDE_DIRS NAMES glib.h PATH_SUFFIXES include HINTS ${PKG_GLIB_INCLUDE_DIRS})
endif()
set(Glib_LIBRARIES ${GLIB_GLIB_LIBRARIES} ${GLIB_GTHREAD_LIBRARIES} ${GLIB_GMODULE_LIBRARIES})
set(GLIB_FOUND ${PKG_GLIB_FOUND})
message(STATUS "Glib_LIBRARIES          = ${Glib_LIBRARIES}")
message(STATUS "Glib_INCLUDE_DIRS       = ${Glib_INCLUDE_DIRS}")
if (NOT GLIB_FOUND)
  if (Glib_LIBRARIES AND Glib_INCLUDE_DIRS)
    set(GLIB_FOUND TRUE)
  endif()
endif()
message(STATUS "Glib_FOUND?             = ${Glib_FOUND}")

# Créé une interface cible pour référencer facilement le package dans les dépendances.
if(GLIB_FOUND)
  set(Glib_FOUND TRUE)
endif()
arccon_register_package_library(Glib Glib)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
