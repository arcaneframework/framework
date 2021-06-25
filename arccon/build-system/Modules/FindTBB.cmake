#
# Find the TBB (Intel Thread Building blocks) includes and library
#
# This module defines
# TBB_INCLUDE_DIR, where to find headers,
# TBB_LIBRARIES, the libraries to link against to use TBB.
# TBB_FOUND, If false, do not try to use TBB.

arccon_return_if_package_found(TBB)

find_library(TBB_LIBRARY_DEBUG NAMES tbb_debug)
find_library(TBB_LIBRARY_RELEASE NAMES tbb)
message(STATUS "TBB DEBUG ${TBB_LIBRARY_DEBUG}")
message(STATUS "TBB RELEASE ${TBB_LIBRARY_RELEASE}")

find_path(TBB_INCLUDE_DIR tbb/tbb_thread.h)

message(STATUS "TBB_INCLUDE_DIR = ${TBB_INCLUDE_DIR}")

if (TBB_LIBRARY_DEBUG)
  set(_TBB_HAS_DEBUG_LIB TRUE)
else()
  set(TBB_LIBRARY_DEBUG ${TBB_LIBRARY_RELEASE})
endif()

set(TBB_FOUND NO)
if (TBB_INCLUDE_DIR AND TBB_LIBRARY_RELEASE AND TBB_LIBRARY_DEBUG)
  set(TBB_FOUND YES)
  if (WIN32)
    set(TBB_LIBRARIES "$<$<CONFIG:Debug>:${TBB_LIBRARY_DEBUG}>$<$<CONFIG:Release>:${TBB_LIBRARY_RELEASE}>")
  else()
    set(TBB_LIBRARIES ${TBB_LIBRARY_RELEASE} )
  endif()
  set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Créé une interface cible pour référencer facilement le package dans les dépendances.
# Utilise TBB_LIBRARY_DEBUG si on compile en mode Debug, TBB_LIBRARY_RELEASE sinon
# En débug, il faut aussi définir TBB_USE_DEBUG=1 pour que les vérifications soient
# activées.
# TODO: il faudrait pouvoir spécifier la version Debug même en compilation
# en mode optimisé.
if (TBB_FOUND)
  arccon_register_package_library(TBB TBB)
  if (CMAKE_BUILD_TYPE STREQUAL Debug)
    if (_TBB_HAS_DEBUG_LIB)
      target_compile_definitions(arcconpkg_TBB INTERFACE TBB_USE_DEBUG=1)
    endif()
  endif()
  # Sous Win32, utilise les generator-expression pour spécifier le choix de la bibliothèque
  # en fonction de la cible 'Debug' ou 'Release'.
  # Sous Unix, on devrait faire la même chose mais cela pose problème avec le fichier
  # .pc généré donc pour l'instant on laisse comme ci dessous.
  if (NOT WIN32)
    if (CMAKE_BUILD_TYPE STREQUAL Debug)
      target_link_libraries(arcconpkg_TBB INTERFACE ${TBB_LIBRARY_DEBUG})
    else()
      target_link_libraries(arcconpkg_TBB INTERFACE ${TBB_LIBRARY_RELEASE})
    endif()
  endif()
endif ()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
