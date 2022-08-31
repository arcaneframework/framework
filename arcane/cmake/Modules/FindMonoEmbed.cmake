#
# Find the Mono embed includes and library
#
# This module defines
# MONOEMBED_INCLUDE_DIR, where to find headers,
# MONOEMBED_LIBRARIES, the libraries to link against to use mono.so.
# MONOEMBED_FOUND, 'true' if the package is found.
#
# If found, a target 'arcane::MonoEmbed' is defined

arccon_return_if_package_found(MonoEmbed)

# Si on a trouvé 'mono', utilise le chemin de l'exécutable
# pour fixer la valeur de la variable d'environnement
# pour pkg-config

if (MONO_EXEC)
  # Récupère les chemins liés à 'mono'.
  get_filename_component(MONO_EXEC_PATH ${MONO_EXEC} PATH CACHE)
  get_filename_component(MONO_ROOT_PATH ${MONO_EXEC_PATH} PATH)

  set(_PREV_CONFIG_PATH ENV{PKG_CONFIG_PATH})
  set(ENV{PKG_CONFIG_PATH} "${MONO_ROOT_PATH}/lib/pkgconfig")
endif()

# A partir de la version 2.10, il existe une version avec le GC boehm et une avec sgen.
# Il est préférable d'utiliser la version avec 'sgen' plutôt que celle de GC boehm qui
# est obsolète.
pkg_check_modules(MONOEMBEDPKG monosgen-2)

message(STATUS "MONOEMBEDPKG_INCLUDE_DIRS = ${MONOEMBEDPKG_INCLUDE_DIRS}")
message(STATUS "MONOEMBEDPKG_LIBRARIES    = ${MONOEMBEDPKG_LIBRARIES}")
message(STATUS "MONOEMBEDPKG_LDFLAGS      = ${MONOEMBEDPKG_LDFLAGS}")
message(STATUS "MONOEMBEDPKG_LIBDIR       = ${MONOEMBEDPKG_LIBDIR}")
message(STATUS "MONOEMBEDPKG_PREFIX       = ${MONOEMBEDPKG_PREFIX}")
message(STATUS "MONOEMBEDPKG_VERSION      = ${MONOEMBEDPKG_VERSION}")

# Comme pkg_check_modules ne retourne pas le chemin des libs dans *_LIBRARIES,
# on va les chercher à la main
if (MONOEMBEDPKG_FOUND)
  foreach(flag ${MONOEMBEDPKG_LDFLAGS})
    string(REGEX MATCH "^-L.*" _flagout ${flag})
    if (_flagout)
      string(LENGTH ${_flagout} _flaglen)
      message(STATUS "FOUND ARG name=${_flagout}")
      math(EXPR _sublen "${_flaglen}-2")
      string(SUBSTRING ${_flagout} 2 ${_sublen} _pathout)
      message(STATUS "LIBPATH name=${_flagout} ${_pathout}")
      set(_monolibspath ${_pathout} ${_monolibspath})
    endif (_flagout)
  endforeach(flag ${MONOEMBEDPKG_LDFLAGS})
  message(STATUS "MONOLIBSPATH =${_monolibspath}")
  foreach(lib ${MONOEMBEDPKG_LIBRARIES})
    find_library(XLIB_${lib} NAMES ${lib} PATHS ${_monolibspath})
    message(STATUS "SEARCH name=${XLIB_${lib}} ${lib}")
    set(MONOEMBED_LIBRARIES ${MONOEMBED_LIBRARIES} ${XLIB_${lib}})
  endforeach(lib ${MONOEMBEDPKG_LIBRARIES})
  set(MONOEMBED_FOUND "YES")
  set(MONOEMBED_INCLUDE_DIRS ${MONOEMBEDPKG_INCLUDE_DIRS})
  arccon_register_package_library(MonoEmbed MONOEMBED)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::MonoEmbed ALIAS arcconpkg_MonoEmbed)
endif (MONOEMBEDPKG_FOUND)

set(MONOEMBED_EXEC_PATH ${MONO_EXEC_PATH})

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
