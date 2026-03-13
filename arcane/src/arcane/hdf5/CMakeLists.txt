set(PUBLIC_PKGS HDF5 MPI)
set(PKGS ${PUBLIC_PKGS})

foreach(package ${PKGS})
  arcane_find_package(${package})
endforeach()

if (NOT HDF5_FOUND)
  message(STATUS "Hdf5 not found. Disabling 'arcane_hdf5'")
  return()
endif()

message(STATUS "[hdf5] HDF5_VERSION=${HDF5_VERSION}")
if (HDF5_VERSION)
  if (HDF5_VERSION VERSION_LESS "1.10")
    message(FATAL_ERROR "Your version (${HDF_VERSION}) of Hdf5 is too old. Version 1.10 or greater is required.")
    return()
  endif()
else()
  message(WARNING
    "Your version of HDF5 is unknown. Version 1.10+ is required."
    "Notwithstanding trying to use HDF5 but this may lead to errors during compilation.")
endif()

# Pour MEDFile, il faut HDF5 mais le find_package fait planter CMake (avec MED 4.0.0)
# si HDF5 n'est pas trouvé. On ne fait donc le find_package que si HDF5
# est trouvé
if (HDF5_FOUND)
  list(APPEND PRIVATE_PKGS MEDFile)
  arcane_find_package(MEDFile QUIET)
endif()

# On regarde si le mode threadsafe est activé.
# En HDF5 v2, l'option HDF5_ENABLE_THREADSAFE est dépréciée,
# HDF5_ENABLE_THREADS est supprimée et HDF5_ENABLE_CONCURRENCY
# a été ajoutée.
# https://github.com/HDFGroup/hdf5/blob/develop/release_docs/HISTORY-2.X.md#added-configuration-option-for-api-concurrency-support

set(HDF5_IS_THREADSAFE FALSE)
# L'option HDF5_IS_PARALLEL n'est pas compatible avec l'option HDF5_IS_THREADSAFE.
if (NOT HDF5_IS_PARALLEL)
  foreach (dir IN LISTS HDF5_INCLUDE_DIRS)
    if (EXISTS "${dir}/H5pubconf.h")
      file(STRINGS "${dir}/H5pubconf.h"
        HDF5_HAVE_THREADSAFE_DEFINE
        REGEX "H5_HAVE_THREADSAFE 1|H5_HAVE_CONCURRENCY 1")
      if (HDF5_HAVE_THREADSAFE_DEFINE)
        set(HDF5_IS_THREADSAFE TRUE)
      else ()
        set(HDF5_IS_THREADSAFE FALSE)
      endif ()
      break()
    endif ()
  endforeach ()
endif ()

set(HDF5_IS_THREADSAFE ${HDF5_IS_THREADSAFE} CACHE BOOL "HDF5 library compiled with threadsafe support" FORCE)

message(STATUS "[hdf5] HDF5_IS_THREADSAFE=${HDF5_IS_THREADSAFE}")

include(srcs.cmake)

if (MEDFILE_FOUND)
  list(APPEND ARCANE_SOURCES MEDMeshReaderService.cc)
endif()

# ----------------------------------------------------------------------------

arcane_add_library(arcane_hdf5
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/hdf5
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

# ----------------------------------------------------------------------------

target_link_libraries(arcane_hdf5 PUBLIC arcane_core)

arcane_add_arccon_packages(arcane_hdf5 PUBLIC ${PUBLIC_PKGS})
arcane_add_arccon_packages(arcane_hdf5 PRIVATE ${PRIVATE_PKGS})

arcane_register_library(arcane_hdf5)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
