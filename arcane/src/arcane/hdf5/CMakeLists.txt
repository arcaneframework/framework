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

include(srcs.cmake)

arcane_add_library(arcane_hdf5
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/hdf5
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

# ----------------------------------------------------------------------------

target_link_libraries(arcane_hdf5 PUBLIC arcane_core)

arcane_add_arccon_packages(arcane_hdf5 PUBLIC ${PUBLIC_PKGS})

arcane_register_library(arcane_hdf5)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
