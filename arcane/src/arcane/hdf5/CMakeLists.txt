set(PUBLIC_PKGS HDF5 MPI)
set(PKGS ${PUBLIC_PKGS})

foreach(package ${PKGS})
  arcane_find_package(${package})
endforeach()
if (NOT HDF5_FOUND)
  message(STATUS "Hdf5 not found. Disabling 'arcane_hdf5'")
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_hdf5
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/hdf5
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

# ----------------------------------------------------------------------------

target_link_libraries(arcane_hdf5 PUBLIC ${ARCANE_BASE_LIBRARIES})

arcane_add_arccon_packages(arcane_hdf5 PUBLIC ${PUBLIC_PKGS})

arcane_register_library(arcane_hdf5)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
