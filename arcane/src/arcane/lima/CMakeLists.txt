include(srcs.cmake)

arcane_find_package(MPI)
arcane_find_package(HDF5)
arcane_find_package(Lima)

if (NOT Lima_FOUND)
  message(STATUS "Lima not found. Disabling 'arcane_lima'")
  return()
endif()

arcane_add_library(arcane_lima
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/lima
  FILES ${ARCANE_SOURCES}
)

target_link_libraries(arcane_lima PUBLIC arcane_core)

if (TARGET Lima::Lima)
  message(STATUS "Add Lima::Lima to 'arcane_lima' target")
  target_link_libraries(arcane_lima PRIVATE Lima::Lima)
elseif (TARGET arcane::lima)
  target_link_libraries(arcane_lima PRIVATE arcane::lima)
  if (ARCANE_LIMA_HAS_MLI2)
    arcane_add_arccon_packages(arcane_lima PRIVATE HDF5)
  endif ()
endif ()

if (ARCANE_LIMA_HAS_MLI2)
  target_compile_definitions(arcane_lima PRIVATE ARCANE_LIMA_HAS_MLI2)
endif ()
if (ARCANE_LIMA_HAS_MLI)
  target_compile_definitions(arcane_lima PRIVATE ARCANE_LIMA_HAS_MLI)
endif ()

# Si HDF5 est compilé avec MPI et que HDF5 n'est pas compilé avec CMake,
# il faut ajouter explicitement le support de MPI.
arcane_add_arccon_packages(arcane_lima PRIVATE MPI)

arcane_register_library(arcane_lima)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
