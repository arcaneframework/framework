# Search la bibliothèque MEDFile de Salome
arccon_return_if_package_found(MEDFile)

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(MEDFile)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

message(STATUS "MEDFile version=${MEDFILE_VERSION}")
message(STATUS "MEDFile MEDFILE_INCLUDE_DIRS=${MEDFILE_INCLUDE_DIRS}")
message(STATUS "MEDFile MEDFILE_C_LIBRARIES=${MEDFILE_C_LIBRARIES}")
message(STATUS "MEDFile found? ${MEDFile_FOUND} ${MEDFILE_FOUND}")

set(MEDFile_FOUND False)
if(MEDFILE_INCLUDE_DIRS AND MEDFILE_C_LIBRARIES)
  set(MEDFile_FOUND TRUE)
  set(MEDFile_INCLUDE_DIRS "${MEDFILE_INCLUDE_DIRS}")
  set(MEDFile_LIBRARIES "${MEDFILE_C_LIBRARIES}")
  arccon_register_package_library(MEDFile MEDFile)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::MEDFile ALIAS arcconpkg_MEDFile)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
