# Search la bibliothèque MEDFile de Salome

if(TARGET arcconpkg_MEDFile)
  return()
endif()

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})                                                                                                                              
unset(CMAKE_MODULE_PATH)                                                                                                                                                        
find_package(MEDFile)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})                                                                                                                              

message(STATUS "MEDFile version=${MEDFILE_VERSION}")
message(STATUS "MEDFile MEDFILE_INCLUDE_DIRS=${MEDFILE_INCLUDE_DIRS}")
message(STATUS "MEDFile MEDFILE_C_LIBRARIES=${MEDFILE_C_LIBRARIES}")
message(STATUS "MEDFile found? ${MEDFile_FOUND} ${MEDFILE_FOUND}")

if(MEDFILE_INCLUDE_DIRS AND MEDFILE_C_LIBRARIES)
  set(MEDFILE_FOUND TRUE)
  set(MEDFile_INCLUDE_DIRS "${MEDFILE_INCLUDE_DIRS}")
  set(MEDFile_LIBRARIES "${MEDFILE_C_LIBRARIES}")
  arcane_add_package_library(MEDFile MEDFile)
endif()
