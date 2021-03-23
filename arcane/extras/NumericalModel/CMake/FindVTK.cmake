#
# Find the VTK includes and library
#
# This module defines
# VTK_INCLUDE_DIRS, where to find headers,
# VTK_LIBRARIES, the libraries to link against to use VTK.
# VTK_FOUND, If false, do not try to use VTK.

find_path(VTK_INCLUDE_DIR vtkDataReader.h
  PATHS ${VTK_INCLUDE_PATH} NO_DEFAULT_PATH)

find_library(VTK_LIBRARY vtkIO
  PATHS ${VTK_LIBRARY_PATH} NO_DEFAULT_PATH)

set(VTK_FOUND "NO")
if(VTK_INCLUDE_DIR AND VTK_LIBRARY)
  set(VTK_FOUND "YES")
  set(VTK_INCLUDE_DIRS ${VTK_INCLUDE_DIR})
  set(VTK_LIBRARIES ${VTK_LIBRARY})
endif(VTK_INCLUDE_DIR AND VTK_LIBRARY)

