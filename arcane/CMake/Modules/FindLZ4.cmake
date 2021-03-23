#
# Find the 'lz4' includes and library
#
# This module defines
# LZ4_INCLUDE_DIR, where to find headers,
# LZ4_LIBRARIES, the libraries to link against to use lz4.
# LZ4_FOUND, If false, do not try to use lz4.

if (TARGET arccon::LZ4)
  return()
endif()

# Il n'y a pas de find_package correspondant à 'LZ4' dans 'CMake'
find_library(LZ4_LIBRARY lz4)
find_path(LZ4_INCLUDE_DIR lz4.h)

message(STATUS "LZ4_INCLUDE_DIR = ${LZ4_INCLUDE_DIR}")
message(STATUS "LZ4_LIBRARY     = ${LZ4_LIBRARY}")

set(LZ4_FOUND FALSE)
if(LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
  set(LZ4_FOUND TRUE)
  set(LZ4_LIBRARIES ${LZ4_LIBRARY} )
  set(LZ4_INCLUDE_DIRS ${LZ4_INCLUDE_DIR})
endif()

arccon_register_package_library(LZ4 LZ4)
