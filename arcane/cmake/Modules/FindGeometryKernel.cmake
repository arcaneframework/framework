#
# Find 'GeometryKernel' headers and libraries.
#
# This module defines
# GEOMETRYKERNEL_INCLUDE_DIR, where to find headers,
# GEOMETRYKERNEL_LIBRARIES
# GEOMETRYKERNEL_FOUND, If false, do not try to use it

if (TARGET arcconpkg_GeometryKernel)
  return()
endif()

# lib extension depends on build type
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(GK_SUFFIX D)
else()
  set(SUFFIX )
endif()

set(GEOMETRYKERNEL_ROOT ${Arcane_SOURCE_DIR}/extras/BasicGeometryKernel)
set(GEOMETRYKERNEL_ROOT_LIB ${GEOMETRYKERNEL_ROOT}/src/GeometryKernel)
message(STATUS "GeometryKernel: GEOMETRYKERNEL_ROOT = ${GEOMETRYKERNEL_ROOT}")
list(APPEND CMAKE_PREFIX_PATH ${GEOMETRYKERNEL_ROOT})

find_path(GEOMETRYKERNEL_INCLUDE_DIR GeometryKernel/base/plane.h)
find_library(GEOMETRYKERNEL_LIBRARY NAMES GeometryKernel${GK_SUFFIX} GeometryKernel PATHS ${GEOMETRYKERNEL_ROOT_LIB})
find_library(GEOMETRYKERNEL_LIBRARY2 lmba${GK_SUFFIX})
find_library(GEOMETRYKERNEL_LIBRARY3 ttl${GK_SUFFIX})

message(STATUS "GeometryKernel: GEOMETRYKERNEL_INCLUDE_DIR = ${GEOMETRYKERNEL_INCLUDE_DIR}")
message(STATUS "GeometryKernel: GEOMETRYKERNEL_LIBRARY     = ${GEOMETRYKERNEL_LIBRARY}")
message(STATUS "GeometryKernel: GEOMETRYKERNEL_LIBRARY2    = ${GEOMETRYKERNEL_LIBRARY2}")
message(STATUS "GeometryKernel: GEOMETRYKERNEL_LIBRARY3    = ${GEOMETRYKERNEL_LIBRARY3}")

set(GEOMETRYKERNEL_FOUND FALSE)
set(GeometryKernel_FOUND FALSE)
if(GEOMETRYKERNEL_INCLUDE_DIR AND GEOMETRYKERNEL_LIBRARY) # AND GEOMETRYKERNEL_LIBRARY2 AND GEOMETRYKERNEL_LIBRARY3)
  set(GEOMETRYKERNEL_FOUND TRUE)
  set(GeometryKernel_FOUND TRUE)
  message(STATUS "GeometryKernel FOUND")
  set(GEOMETRYKERNEL_LIBRARIES ${GEOMETRYKERNEL_LIBRARY}) # ${GEOMETRYKERNEL_LIBRARY2} ${GEOMETRYKERNEL_LIBRARY3})
  get_filename_component(GEOMETRYKERNEL_LIB_PATH ${GEOMETRYKERNEL_LIBRARY} PATH)
  set(GEOMETRYKERNEL_INCLUDE_DIRS ${GEOMETRYKERNEL_INCLUDE_DIR})
  # NOTE: Ce package ne doit pas être enregistré via 'arccon' car il n'est pas vraiment
  # externe puisqu'il référence des sources de Arcane. Il ne doit donc pas être exporté.
  #arccon_register_package_library(GeometryKernel GEOMETRYKERNEL)
endif()
