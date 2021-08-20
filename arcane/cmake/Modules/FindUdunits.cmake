#
# Find the 'Udunits' includes and library
#
# This module defines
# Udunits_INCLUDE_DIR, where to find headers,
# Udunits_LIBRARIES, the libraries to link against to use Udunits.
# Udunits_FOUND, If false, do not try to use Udunits.
 
find_library(Udunits_LIBRARY udunits2)
find_library(Udunits_EXPAT_LIBRARY expat)
find_path(Udunits_INCLUDE_DIR udunits2.h)

message(STATUS "Udunits_INCLUDE_DIR        = ${Udunits_INCLUDE_DIR}")
message(STATUS "Udunits_LIBRARY            = ${Udunits_LIBRARY}")
message(STATUS "Udunits_EXPAT_LIBRARY      = ${Udunits_EXPAT_LIBRARY}")

set( Udunits_FOUND "NO" )
if (Udunits_INCLUDE_DIR AND Udunits_LIBRARY AND Udunits_EXPAT_LIBRARY)
  set(Udunits_FOUND "YES" )
  set(Udunits_LIBRARIES ${Udunits_LIBRARY} ${Udunits_EXPAT_LIBRARY} )
  set(Udunits_INCLUDE_DIRS ${Udunits_INCLUDE_DIR})
  arccon_register_package_library(Udunits Udunits)
endif ()
