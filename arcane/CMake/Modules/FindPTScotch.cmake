#
# Find the Ptscotch includes and library
#
# This module defines
# PTScotch_INCLUDE_DIR, where to find headers,
# PTScotch_LIBRARIES, the libraries to link against to use Ptscotch.
# PTScotch_FOUND, If false, do not try to use Ptscotch.
 
find_path(PTScotch_INCLUDE_DIR ptscotch.h)
 
FIND_LIBRARY(PTScotch_LIBRARY ptscotch)
FIND_LIBRARY(PTScotchERR_LIBRARY ptscotcherrexit)
FIND_LIBRARY(SCOTCH_LIBRARY scotch)

SET(PTScotch_EXTRA_LIBS)
IF(WIN32)
  # Not portable patch due to WIN32 version compiled with Intel compiler
  FIND_LIBRARY(LIBMMD_LIBRARY libmmd)
  FIND_LIBRARY(LIBIRC_LIBRARY libirc)
  FIND_LIBRARY(SVML_LIBRARY svml_dispmd)
  FIND_LIBRARY(DECIMAL_LIBRARY libdecimal)
  SET(PTScotch_EXTRA_LIBS ${LIBMMD_LIBRARY} ${LIBIRC_LIBRARY} ${SVML_LIBRARY} ${DECIMAL_LIBRARY})
ENDIF()

MESSAGE(STATUS "PTScotch_INCLUDE_DIR = ${PTScotch_INCLUDE_DIR}")
MESSAGE(STATUS "PTScotch_LIBRARY = ${PTScotch_LIBRARY}")
MESSAGE(STATUS "PTScotchERR_LIBRARY = ${PTScotchERR_LIBRARY}")
MESSAGE(STATUS "SCOTCH_LIBRARY = ${SCOTCH_LIBRARY}")
MESSAGE(STATUS "PTScotch_EXTRA_LIBS = ${PTScotch_EXTRA_LIBS}")

SET( PTScotch_FOUND "NO" )
if(PTScotch_INCLUDE_DIR AND PTScotch_LIBRARY)
  SET( PTScotch_FOUND "YES" )
  SET( PTScotch_LIBRARIES ${PTScotch_LIBRARY} ${PTScotchERR_LIBRARY})
  SET( PTScotch_INCLUDE_DIRS ${PTScotch_INCLUDE_DIR})
  # Pour PTScotch version 6 besoin d'une lib supplémentaire
  if(SCOTCH_LIBRARY)
    list(APPEND PTScotch_LIBRARIES ${SCOTCH_LIBRARY})
  endif(SCOTCH_LIBRARY)	
  if(PTScotch_EXTRA_LIBS)
    list(APPEND PTScotch_LIBRARIES ${PTScotch_EXTRA_LIBS})
  endif(PTScotch_EXTRA_LIBS)
  arccon_register_package_library(PTScotch PTScotch)
endif()
