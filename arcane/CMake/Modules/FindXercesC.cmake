#
# Find the XercesC includes and library
#
# This module defines
# XercesC_INCLUDE_DIR, where to find headers,
# XercesC_LIBRARIES, the libraries to link against to use XercesC.
# XercesC_FOUND, If false, do not try to use XercesC.

# Les version qui terminent par 'D' sont les versions Windows en mode Debug.
FIND_LIBRARY(XercesC_LIBRARY NAMES xerces-c xerces-c_3 xerces-c_3D)

FIND_PATH(XercesC_INCLUDE_DIR xercesc/dom/DOM.hpp)

MESSAGE(STATUS "XercesC_LIBRARY = ${XercesC_LIBRARY}")
MESSAGE(STATUS "XercesC_INCLUDE_DIR = ${XercesC_INCLUDE_DIR}")

SET( XercesC_FOUND "NO" )
if(XercesC_INCLUDE_DIR AND XercesC_LIBRARY)
  set( XercesC_FOUND "YES" )
  set( XercesC_LIBRARIES ${XercesC_LIBRARY} )
  set( XercesC_INCLUDE_DIRS ${XercesC_INCLUDE_DIR} )
  get_filename_component(XercesC_LIB_PATH ${XercesC_LIBRARY} PATH)
  arccon_register_package_library(XercesC XercesC)
endif()
