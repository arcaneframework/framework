#
# Find the LIBICONV includes and library
#
# This module defines
# LIBICONV_INCLUDE_DIR, where to find headers,
# LIBICONV_LIBRARIES, the libraries to link against to use LIBICONV.
# LIBICONV_FOUND, If false, do not try to use LIBICONV.

# Note: pour CMake, il faut faire un find_package(LibXml2)
# mais les variables associées sont préfixées par LIBXML2
# au lieu de 'LibXml2'.

arccon_return_if_package_found(LibXml2)

# Supprime temporairement CMAKE_MODULE_PATH pour éviter une récursion
# infinie.

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(LibXml2)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

set(LibXml2_FOUND FALSE)
if (LIBXML2_FOUND)
  set(LibXml2_FOUND TRUE)
  set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})
endif()

arccon_register_package_library(LibXml2 LIBXML2)
