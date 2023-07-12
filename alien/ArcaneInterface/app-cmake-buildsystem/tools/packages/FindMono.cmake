#
# Find the MONO includes and library
#
# This module defines
# MONO_INCLUDE_DIR, where to find headers,
# MONO_LIBRARIES, the libraries to link against to use MONO.
# MONO_FOUND, If false, do not try to use MONO
 
find_program(MONO_EXEC mono)
find_program(MONO_MKBUNDLE mkbundle)

set(Mkbundle_EXEC)
if (MONO_MKBUNDLE)
   set(Mkbundle_EXEC ${CMAKE_CURRENT_BINARY_DIR}/mkbundle.exe)
endif (MONO_MKBUNDLE)

include(FindPackageHandleStandardArgs)

# pour limiter le mode verbose
set(MONO_FIND_QUIETLY ON)

find_package_handle_standard_args(MONO
  DEFAULT_MSG 
  MONO_EXEC 
  Mkbundle_EXEC
  )

get_filename_component(MONO_EXEC_PATH ${MONO_EXEC} PATH)

find_file(MONO_PKG_CONFIG_PATH
	  NAMES pkgconfig
	  HINTS ${MONO_EXEC_PATH}/../lib
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/mkbundle.exe.in mkbundle.exe @ONLY)

get_filename_component(MONOembed_ROOT_PATH ${MONO_EXEC} PATH)
 
mark_as_advanced(MONOembed_ROOT_PATH)
