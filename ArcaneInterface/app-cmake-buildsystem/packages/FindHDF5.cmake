#
# Find the HDF5 includes and library
#
# This module uses
# HDF5_ROOT
#
# This module defines
# HDF5_FOUND
# HDF5_INCLUDE_DIRS
# HDF5_LIBRARIES
#
# Target hdf5
if(HDF5_FOUND)
  return()
endif()

if(NOT HDF5_ROOT)
  set(HDF5_ROOT $ENV{HDF5_ROOT})
endif()

if(NOT ZLIB_ROOT)
  set(ZLIB_ROOT $ENV{ZLIB_ROOT})
endif()

if(HDF5_ROOT)
  set(_HDF5_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_HDF5_SEARCH_OPTS)
endif()

if(ZLIB_ROOT)
  set(_ZLIB_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_ZLIB_SEARCH_OPTS)
endif()

if(NOT HDF5_FOUND) 

  find_library(HDF5_LIBRARY 
    NAMES hdf5dll hdf5ddll hdf5
    HINTS ${HDF5_ROOT}
    PATH_SUFFIXES lib
    ${_HDF5_SEARCH_OPTS}
    )
  mark_as_advanced(HDF5_LIBRARY)
  
if(WIN32)
  find_library(SZIP_LIBRARY 
    NAMES szip
    HINTS ${HDF5_ROOT}
    PATH_SUFFIXES lib
    ${_HDF5_SEARCH_OPTS}
    )
  mark_as_advanced(SZIP_LIBRARY)
endif()

if(WIN32)
  find_library(Z_LIBRARY 
    NAMES zlib
    HINTS ${HDF5_ROOT}
    PATH_SUFFIXES lib
    ${_HDF5_SEARCH_OPTS}
    )
else()
  find_library(Z_LIBRARY 
    HINTS ${ZLIB_ROOT} 
    NAMES z
    PATH_SUFFIXES lib lib64
    ${_ZLIB_SEARCH_OPTS}
    )
endif()
  mark_as_advanced(Z_LIBRARY)
  
  find_path(HDF5_INCLUDE_DIR hdf5.h
    HINTS ${HDF5_ROOT} 
    PATH_SUFFIXES include
    ${_HDF5_SEARCH_OPTS}
    )
  mark_as_advanced(HDF5_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(HDF5_FIND_QUIETLY ON)

if(WIN32)
  find_package_handle_standard_args(HDF5 
    HDF5_INCLUDE_DIR 
    HDF5_LIBRARY
    SZIP_LIBRARY
    Z_LIBRARY
    )
else()
  find_package_handle_standard_args(HDF5 
    HDF5_INCLUDE_DIR 
    HDF5_LIBRARY
    Z_LIBRARY
    )
endif()

if(HDF5_FOUND AND NOT TARGET hdf5)
  
  set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
   
  if(WIN32)
    set(HDF5_LIBRARIES ${HDF5_LIBRARY}
                       ${SZIP_LIBRARY}
                       ${Z_LIBRARY})
  else()  
    set(HDF5_LIBRARIES ${HDF5_LIBRARY}
                       ${Z_LIBRARY})
  endif()
  
  # hdf5 main
	  
  add_library(hdf5_main UNKNOWN IMPORTED)
	 
  set_target_properties(hdf5_main PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}")
    
  set_target_properties(hdf5_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${HDF5_LIBRARY}")
  
  if(WIN32)	
    # szip
    
    add_library(szip UNKNOWN IMPORTED)
	
    set_target_properties(szip PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${SZIP_LIBRARY}")
  endif()

  # zlib
    
  add_library(zlib UNKNOWN IMPORTED)
	
  set_target_properties(zlib PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${Z_LIBRARY}")
	
  # hdf5
    
  add_library(hdf5 INTERFACE IMPORTED)
	  
  set_property(TARGET hdf5 APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "hdf5_main")

  if(WIN32)
    set_property(TARGET hdf5 APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "szip")	
  endif()

  set_property(TARGET hdf5 APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "zlib")
  
endif()
