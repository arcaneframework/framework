#
# Find the ONNX includes and library
#
# This module uses
# ONNX_ROOT
#
# This module defines
# ONNX_FOUND
# ONNX_INCLUDE_DIRS
# ONNX_LIBRARIES
#
# Target onnx

if(NOT ONNX_ROOT)
  set(ONNX_ROOT $ENV{ONNX_ROOT})
endif()

if(ONNX_ROOT)
  set(_ONNX_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_ONNX_SEARCH_OPTS)
endif()

if(NOT ONNX_FOUND) 

  find_library(ONNX_LIBRARY 
    NAMES  onnxruntime
		HINTS ${ONNX_ROOT}
		PATH_SUFFIXES lib64
		${_ONNX_SEARCH_OPTS}
    )
  mark_as_advanced(ONNX_LIBRARY)
  
  find_path(ONNX_INCLUDE_DIR /onnxruntime/core/session/onnxruntime_cxx_api.h
    HINTS ${ONNX_ROOT}
		PATH_SUFFIXES include
    ${_ONNX_SEARCH_OPTS}
    )
  mark_as_advanced(ONNX_INCLUDE_DIR)
  
endif()

# pour limiter le mode verbose
set(ONNX_FIND_QUIETLY ON)

find_package_handle_standard_args(ONNX 
	DEFAULT_MSG 
	ONNX_INCLUDE_DIR 
	ONNX_LIBRARY)

if(ONNX_FOUND AND NOT TARGET onnx)
  
  set(ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIR})
  
  set(ONNX_LIBRARIES ${ONNX_LIBRARY})
  
  # onnx
	  
  add_library(onnx UNKNOWN IMPORTED)
	 
  set_target_properties(onnx PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${ONNX_INCLUDE_DIRS}")
    
	set_target_properties(onnx PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${ONNX_LIBRARY}")
    
endif()
