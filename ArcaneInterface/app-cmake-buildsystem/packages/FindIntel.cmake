#
# Find the INTEL includes and library
#
# This module uses
# INTEL_ROOT
#
# This module defines
# INTEL_FOUND
# INTEL_LIBRARIES
#
# Target INTEL

if(NOT INTEL_ROOT)
  set(INTEL_ROOT $ENV{INTEL_ROOT})
endif()

if(INTEL_ROOT)
  set(_INTEL_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_INTEL_SEARCH_OPTS)
endif()

if(NOT WIN32)
  
  if(NOT INTEL_FOUND)
    
    find_library(INTEL_IRC_LIBRARY
      NAMES irc
      HINTS ${INTEL_ROOT} 
	    PATH_SUFFIXES lib lib/intel64
      ${_INTEL_SEARCH_OPTS}
      )
    mark_as_advanced(INTEL_IRC_LIBRARY)

  endif()
  
  # pour limiter le mode verbose
  set(INTEL_FIND_QUIETLY ON)

  find_package_handle_standard_args(INTEL 
	  DEFAULT_MSG 
	  INTEL_IRC_LIBRARY)
   
  if(INTEL_FOUND AND NOT TARGET intel)
    
    set(INTEL_LIBRARIES ${INTEL_IRC_LIBRARY})

    add_library(intel UNKNOWN IMPORTED)
    
    set_target_properties(intel PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${INTEL_IRC_LIBRARY}")
    
  endif()
  
else()
  
  if(NOT INTEL_FOUND)
    
    find_library(INTEL_IRC_LIBRARY 
	    NAMES libirc
	    HINTS ${INTEL_ROOT}
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_IRC_LIBRARY)
	  
    find_library(INTEL_SVLM_DISPMT_LIBRARY
	    NAMES svml_dispmt
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_SVLM_DISPMT_LIBRARY)
	  
	  find_library(INTEL_SVLM_DISPMD_LIBRARY
	    NAMES svml_dispmd
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_SVLM_DISPMD_LIBRARY)
	  
    find_library(INTEL_M_LIBRARY
	    NAMES libm
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_M_LIBRARY)
	  
    find_library(INTEL_MMT_LIBRARY 
	    NAMES libmmt
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_MMT_LIBRARY)
    
	  find_library(INTEL_MMD_LIBRARY 
	    NAMES libmmd
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_MMD_LIBRARY)
	  
    find_library(INTEL_DECIMAL_LIBRARY 
	    NAMES libdecimal
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
    mark_as_advanced(INTEL_DECIMAL_LIBRARY)
	  
	  find_library(INTEL_IFCONSOL_LIBRARY 
	    NAMES ifconsol
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
    mark_as_advanced(INTEL_IFCONSOL_LIBRARY)
	  
	  find_library(INTEL_IFCOREMD_LIBRARY 
	    NAMES libifcoremd
	    HINTS ${INTEL_ROOT} 
		PATH_SUFFIXES lib lib/intel64
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_IFCOREMD_LIBRARY)
	  
    find_library(INTEL_IFPORTMD_LIBRARY 
	    NAMES libifportmd
	    HINTS ${INTEL_ROOT}
		PATH_SUFFIXES lib lib/intel64		
	    ${_INTEL_SEARCH_OPTS}
	    )
	  mark_as_advanced(INTEL_IFPORTMD_LIBRARY)
    
  endif()
  
  include(FindPackageHandleStandardArgs)

  # pour limiter le mode verbose
  set(INTEL_FIND_QUIETLY ON)

  find_package_handle_standard_args(INTEL
    DEFAULT_MSG
    INTEL_M_LIBRARY 
	  INTEL_MMT_LIBRARY 
	  INTEL_MMD_LIBRARY
	  INTEL_IRC_LIBRARY
	  INTEL_SVLM_DISPMT_LIBRARY 
	  INTEL_SVLM_DISPMD_LIBRARY
	  INTEL_DECIMAL_LIBRARY 
	  INTEL_IFCONSOL_LIBRARY 
	  INTEL_IFCOREMD_LIBRARY
	  INTEL_IFPORTMD_LIBRARY
    )
  
  if(INTEL_FOUND AND NOT TARGET intel)
    
    set(INTEL_LIBRARIES ${INTEL_M_LIBRARY} 
	    ${INTEL_MMT_LIBRARY}
	    ${INTEL_MMD_LIBRARY}
	    ${INTEL_IRC_LIBRARY}
	    ${INTEL_SVLM_DISPMT_LIBRARY}
	    ${INTEL_SVLM_DISPMD_LIBRARY}
	    ${INTEL_DECIMAL_LIBRARY}
	    ${INTEL_IFCONSOL_LIBRARY}
	    ${INTEL_IFCOREMD_LIBRARY}
	    ${INTEL_IFPORTMD_LIBRARY})
    
    add_library(intel_m UNKNOWN IMPORTED)
    
    set_target_properties(intel_m PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_M_LIBRARY}")
	  
	  add_library(intel_mmt UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_mmt PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_MMT_LIBRARY}")
	  
	  add_library(intel_mmd UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_mmd PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_MMD_LIBRARY}")
	  
	  add_library(intel_irc UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_irc PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_IRC_LIBRARY}")
	  
	  add_library(intel_svlm_dispmt UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_svlm_dispmt PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_SVLM_DISPMT_LIBRARY}")
	  
    add_library(intel_svlm_dispmd UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_svlm_dispmd PROPERTIES 
 	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${INTEL_SVLM_DISPMD_LIBRARY}")
	  
	  add_library(intel_decimal UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_decimal PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_DECIMAL_LIBRARY}")
	  
	  add_library(intel_ifconsol UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_ifconsol PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_IFCONSOL_LIBRARY}")
	  
	  add_library(intel_ifcoremd UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_ifcoremd PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_IFCOREMD_LIBRARY}")
	  
	  add_library(intel_ifportmd UNKNOWN IMPORTED)
	  
	  set_target_properties(intel_ifportmd PROPERTIES 
	    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
	    IMPORTED_LOCATION "${INTEL_IFPORTMD_LIBRARY}")
    
    # intel 
	  
	  add_library(intel INTERFACE IMPORTED)
    
    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_m")

    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_mmt")

    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_mmd")

    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_irc")

    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_svlm_dispmt")

    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_svlm_dispmd")
    
	  set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_decimal")

    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_ifconsol")
  	
	  set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_ifcoremd")

    set_property(TARGET intel APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "intel_ifportmd")
    
  endif()

endif()
