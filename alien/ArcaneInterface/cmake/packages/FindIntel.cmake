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
# Target intel

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
  
  find_library(Intel_IRC_LIBRARY libirc)
  find_library(Intel_SVLM_DISPMT_LIBRARY svml_dispmt)
  find_library(Intel_SVLM_DISPMD_LIBRARY svml_dispmd)
  find_library(Intel_M_LIBRARY libm)
  find_library(Intel_MMT_LIBRARY libmmt)
  find_library(Intel_MMD_LIBRARY libmmd)
  find_library(Intel_DECIMAL_LIBRARY libdecimal)
  find_library(Intel_IFCONSOL_LIBRARY ifconsol)
  find_library(Intel_IFCOREMD_LIBRARY libifcoremd)
  find_library(Intel_IFPORTMD_LIBRARY libifportmd)
  
  include(FindPackageHandleStandardArgs)

  # pour limiter le mode verbose
  set(Intel_FIND_QUIETLY ON)

  find_package_handle_standard_args(Intel
    FAIL_MESSAGE "Can't find package Intel..."
    FOUND_VAR Intel_FOUND
    REQUIRED_VARS 
	  Intel_M_LIBRARY 
	  Intel_MMT_LIBRARY 
	  Intel_MMD_LIBRARY
	  Intel_IRC_LIBRARY
	  Intel_SVLM_DISPMT_LIBRARY 
	  Intel_SVLM_DISPMD_LIBRARY
	  Intel_DECIMAL_LIBRARY 
	  Intel_IFCONSOL_LIBRARY 
	  Intel_IFCOREMD_LIBRARY
	  Intel_IFPORTMD_LIBRARY
    )
	
  if(Intel_FOUND AND NOT TARGET intel)
    # création de la cible
    add_library(intel INTERFACE IMPORTED)
    # librairie
    set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_M_LIBRARY})
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_MMT_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_MMD_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_IRC_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_SVLM_DISPMT_LIBRARY} APPEND)
    set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_SVLM_DISPMD_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_DECIMAL_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_IFCONSOL_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_IFCOREMD_LIBRARY} APPEND)
	  set_property(TARGET intel PROPERTY INTERFACE_LINK_LIBRARIES ${Intel_IFPORTMD_LIBRARY} APPEND)
  endif()

endif()
