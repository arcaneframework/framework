# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(WIN32)
 
  message_separator()

  logStatus("Finishing build system :")
  
  list(LENGTH EXTRA_DLLS_TO_COPY nb_dlls)
  
  logStatus(" ** copying ${nb_dlls} extra dll(s) to binary configurations directories")

  foreach(dll ${EXTRA_DLLS_TO_COPY})
    if(VERBOSE)
      logStatus("  * dll : ${dll}")
	  endif()
    copyOneDllFile(${dll})
  endforeach()
  
  get_property(DLLS GLOBAL PROPERTY ${PROJECT_NAME}_DLLS_TO_COPY)
 
  list(LENGTH DLLS nb_dlls)
  
  logStatus(" ** copying ${nb_dlls} dll(s) to binary configurations directories")
  
  if(VERBOSE)
    foreach(dll ${DLLS})
      logStatus(" * dll : ${dll}")
    endforeach()
  endif()
  
  add_custom_target(copy_dll ALL
    DEPENDS ${DLLS}
	)
  
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message_separator()

logStatus("Generate xml package file")

generatePackageXmlFile()

logStatus("Generate axl database")
if(USE_AXLSTAR)
generateAxlDataBase()
endif()

logStatus("Generate eclipse CDT settings")

generateEclipseCDTXmlFile()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message_separator()

logStatus("To compile, I suggest the following compile command:")
if(WIN32)
  if(CMAKE_BUILD_TYPE)
    logStatus("\tdevenv ${CMAKE_PROJECT_NAME}.sln /build ${CMAKE_BUILD_TYPE}")
  else()
    logStatus("\tdevenv ${CMAKE_PROJECT_NAME}.sln /build Debug")
    logStatus("\tor")
    logStatus("\tdevenv ${CMAKE_PROJECT_NAME}.sln /build Release")
  endif()
else()
  logStatus("\tmake -j 4")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
