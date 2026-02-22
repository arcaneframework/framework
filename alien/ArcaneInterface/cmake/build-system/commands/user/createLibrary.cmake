function(createLibrary library)

  set(options       )
  set(oneValueArgs   EXPORT)
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("createLibrary error, argument error : ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  if(TARGET ${library})
    logFatalError("library ${library} already defined")
  endif()

  logStatus(" ** Load library component ${BoldBlue}${library}${ColourReset}")

  # création de la librarie
  add_library(${library} "")
  
  if(USE_PROJECT_CONFIG_HEADER)
  # Pour trouver le bon <Project>Config.h
  target_include_directories(${library} PUBLIC
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
          )
  endif()
  if(NOT USE_AXLSTAR)
    target_include_directories(${library} PRIVATE ${AXL_HEADERS_PATH})
  endif()

  # librarie non commitée
  set_target_properties(${library} PROPERTIES BUILDSYSTEM_COMMITTED OFF)

  # nommage de l'export pour dll
  if(NOT ARGS_EXPORT)
    set_target_properties(${library} PROPERTIES BUILDSYSTEM_EXPORT_DLL ${library}Export.h)
  else()
    set_target_properties(${library} PROPERTIES BUILDSYSTEM_EXPORT_DLL ${ARGS_EXPORT})
  endif()
   
  # librarie 
  set_target_properties(${library} PROPERTIES BUILDSYSTEM_TYPE LIBRARY)

  # liste des builtin libraries  
  set_property(GLOBAL APPEND PROPERTY BUILDSYSTEM_BUILTIN_LIBRARIES ${library})

endfunction()
