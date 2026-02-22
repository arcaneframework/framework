function(createExecutable exe)

  if(TARGET ${exe})
    logFatalError("executable ${exe} already defined")
  endif()

  logStatus(" ** Load executable component ${BoldRed}${exe}${ColourReset}")

  # création de l'executable
  add_executable(${exe} "")
  
  if(USE_PROJECT_CONFIG_HEADER)
  # Pour trouver le bon <Project>Config.h
  target_include_directories(${exe} PUBLIC
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
          )
  endif()

  if(NOT USE_AXLSTAR)
    target_include_directories(${exe} PRIVATE ${AXL_HEADERS_PATH})
  endif()

  # executable non commité
  set_target_properties(${exe} PROPERTIES BUILDSYSTEM_COMMITTED OFF)
  
  # executable 
  set_target_properties(${exe} PROPERTIES BUILDSYSTEM_TYPE EXECUTABLE)
 
  # liste des builtin executables  
  set_property(GLOBAL APPEND PROPERTY BUILDSYSTEM_BUILTIN_EXECUTABLES ${exe})

endfunction()
