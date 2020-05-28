function(createExecutable exe)

  if(TARGET ${exe})
    logFatalError("executable ${exe} already defined")
  endif()

  logStatus(" ** Load executable component ${BoldRed}${exe}${ColourReset}")

  # création de l'executable
  add_executable(${exe} "")
  
  # executable non commité
  set_target_properties(${exe} PROPERTIES BUILDSYSTEM_COMMITTED OFF)
  
  # executable 
  set_target_properties(${exe} PROPERTIES BUILDSYSTEM_TYPE EXECUTABLE)
 
  # liste des builtin executables  
  set_property(GLOBAL APPEND PROPERTY BUILDSYSTEM_BUILTIN_EXECUTABLES ${exe})

endfunction()
