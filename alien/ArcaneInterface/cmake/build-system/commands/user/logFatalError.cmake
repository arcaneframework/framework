function(logFatalError message)
  
  message(FATAL_ERROR "${BoldRed}FATAL ERROR :${ColourReset} ${message}")
  
endfunction()
