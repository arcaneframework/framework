function(printLanguageInformations)

  get_property(languages GLOBAL PROPERTY ${PROJECT_NAME}_LANGUAGES)
   
  foreach(language ${languages})
    logStatus(" ** loaded language : ${Blue}${language}${ColourReset}")
  endforeach()

endfunction()
