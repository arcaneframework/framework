# NB: macro et non fonction car sinon, toutes les variable_requires
# éventuellement définies dans les langages restent uniquement
# dans le scope de la fonction
macro(loadLanguage)
  
  set(options)
  set(oneValueArgs NAME PATH)
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
  
  if(NOT ARGS_NAME) 
    logFatalError("load_language error, name is undefined")
  endif()

  string(TOUPPER ${ARGS_NAME} upper)

  if(NOT DEFINED USE_LANGUAGE_${upper})
    set(USE_LANGUAGE_${upper} ON)
  endif()
  
  if(${USE_LANGUAGE_${upper}})
    if(NOT ARGS_PATH) 
      get_filename_component(SELF_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
      set(path ${SELF_DIR}/languages)
    else()
      if(IS_ABSOLUTE ${ARGS_PATH})
        set(path ${ARGS_PATH})
      else()
        get_filename_component(SELF_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
        set(path ${SELF_DIR}/${ARGS_PATH})
      endif()
    endif()
    
    if(NOT EXISTS ${path}/${ARGS_NAME}.cmake)
      logFatalError("langage file '${path}/${ARGS_NAME}.cmake' not found")
    endif()
    
    include(${path}/${ARGS_NAME}.cmake)

    set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_LANGUAGES ${ARGS_NAME})
  else()
    logStatus("${Yellow}Warning${ColourReset} Language '${ARGS_NAME}' disabled by user")
  endif()

endmacro()
