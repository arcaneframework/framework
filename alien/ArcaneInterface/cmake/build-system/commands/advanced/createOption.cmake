macro(createOption)

  set(options)
  set(oneValueArgs COMMANDLINE NAME MESSAGE DEFAULT)
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
  
  # ajout d'option cmake
  option(${ARGS_COMMANDLINE} ${ARGS_MESSAGE} ${ARGS_DEFAULT})
  
  # création d'une variable par option
  if(${${ARGS_COMMANDLINE}})
    set(${ARGS_NAME} True) 
  else()
    set(${ARGS_NAME} False)
  endif()
  set(${ARGS_NAME}_MESSAGE ${ARGS_MESSAGE}) 
  set(${ARGS_NAME}_IS_BOOLEAN True) 
  set(${ARGS_NAME}_COMMANDLINE ${ARGS_COMMANDLINE}) 

  set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_OPTIONS ${ARGS_NAME})
  
endmacro()

macro(createStringOption)

  set(options)
  set(oneValueArgs COMMANDLINE NAME MESSAGE DEFAULT)
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()

  # création d'une variable par option
  if(DEFINED ${ARGS_COMMANDLINE})
    set(${ARGS_NAME} True)
    set(${ARGS_NAME}_VALUE ${${ARGS_COMMANDLINE}}) 
  elseif(DEFINED ARGS_DEFAULT)
    set(${ARGS_NAME} True)
    set(${ARGS_NAME}_VALUE ${ARGS_DEFAULT}) 
  else()
    set(${ARGS_NAME} False)
  endif()
  set(${ARGS_NAME}_MESSAGE ${ARGS_MESSAGE}) 
  set(${ARGS_NAME}_IS_BOOLEAN OFF) 
  set(${ARGS_NAME}_COMMANDLINE ${ARGS_COMMANDLINE}) 

  set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_OPTIONS ${ARGS_NAME})
  
endmacro()
