macro(disablePackage)

  set(options)
  set(oneValueArgs NAME WHY)
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()

  if(NOT ARGS_NAME) 
    logFatalError("disable_package error, name is undefined")
  endif()

  string(TOLOWER ${ARGS_NAME} package)

  set(${package}_IS_DISABLED ON)
  set(${package}_WHY ${ARGS_WHY})

endmacro()
