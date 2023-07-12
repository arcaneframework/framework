macro(loadMeta)

  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
 
  if(NOT ARGS_NAME)
    logFatalError("create_meta needs NAME")
  endif()
 
  if(TARGET ARGS_NAME)
    logFatalError("target ${ARGS_NAME} for meta already defined")
  endif()

  string(TOLOWER ${ARGS_NAME} target)
  string(TOUPPER ${ARGS_NAME} TARGET)

  if(NOT ${target}_IS_DISABLED)

    add_library(${target} INTERFACE)
    set(${target}_IS_META ON)
    set(${TARGET}_FOUND ON)

    set(${PROJECT_NAME}_${TARGET} TRUE)
	
    get_property(TARGETS GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS)
    list(APPEND TARGETS ${target})
    list(REMOVE_DUPLICATES TARGETS)
	  set_property(GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS ${TARGETS})
	
  endif()
  
endmacro()
