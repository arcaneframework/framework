function(importPackageXmlFile)

  set(options)
  set(oneValueArgs TARGET XML)
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()

  if(NOT ARGS_TARGET) 
    logFatalError("import_targets error, target is undefined")
  endif()

  if(NOT ARGS_XML) 
    logFatalError("import_targets error, from is undefined")
  endif()

  execute_process(COMMAND ${MONO_EXEC} 
    ${PKGLIST_LOADER}
    ${ARGS_XML}
    ${PROJECT_BINARY_DIR}/${ARGS_TARGET}.cmake
    --add_prefix
    )
  
  include(${PROJECT_BINARY_DIR}/${ARGS_TARGET}.cmake)
  
  set(conflicts OFF)

  foreach(TARGET ${DELEGATED_DEPENDENCIES})
    string(TOLOWER ${TARGET} target)
    if(NOT TARGET ${target})
      if(NOT ${target}_IS_DISABLED)
        add_library(${target} INTERFACE IMPORTED)
        if(XML_${TARGET}_LIBRARIES)
          set_property(TARGET ${target} APPEND PROPERTY 
            INTERFACE_LINK_LIBRARIES "${XML_${TARGET}_LIBRARIES}")
		      set(${TARGET}_LIBRARIES ${XML_${TARGET}_LIBRARIES})
          set(${TARGET}_LIBRARIES ${${TARGET}_LIBRARIES} PARENT_SCOPE)
        endif()
        if(XML_${TARGET}_INCLUDE_DIRS)
          set_property(TARGET ${target} APPEND PROPERTY 
            INTERFACE_INCLUDE_DIRECTORIES "${XML_${TARGET}_INCLUDE_DIRS}")
		      set(${TARGET}_INCLUDE_DIRS ${XML_${TARGET}_INCLUDE_DIRS})
          set(${TARGET}_INCLUDE_DIRS ${${TARGET}_INCLUDE_DIRS} PARENT_SCOPE)
        endif()
        if(XML_${TARGET}_FLAGS)
          set_property(TARGET ${target} APPEND PROPERTY 
            INTERFACE_COMPILE_DEFINITIONS "${XML_${TARGET}_FLAGS}")
          set(${TARGET}_FLAGS ${XML_${TARGET}_FLAGS})
          set(${TARGET}_FLAGS ${${TARGET}_FLAGS} PARENT_SCOPE)
        endif()
	      set(${target}_IS_LOADED ON PARENT_SCOPE)
        set(${target}_IMPORTED ${ARGS_TARGET} PARENT_SCOPE)
        if(WIN32)
          copyAllDllFromTarget(${target})
        endif()
	      set_property(TARGET ${ARGS_TARGET} APPEND PROPERTY 
          INTERFACE_LINK_LIBRARIES "${target}")
      endif()
      list(APPEND ${PROJECT_NAME}_TARGETS ${target})
    else()
      foreach(lib ${XML_${TARGET}_LIBRARIES})
        if(NOT (${lib} IN_LIST ${TARGET}_LIBRARIES))
          if(NOT ${conflicts})
            message_separator()
            set(conflicts ON)
          endif()
          logStatus("${Red}Conflicting${ColourReset} configuration between ${BoldMagenta}${ARGS_TARGET}${ColourReset} and ${BoldMagenta}${PROJECT_NAME}${ColourReset} for target ${target}")
          logStatus("${lib} not in ${${TARGET}_LIBRARIES}")
          break()
        endif()
      endforeach()
      foreach(lib ${XML_${TARGET}_INCLUDE_DIRS})
        if(NOT (${lib} IN_LIST ${TARGET}_INCLUDE_DIRS))
          if(NOT ${conflicts})
            message_separator()
            set(conflicts ON)
          endif()
          logStatus("${Red}Conflicting${ColourReset} configuration between ${BoldMagenta}${ARGS_TARGET}${ColourReset} and ${BoldMagenta}${PROJECT_NAME}${ColourReset} for target ${target}")
          logStatus("${lib} not in ${${TARGET}_INCLUDE_DIRS}")
          break()
        endif()
      endforeach()
    endif()
  endforeach()

  get_property(TARGETS GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS)
  list(APPEND TARGETS ${target})
  list(REMOVE_DUPLICATES TARGETS)
  set_property(GLOBAL PROPERTY ${PROJECT_NAME}_TARGETS ${TARGETS})

endfunction()