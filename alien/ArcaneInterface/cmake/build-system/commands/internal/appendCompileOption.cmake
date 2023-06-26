macro(appendCompileOption)
  
  set(options        )
  set(oneValueArgs   )
  set(multiValueArgs FLAG CONFIGURATION LANGUAGE)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("append_compile_option error, unparsed arguments")
  endif()

  if(NOT ARGS_FLAG)
    logFatalError("append_compile_option needs FLAG")
  endif()

  if(NOT ARGS_CONFIGURATION)
    set(ARGS_CONFIGURATION DEBUG RELEASE) 
  endif()

  if(NOT ARGS_LANGUAGE)
    set(ARGS_LANGUAGE CXX CC) 
  endif()

  foreach(flag ${ARGS_FLAG})
    foreach(configuration ${ARGS_CONFIGURATION})
      foreach(language ${ARGS_LANGUAGE})
        if(${CMAKE_C_COMPILER_ID} STREQUAL MSVC)
		  set(CMAKE_${language}_FLAGS_${configuration} "${CMAKE_${language}_FLAGS_${configuration}} /${flag}")
        else()
          set(CMAKE_${language}_FLAGS_${configuration} "${CMAKE_${language}_FLAGS_${configuration}} -${flag}")
        endif()
      endforeach()
    endforeach()
  endforeach()

endmacro()
