function(linkWholeArchiveLibraries target)

  set(options        )
  set(oneValueArgs   )
  set(multiValueArgs PUBLIC PRIVATE)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(WIN32)
    
    foreach(lib ${ARGS_PUBLIC})
      target_link_libraries(${target} PUBLIC ${lib})
	  endforeach()
    foreach(lib ${ARGS_PRIVATE})
      target_link_libraries(${target} PRIVATE ${lib})
	  endforeach()
    
	if(NOT ${BUILD_SHARED_LIBS})
	  logStatus("whole archive for target ${target}")
      add_custom_target(
        vcproj_patch_for_${target}  ALL
        COMMAND ${WHOLEARCHIVE_VCPROJ_TOOL} --visual="${CMAKE_GENERATOR}" --path="${CMAKE_CURRENT_BINARY_DIR}" --project=${target} --dont-remove-suffix
        COMMENT "Patch visual project for ${target}"
        )
	  add_dependencies(${target} vcproj_patch_for_${target})
    endif()
	
  else()
    
    foreach(lib ${ARGS_PUBLIC})
      if(${BUILD_SHARED_LIBS})
        target_link_libraries(${target} PUBLIC ${lib})
      else()
        target_link_libraries(${target} PUBLIC -Wl,-whole-archive ${lib} -Wl,-no-whole-archive)
      endif()
    endforeach()
    
    foreach(lib ${ARGS_PRIVATE})
      if(${BUILD_SHARED_LIBS})
        target_link_libraries(${target} PRIVATE ${lib})
      else()
        target_link_libraries(${target} PRIVATE -Wl,-whole-archive ${lib} -Wl,-no-whole-archive)
      endif()
    endforeach()
	  
  endif()
  
endfunction()
