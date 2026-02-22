function(generateDynamicLoading target)

  get_filename_component(EXE_NAME ${target} NAME_WE)
  
  get_target_property(libraries ${target} DYNAMIC_LIBRARIES)
  
  if(libraries)
    foreach(library ${libraries})
      set(DYNAMIC_LIBRARIES "${DYNAMIC_LIBRARIES}      loader->load(\"${library}\");\n")
    endforeach()
  endif()

  configure_file(${BUILD_SYSTEM_PATH}/dynamicloading/DynamicLoading.h.in
                 ${EXE_NAME}DynamicLoading.h
                 @ONLY
    )

  target_include_directories(${target} PRIVATE 
    ${BUILD_SYSTEM_PATH}/dynamicloading
    ${CMAKE_CURRENT_BINARY_DIR}
    )
  
  if(NOT TARGET glib)
    logFatalError("glib is mandatory for dynamic loading")
  endif()

  target_link_libraries(${target} PRIVATE glib)

endfunction()
