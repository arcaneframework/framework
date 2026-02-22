function(copyOneDllFile dll)

  get_filename_component(name ${dll} NAME)
  
  set(dll_copy ${BUILDSYSTEM_DLL_COPY_DIRECTORY}/${name})

  logStatus("Copying dll ${dll} into ${dll_copy}")

  add_custom_command(
	  OUTPUT ${dll_copy}
	  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dll} ${dll_copy}
	)

  set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_DLLS_TO_COPY ${dll_copy})
  
endfunction()
