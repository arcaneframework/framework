include(${ARCGEOSIM_BUILD_SYSTEM_PATH}/languages/gump/LoadGumpCompiler.cmake)

set(gump_share_path ${CMAKE_BINARY_DIR}/share/gump)

if(NOT EXISTS ${gump_share_path})
  file(MAKE_DIRECTORY ${gump_share_path})
endif()

function(generateGumpModel xml)
  
  get_filename_component(model ${xml} NAME_WE)
  
  logStatus("  * Load gump component ${BoldMagenta}${model}${ColourReset}")

  set(gump_path ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/gump/${model})
  
  if(NOT EXISTS ${gump_path})
    file(MAKE_DIRECTORY ${gump_path})
  endif()
  
  if(VERBOSE)
    set(verbose_args "--verbose")
  endif()
  
  if(IS_ABSOLUTE ${xml})
    set(file ${xml})
  else()
    set(file ${CMAKE_CURRENT_LIST_DIR}/${xml})
  endif()
  
  if(NOT EXISTS ${file})
    logFatalError("gump xml file ${file} doesn't exist")
  endif()
  
  add_custom_command(
    OUTPUT  ${gump_path}/Entities.h
    DEPENDS ${file} gump
    COMMAND ${GUMPCOMPILER} 
    ARGS    ${verbose_args}
            --path=${CMAKE_BINARY_DIR}/${PROJECT_NAME}/gump
            --xml=${file}
    COMMENT "Building GUMP model ${PROJECT_NAME}/gump/${model}"
    )

  set_source_files_properties(
    ${gump_path}/Entities.h PROPERTIES GENERATED ON
    )
  
  configure_file(
    ${ARCGEOSIM_BUILD_SYSTEM_PATH}/languages/gump/gump.cc.in
    ${gump_path}/${model}.cc
    @ONLY
    )

  string(TOLOWER ${model} lib) 

  add_library(${lib} 
    ${gump_path}/${model}.cc 
    ${gump_path}/Entities.h
    )
	  
  target_include_directories(${lib} PUBLIC 
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/gump)

endfunction()
