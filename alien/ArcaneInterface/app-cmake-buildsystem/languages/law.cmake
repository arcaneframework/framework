include(${ARCGEOSIM_BUILD_SYSTEM_PATH}/languages/law/LoadLawCompiler.cmake)

set(law_share_path ${CMAKE_BINARY_DIR}/share/law)

if(NOT EXISTS ${law_share_path})
  file(MAKE_DIRECTORY ${law_share_path})
endif()

function(generateLaw target)

  set(options        NO_COPY DEBUG_MODE SEQUENTIAL_MODE)
  set(oneValueArgs   )
  set(multiValueArgs )
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  set(law_path ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/law)
  
  set(multithreading_mode_generation "")
  if(NOT ARGS_SEQUENTIAL_MODE)
    if(DEFINED USE_LANGUAGE_LAW_MULTI_THREADING)
      if(USE_LANGUAGE_LAW_MULTI_THREADING STREQUAL "ArcaneTBB")
        set(multithreading_mode_generation "--multi-threading=ArcaneTBB")
      endif()
      if(USE_LANGUAGE_LAW_MULTI_THREADING STREQUAL "Kokkos")
        set(multithreading_mode_generation "--multi-threading=Kokkos")
      endif()
    endif()
  endif()
  
  set(debug_mode_generation "")
  if(ARGS_DEBUG_MODE)
    set(debug_mode_generation "--debug")
  endif()
  if(DEFINED USE_LANGUAGE_LAW_DEBUG)
    if(${USE_LANGUAGE_LAW_DEBUG})
      set(debug_mode_generation "--debug")
    endif()
  endif()
  
  if(NOT EXISTS ${law_path})
    file(MAKE_DIRECTORY ${law_path})
  endif()

  set(law ${ARGS_UNPARSED_ARGUMENTS})
  
  foreach(law_file ${law})
    
    get_filename_component(name ${law_file} NAME_WE)
   
    if(IS_ABSOLUTE ${law_file})
      set(file ${law_file})
    else()
      set(file ${CMAKE_CURRENT_LIST_DIR}/${law_file})
    endif()

    if(NOT EXISTS ${file})
      logFatalError("law file ${file} doesn't exist")
    endif()

    add_custom_command(
      OUTPUT  ${law_path}/${name}_law.h
      DEPENDS ${file} law
      COMMAND ${LAWCOMPILER} 
      ARGS    --law=${file} ${debug_mode_generation} ${multithreading_mode_generation}
              --path=${law_path}
      COMMENT "Building LAW generated file ${PROJECT_NAME}/law/${name}_law.h"
      )

    set_source_files_properties(
      ${law_path}/${name}_law.h PROPERTIES GENERATED ON
      )
    
    target_sources(${target} PRIVATE ${law_path}/${name}_law.h)

  endforeach()

  target_include_directories(${target} PUBLIC 
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/law)

endfunction()
