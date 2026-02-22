function(addSources target)

  # commit ?
  get_target_property(committed ${target} BUILDSYSTEM_COMMITTED)

  if(${committed})
    logFatalError("target ${target} is alreday committed, can't add sources")
  endif()

  foreach(source ${ARGN})
    if(IS_ABSOLUTE ${source})
      set(file ${source})
    else()
      set(file ${CMAKE_CURRENT_LIST_DIR}/${source})
    endif()
    if(NOT EXISTS ${file})
      logFatalError("Source file ${file} doesn't exist")
    endif()
    list(APPEND sources ${file})
  endforeach()

  # ajout des sources
  set_property(TARGET ${target} APPEND PROPERTY BUILDSYSTEM_SOURCES ${sources})

endfunction()
