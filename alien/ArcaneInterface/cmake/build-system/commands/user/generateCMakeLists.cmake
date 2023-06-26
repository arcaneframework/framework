function(generateCMakeLists library)
 
  if(NOT TARGET ${library})
    logFatalError("Library ${library} is not defined - use createLibrary(${library})")
  endif()

  if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/config.xml)
    logFatalError("File config.xml doesn't exist in directory ${CMAKE_CURRENT_LIST_DIR}")
  endif()

  # chemin relatif par rapport au répertoire de la librairie
  file(RELATIVE_PATH RELPATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR})

  if(RELPATH)
    set(path ${CMAKE_CURRENT_BINARY_DIR}/${RELPATH})
  else()
    set(path ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  execute_process(COMMAND ${MONO_EXEC} 
    ${CMAKELIST_GENERATOR}
    ${library}
    ${CMAKE_CURRENT_LIST_DIR}
    ${path}
    --recursive-convert
    --force
    )

  include(${path}/CMakeLists.txt)

endfunction()
