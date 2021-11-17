function(addDirectory directory)

  # inclusion du répertoire
  # on utilise include car add_subdirectory change le scope 
  # et cela est problématique avec add_custom_command 

  if(IS_ABSOLUTE ${directory})
    set(path ${directory})
    file(RELATIVE_PATH directory ${CMAKE_CURRENT_LIST_DIR} ${directory})
  else()
    set(path ${CMAKE_CURRENT_LIST_DIR}/${directory})
  endif()

  if(NOT IS_DIRECTORY ${path})
    logFatalError("path ${path} is not a directory")
  endif()

  if(NOT EXISTS ${path}/CMakeLists.txt)
    logFatalError("Directory ${directory} doesn't contains CMakeLists.txt file")
  endif()
 
  file(RELATIVE_PATH optional ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${directory})

  if(NOT DEFINED ${optional})
    include(${path}/CMakeLists.txt)
  elseif(${${optional}})
    include(${path}/CMakeLists.txt)
  else()
    logStatus("${Yellow}Warning${ColourReset} Sub-directory '${directory}' excluded by user")
  endif()
  
endfunction()
