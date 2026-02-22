function(__commit_library target)

  # export pour les dlls
  get_target_property(export_dll ${target} BUILDSYSTEM_EXPORT_DLL)

  generate_export_header(${target}
    EXPORT_FILE_NAME ${export_dll}
    )
  target_sources(${target} PRIVATE ${export_dll})

  get_filename_component(path ${export_dll} DIRECTORY)

  # installation du fichier d'export pour les dll
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${export_dll}
          DESTINATION include/${path}
          )

  # installation de la librairie
  if(REQUIRE_INSTALL_PROJECTTARGETS)
  install(TARGETS ${target}
	  DESTINATION lib
	  EXPORT ${PROJECT_NAME}Targets
	  )

  # installation du CMake pour l'installation
  install(EXPORT ${PROJECT_NAME}Targets
	  DESTINATION lib/cmake/${PROJECT_NAME}
	  EXPORT_LINK_INTERFACE_LIBRARIES
	  )
  endif()
  # sources 
  get_target_property(sources ${target} BUILDSYSTEM_SOURCES)

  target_sources(${target} PRIVATE ${sources})

  # libraries 
  get_target_property(libraries ${target} BUILDSYSTEM_LIBRARIES)

  message(STATUS "BUILDSYSTEM_LIBRARIES ${libraries}")
  get_property(target_names DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
  message(STATUS "TARGET NAMES ${target_names}")

  list(REMOVE_DUPLICATES libraries)

  # cibles internes (compilées par le projet et déclarées via createLibrary)
  get_property(BUILTIN GLOBAL PROPERTY BUILDSYSTEM_BUILTIN_LIBRARIES)

  # check
  foreach(library ${libraries})
    if (TARGET ${library})
      set (MY_TARGET ${library})
    elseif (TARGET arccon::${library})
      set (MY_TARGET arccon::${library})
    elseif (${ARCCON_TARGET_${library}})
      if (TARGET ${ARCCON_TARGET_${library}})
      set(MY_TARGET ${ARCCON_TARGET_${library}})
      endif ()
    else ()
      logFatalError("undefined library ${library} linked with ${target}")
    endif ()
    if(${MY_TARGET} IN_LIST BUILTIN)
      get_target_property(committed ${MY_TARGET} BUILDSYSTEM_COMMITTED)
      if(NOT ${committed})
        logFatalError("builtin library ${MY_TARGET} is not committed - add commit(${MY_TARGET}) in CMakeLists.txt")
      endif()
    else()
      set_property(GLOBAL APPEND PROPERTY BUILDSYSTEM_EXTERNAL_LIBRARIES ${MY_TARGET})
    endif()
    if (FRAMEWORK_INSTALL)
      target_link_libraries(${target} PUBLIC $<BUILD_INTERFACE:${MY_TARGET}>)
    else ()
      target_link_libraries(${target} PUBLIC ${MY_TARGET})
    endif ()
  endforeach()

endfunction()

function(__commit_executable target)
  
  # sources 
  get_target_property(sources ${target} BUILDSYSTEM_SOURCES)

  target_sources(${target} PRIVATE ${sources})

  # libraries 
  get_target_property(libraries ${target} BUILDSYSTEM_LIBRARIES)

  list(REMOVE_DUPLICATES libraries)
  
  # cibles internes (compilées par le projet et déclarées via createLibrary)
  get_property(BUILTIN GLOBAL PROPERTY BUILDSYSTEM_BUILTIN_LIBRARIES)

  set(libraries_whole_archive)
	
  foreach(library ${libraries})

    # check
    if(NOT TARGET ${library})
      logFatalError("undefined library ${library} linked with ${target}")
    endif()

    if(${library} IN_LIST BUILTIN)
      get_target_property(committed ${library} BUILDSYSTEM_COMMITTED)
      if(NOT ${committed})
        logFatalError("builtin library ${library} is not committed - add commit(${library}) in CMakeLists.txt")
      endif()
      get_target_property(lib_type ${library} BUILDSYSTEM_TYPE)
      if(${lib_type} STREQUAL LIBRARY)
	    list(APPEND libraries_whole_archive ${library})
        # pour le chargement dynamique
        if(${BUILD_SHARED_LIBS} AND WIN32)
          set_property(TARGET ${target} APPEND PROPERTY DYNAMIC_LIBRARIES ${library})
        endif()
      else()
	    target_link_libraries(${target} PUBLIC ${library})
      endif()
    else()
      target_link_libraries(${target} PUBLIC ${library})
    endif()
	
  endforeach()
  
  # whole archive sur les libraries builtin
  linkWholeArchiveLibraries(${target} PUBLIC ${libraries_whole_archive})

  generateDynamicLoading(${target})

  # increase stack size
  if(WIN32)
    if (MSVC)
      set_target_properties(${target} PROPERTIES LINK_FLAGS /STACK:10000000)
    endif()
  endif()
  
endfunction()

function(commit target)

  set(options       )
  set(oneValueArgs  )
  set(multiValueArgs)
  
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("commit error, argument error : ${ARGS_UNPARSED_ARGUMENTS}")
  endif()

  # commit ?
  get_target_property(committed ${target} BUILDSYSTEM_COMMITTED)

  if(${committed})
    message(FATAL_ERROR "target ${target} is alreday committed")
  endif()

  # librairie ou executable
  get_target_property(type ${target} BUILDSYSTEM_TYPE)

  set(is_exe OFF)
  set(is_lib OFF)

  if(${type} STREQUAL EXECUTABLE)
    set(is_exe ON)
  endif()
  if(${type} STREQUAL LIBRARY)
    set(is_lib ON)
  endif()
    
  if(${is_exe} AND ${is_lib})
    logFatalError("Internal error, target is library and executable")
  endif()

  if(NOT ${is_exe} AND NOT ${is_lib})
    logWarning("target is not library neither executable built by buildsystem")
    return()
  endif()

  if(${is_lib})
    __commit_library(${target})
  endif()

  if(${is_exe})
    __commit_executable(${target})
  endif()

  set_target_properties(${target} PROPERTIES BUILDSYSTEM_COMMITTED ON)

endfunction()
