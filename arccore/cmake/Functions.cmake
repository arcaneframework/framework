# ----------------------------------------------------------------------------
# Fonction pour créér une bibliothèque avec des sources.
#
# Par exemple:
#
# arccore_add_library(arccore_base
#   INTUT_PATH ...
#   RELATIVE_PATH arccore/base
#   FILES ArccoreGlobal.cc ArccoreGlobal.h ...
# )
#
# Chaque fichier spécifié dans ${FILES} doit trouver dans le
# répertoire ${INPUT_PATH}/${RELATIVE_PATH}. FILES peut contenir
# des sous-répertoires.

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

function(arccore_add_library target)
  set(options        )
  set(oneValueArgs   INPUT_PATH RELATIVE_PATH)
  set(multiValueArgs FILES)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(_FILES)
  #set(_INSTALL_FILES)
  foreach(isource ${ARGS_FILES})
    set(_FILE ${ARGS_INPUT_PATH}/${ARGS_RELATIVE_PATH}/${isource})
    #message(STATUS "FOREACH i='${_FILE}'")
    # Regarde si le fichier est un fichier d'en-tête et si
    # c'est le cas, l'installe automatiquement
    # TOTO: rendre l'installation optionnelle.
    string(REGEX MATCH "\.(h|H)$" _OUT_HEADER ${isource})
    if (_OUT_HEADER)
      #list(APPEND _INSTALL_FILES ${_FILE})
      get_filename_component(_HEADER_RELATIVE_PATH ${isource} DIRECTORY)
      string(REGEX MATCH "internal/" _INTERNAL_HEADER ${isource})
      if (NOT _INTERNAL_HEADER)
        if (DEFINED ARGS_RELATIVE_PATH)
          set(_FILE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${ARGS_RELATIVE_PATH}/${_HEADER_RELATIVE_PATH}")
        else()
          set(_FILE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_HEADER_RELATIVE_PATH}")
        endif()
        install(FILES ${_FILE} DESTINATION "${_FILE_DESTINATION}")
      endif()
    endif()
    list(APPEND _FILES ${_FILE})
  endforeach()
  #message(STATUS "TARGET=${target} FILES=${_FILES}")
  add_library(${target} ${_FILES})

  if (ARCCORE_EXPORT_TARGET)
    install(TARGETS ${target} EXPORT ${ARCCORE_EXPORT_TARGET}
      LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
      RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
      ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    add_library(Arccore::${target} ALIAS ${target})
  endif()

  set(_libpath ${CMAKE_BINARY_DIR}/lib)
  if (WIN32)
    set_target_properties(${target}
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_DEBUG ${_libpath}
      RUNTIME_OUTPUT_DIRECTORY_DEBUG ${_libpath}
      LIBRARY_OUTPUT_DIRECTORY_RELEASE ${_libpath}
      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${_libpath}
      LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_libpath}
      RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_libpath}
      )
  else()
    set_target_properties(${target}
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${_libpath}
      RUNTIME_OUTPUT_DIRECTORY ${_libpath}
      )
  endif()
  # Ajoute au RPATH celui des bibliothèques et utilise $ORIGIN
  set_target_properties(${target}
    PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH 1
    BUILD_RPATH_USE_ORIGIN 1
    )
endfunction()

# ----------------------------------------------------------------------------
# Fonction pour créér un composant 'Arccore'
#
# arccore_add_component_library(component_name
#   FILES ...
# )
#
# Par exemple:
#
# arccore_add_component_library(base
#   FILES ArccoreGlobal.cc ArccoreGlobal.h ...
# )
#
# Cette fonction va enregistrer une cible 'arccore_${component_name}'
# Par convention, chaque fichier spécifié dans FILES doit se trouver dans le
# répertoire ${Arccore_SOURCE_DIR}/src/${component_name}.
#
function(arccore_add_component_library component_name)
  set(options)
  set(oneValueArgs LIB_NAME)
  set(multiValueArgs FILES)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (NOT ARGS_LIB_NAME)
    set(_LIB_NAME arccore_${component_name})
  else()
    set(_LIB_NAME ${ARGS_LIB_NAME})
    message(STATUS "Adding component library '${component_name}' with library name '${_LIB_NAME}'")
  endif()

  arccore_add_library(${_LIB_NAME}
          INPUT_PATH ${Arccore_SOURCE_DIR}/src/${component_name}
          RELATIVE_PATH arccore/${component_name}
          FILES ${ARGS_FILES}
          )

  target_compile_definitions(${_LIB_NAME} PRIVATE ARCCORE_COMPONENT_${_LIB_NAME})
  target_include_directories(${_LIB_NAME} PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/${component_name}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(arccore_full INTERFACE ${_LIB_NAME})

  # Génère les bibliothèques dans le répertoire 'lib' du projet.
  #set_target_properties(${_LIB_NAME} PROPERTIES
  #  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  #  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  #  )

endfunction()

# ----------------------------------------------------------------------------
# Fonction pour créér l'exécutable de test d'un composant 'Arccore'
#
# arccore_add_component_test_executable(component_name
#   FILES ...
# )
#
# Par exemple:
#
# arccore_add_component_test_executable(base
#   FILES ArccoreGlobal.cc ArccoreGlobal.h ...
# )
#
function(arccore_add_component_test_executable component_name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs FILES)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(_EXE_NAME arccore_${component_name}.tests)
  add_executable(${_EXE_NAME} ${ARGS_FILES})

  # Génère les exécutables dans le répertoire 'lib' du projet.
  # Cela permet sous windows de trouver automatiquement les dll des composantes

  set(_exepath ${CMAKE_BINARY_DIR}/lib)
  if (WIN32)
    set_target_properties(${_EXE_NAME}
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY_DEBUG ${_exepath}
      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${_exepath}
      RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_exepath}
      )
  else()
    set_target_properties(${_EXE_NAME}
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${_exepath}
      )
  endif()

endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Fonction pour ajoute un répertoire contenant un composant 'Arccore'.
#
# arccore_add_component_directory(component_name)
#
# Cette fonction va regarder si un répertoire 'src/${X}/arccore/${X} existe
# (avec ${X} valant ${component_name} et si c'est le cas, l'ajouter au
# projet.
# Si un répertoire 'src/${X}/tests' existe, il est aussi ajouté pour les
# tests si ces derniers sont demandés et que google-test est disponible
#
function(arccore_add_component_directory name)
  set(_BASE_DIR src/${name})
  set(_COMPONENT_DIR ${_BASE_DIR}/arccore/${name})
  set(_TEST_DIR ${_BASE_DIR}/tests)
  message(STATUS "Check component '${name}' in directory '${_COMPONENT_DIR}'")
  if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_COMPONENT_DIR})
    message(STATUS "Adding component '${name}' in directory '${_COMPONENT_DIR}'")
    add_subdirectory(${_COMPONENT_DIR})
    if (ARCCORE_WANT_TEST AND GTEST_FOUND)
      if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_TEST_DIR}/CMakeLists.txt)
        add_subdirectory(${_TEST_DIR})
      endif()
    endif()
  endif()
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Indique que les fichiers passés en argument doivent être compilés avec
# le support accélérateur correspondant.
macro(arccore_accelerator_add_source_files)
  if (ARCCORE_HAS_CUDA)
    foreach(_x ${ARGN})
      message(STATUS "Add CUDA language to file '${_x}'")
      set_source_files_properties(${_x} PROPERTIES LANGUAGE CUDA)
    endforeach()
  endif()
  if (ARCCORE_HAS_HIP)
    foreach(_x ${ARGN})
      message(STATUS "Add HIP language to file '${_x}'")
      set_source_files_properties(${_x} PROPERTIES LANGUAGE HIP)
    endforeach()
  endif()
  if (ARCCORE_HAS_SYCL)
    foreach(_x ${ARGN})
      message(STATUS "Add SYCL language to file '${_x}'")
      set_source_files_properties(${_x} PROPERTIES COMPILE_OPTIONS "${ARCCORE_CXX_SYCL_FLAGS}")
    endforeach()
  endif()
endmacro()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
