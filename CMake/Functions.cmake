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

function(arccore_add_library target)
  set(options        )
  set(oneValueArgs   INPUT_PATH RELATIVE_PATH)
  set(multiValueArgs FILES)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(rel_path ${ARGS_RELATIVE_PATH})
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
      install(FILES ${_FILE} DESTINATION include/${rel_path}/${_HEADER_RELATIVE_PATH})
    endif()
    list(APPEND _FILES ${_FILE})
  endforeach()
  #message(STATUS "TARGET=${target} FILES=${_FILES}")
  add_library(${target} ${_FILES})
  #if (_INSTALL_FILES)
  #install(FILES ${_INSTALL_FILES} DESTINATION include/${rel_path})
  #endif()
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
  set(oneValueArgs)
  set(multiValueArgs FILES)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(_LIB_NAME arccore_${component_name})
  arccore_add_library(${_LIB_NAME}
          INPUT_PATH ${Arccore_SOURCE_DIR}/src/${component_name}
          RELATIVE_PATH arccore/${component_name}
          FILES ${ARGS_FILES}
          )

  target_compile_definitions(${_LIB_NAME} PRIVATE ARCCORE_COMPONENT_${_LIB_NAME})
  target_include_directories(${_LIB_NAME} PUBLIC $<BUILD_INTERFACE:${Arccore_SOURCE_DIR}/src/${component_name}> $<INSTALL_INTERFACE:include>)
  if (ARCCORE_EXPORT_TARGET)
    install(TARGETS ${_LIB_NAME} EXPORT ${ARCCORE_EXPORT_TARGET} DESTINATION lib)
  endif()

  # Génère les bibliothèques dans le répertoire 'lib' du projet.
  set_target_properties(${_LIB_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${Arccore_BINARY_DIR}/lib
    )
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
# Si un répertoire 'src/${X}/tests' existe, il est aussi ajouter pour les
# tests si ces derniers sont demandés.
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
      if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${_TEST_DIR})
        add_subdirectory(${_TEST_DIR})
      endif()
    endif()
  endif()
endfunction()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
