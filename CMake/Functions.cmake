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
