
# ----------------------------------------------------------------------------
# Fonction pour rechercher et enregistrer sous la forme d'une cible
# importée un package classique dont il n'existe pas de module correspondant
# dans CMake.
# TODO: Supporter plusieurs .h et pluseurs libs et mettre dans Arccon
function(arccon_find_legacy_package)
  set(options)
  set(oneValueArgs   NAME LIBRARIES HEADERS)
  set(multiValueArgs)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(_PKG_NAME ${ARGS_NAME})
  arccon_return_if_package_found(${_PKG_NAME})
  message(STATUS "Searching library '${ARGS_LIBRARIES}'")
  find_library(${_PKG_NAME}_LIBRARY ${ARGS_LIBRARIES})

  message(STATUS "Searching header '${ARGS_HEADERS}'")
  # On debian/ubuntu, headers can be found in a /usr/include/"pkg"
  string(TOLOWER ${_PKG_NAME} _LOW_PKG_NAME)
  find_path(${_PKG_NAME}_HEADER ${ARGS_HEADERS}
          PATH_SUFFIXES ${_PKG_NAME} ${_LOW_PKG_NAME})

  message(STATUS "Pkg: ${_PKG_NAME} libs=${${_PKG_NAME}_LIBRARY}")
  message(STATUS "Pkg: ${_PKG_NAME} incs=${${_PKG_NAME}_HEADER}")
  set(${_PKG_NAME}_FOUND FALSE)
  if (${_PKG_NAME}_LIBRARY AND ${_PKG_NAME}_HEADER)
    set(${_PKG_NAME}_FOUND TRUE)
    set(${_PKG_NAME}_LIBRARIES ${${_PKG_NAME}_LIBRARY})
    set(${_PKG_NAME}_INCLUDE_DIRS ${${_PKG_NAME}_HEADER})
  endif()
  arccon_register_package_library(${_PKG_NAME} ${_PKG_NAME})
endfunction()
