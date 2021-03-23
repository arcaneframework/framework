# ----------------------------------------------------------------------------
# Attention: ne marche qu'à partir de CMake 3.16.0
# (car la commande file(GET_RUNTIME_DEPENDENCIES) n'existe pas avant
#
# Ce fichier doit s'exécuter à partir d'une cible CMake.
# (voir le CMakeListst.txt pour un exemple)
# ----------------------------------------------------------------------------
message(STATUS "Execute CopyDlls")
if (CMAKE_VERSION VERSION_LESS "3.16")
  message(FATAL_ERROR "Your version of CMake (${CMAKE_VERSION}) is tool old. Minimum is 3.16")
endif()
if (NOT CONFIG_INCLUDE_FILE)
  message(FATAL_ERROR "Variable 'CONFIG_INCLUDE_FILE' is not defined")
endif()
message(STATUS "Including configuration file ${CONFIG_INCLUDE_FILE}")
include(${CONFIG_INCLUDE_FILE})

# TODO: à modifier en fonction de l'OS
if (WIN32)
  # Enlève sous windows le répertoire système contenant Windows.
  # Cela est utilisé avec l'argument POST_EXCLUDE_REGEXES
  # mais il semble que cela ne fonctionne pas toujours notamment
  # si le chemin est au format natif Windows (par exemple c:\windows)
  set(_SYSTEM_ROOT "$ENV{SystemRoot}")
  file(TO_CMAKE_PATH "${_SYSTEM_ROOT}" _SYSTEM_ROOT1)
  list(APPEND ARCANE_SYSTEM_DIRS "${_SYSTEM_ROOT1}")
endif()
if (UNIX)
  set(ARCANE_SYSTEM_DIRS "/lib64" "/lib/x86_64-linux-gnu" "/usr/lib64")
endif()
message(STATUS "SYSTEM_DIR=${ARCANE_SYSTEM_DIRS}")

set(LIBS_DIR)
foreach(_lib ${LIBS_LIST})
  get_filename_component(_dir ${_lib} DIRECTORY)
  message(STATUS "LIB=${_lib} DIR=${_dir}")
  if (WIN32)
  endif()
  if (WIN32)
    if (${_lib} MATCHES "(.*).lib$")
      list(APPEND LIBS_DIR "${_dir}")
      # Par convention sous windows, si une bibliothèque est dans
      # un répertoire 'x', alors la DLL correspondante est
      # dans ${x}/.../bin
      list(APPEND LIBS_DIR "${_dir}/../bin")
    endif()
    if (${_lib} MATCHES "(.*).dll$")
      message(STATUS "MATCH! ${CMAKE_MATCH_1}")
      list(APPEND LIBS_DIR "${_dir}")
    endif()
  endif()
  message(STATUS "libs_for '${_lib}':")
endforeach()

foreach(_dir ${LIBS_DIR})
  message(STATUS "DirToSearch=${_dir}")
endforeach()

foreach(_exe ${EXE_LIST})
  get_filename_component(_dir ${_exe} DIRECTORY)
  message(STATUS "EXE=${_exe} DIR=${_dir}")
  file(GET_RUNTIME_DEPENDENCIES
    RESOLVED_DEPENDENCIES_VAR var_ok
    UNRESOLVED_DEPENDENCIES_VAR var_unresolved
    CONFLICTING_DEPENDENCIES_PREFIX var_conflict
    EXECUTABLES ${_exe}
    DIRECTORIES ${_dir} ${LIBS_DIR} ${ARCANE_SYSTEM_DIRS}
    POST_EXCLUDE_REGEXES ${ARCANE_SYSTEM_DIRS}
  )
  message(STATUS "libs_for '${_lib}':")
  foreach(_libdepend ${var_ok})
    message(STATUS "  ${_libdepend}")
  endforeach()

  # Filtre les DLLs qui sont déja dans le chemin de l'exécutable
  # et qui sont dans le répertoire système si elles n'ont pas bien
  # été filtrées auparavant
  set(FINAL_LIST)
  foreach(_libdepend ${var_ok})
    file(TO_CMAKE_PATH "${_libdepend}" _libdepend1)
    set(_is_match FALSE)
    foreach(_regex ${ARCANE_SYSTEM_DIRS} ${_dir})
      message(STATUS "REGEX=${_regex} '${_libdepend1}'")
      if (${_libdepend1} MATCHES "^${_regex}")
        message(STATUS "MATCH1: ${_libdepend1}")
        set(_is_match TRUE)
      endif()
    endforeach()
    if (NOT _is_match)
      list(APPEND FINAL_LIST "${_libdepend1}")
    endif()
  endforeach()

  message(STATUS "Lib to copy: ...")
  foreach(_libdepend ${FINAL_LIST})
    message(STATUS "  final:${_libdepend}")
    file(COPY "${_libdepend}" DESTINATION "${_dir}")
  endforeach()

endforeach()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
