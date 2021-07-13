# Fonction pour installer un répertoire en positionnant les droits
# correctement pour les groupes afin que les fichiers créés soient
# dans le même groupe que le répertoire parent
# Cela est nécessaire pour les systèmes de fichiers où les quotas
# existent
# Usage:
#
#  arccon_install_directory(NAMES dirname ... DESTINATION destination [PATTERN pattern ...])
#
function(arccon_install_directory)
  set(_func_name "arccon_install_directory")
  set(oneValueArgs NAME DESTINATION)
  set(multiValueArgs NAMES PATTERN)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "In ${_func_name}: unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
  if (NOT ARGS_NAMES)
    message(FATAL_ERROR "In ${_func_name}: missing 'NAME' argument")
  endif()
  if (NOT ARGS_DESTINATION)
    message(FATAL_ERROR "In ${_func_name}: missing 'DESTINATION' argument")
  endif()
  message(STATUS "INSTALL_DIR name=${ARGS_NAMES} dest=${ARGS_DESTINATION} pattern=${ARGS_PATTERN}")
  install(DIRECTORY ${ARGS_NAMES}
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE SETGID
    USE_SOURCE_PERMISSIONS
    DESTINATION ${ARGS_DESTINATION}
    PATTERN ${ARGS_PATTERN})
endfunction()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
