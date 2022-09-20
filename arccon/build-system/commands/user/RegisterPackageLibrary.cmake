# ----------------------------------------------------------------------------
# Macro interne pour positionner la variable <pkg>_FOUND associée à un package et
# la mettre dans le cache
macro(arccon_internal_set_package_found package_name is_found)
  if (NOT package_name)
    message(FATAL_ERROR "Missing argument 'package_name'")
  endif()
  # Pour une variable donnée, CMake gère plusieurs instances qui n'ont pas forcément
  # la même valeur. Il faut donc toutes les changer en même temps pour conserver
  # la cohérence.
  set(${package_name}_FOUND ${is_found})
  set(${package_name}_FOUND ${is_found} CACHE BOOL "Is Package '${package_name}' Found" FORCE)
  set(${package_name}_FOUND ${is_found} PARENT_SCOPE)
endmacro()

# ----------------------------------------------------------------------------
# Macro interne pour positionner une liste de cibles pour un package donnée.
#
# NOTE: Cette macro est interne. Les versions publiques sont 'arccon_register_package_library',
# 'arccon_register_cmake_config_target' ou 'arccon_register_cmake_multiple_config_target'
#
# USAGE:
#
#    arccon_internal_set_package_name_and_targets(NAME package_name TARGETS target1 [target2] ...)
#
# Cette macro permet d'indiquer qu'un package est trouvé (met ${package_name}_FOUND à TRUE)
# et associer une liste de cibles à un package. La variable ARCCON_PACKAGE_${package_name}_TARGETS
# contiendra cette liste de cibles.

macro(arccon_internal_set_package_name_and_targets)

  set(options        )
  set(multiValueArgs TARGETS)
  set(oneValueArgs   NAME)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_NAME)
    message(FATAL_ERROR "Missing argument NAME")
  endif()

  if (NOT ARGS_TARGETS)
    message(FATAL_ERROR "Missing argument TARGETS")
  endif()

  set(package_name "${ARGS_NAME}")
  arccon_internal_set_package_found(${package_name} TRUE)
  set(ARCCON_PACKAGE_${package_name}_TARGETS ${ARGS_TARGETS} CACHE STRING "Targets for package ${package_name}" FORCE)
  message(VERBOSE "SetTargetsForPackage Package '${ARGS_NAME}' TARGETS=${ARGS_TARGETS}")

endmacro()

# ----------------------------------------------------------------------------
# Fonction pour créer une bibliothèque interface pour référencer facilement
# le package dans les dépendances. 'package_name' contient le nom
# de la bibliothèque interface et 'var_name' le prefix des variables
# contenant les répertoires d'include et les bibliothèques.
#
# Par convention la bibliothèque utilise le namespace 'arccon::${package_name}'.
#
# Par exemple, pour la glib il suffit de mettre la macro comme suit:
#
#   arccon_register_package_library(Glib GLIB)
#
# Si la variable ARCCON_EXPORT_TARGET est définie, alors la cible
# créée par cette fonction sera exportée dans ${ARCCON_EXPORT_TARGET}.
# Sinon, la cible créée est une cible importée classique.
#
# La cible n'est créé que si la variable ${package_name}_FOUND vaut 'TRUE'.
# Dans tous les cas, la valeur de la variable ${package_name}_FOUND est
# conservée dans le cache.
#
# TODO: pouvoir étendre cette fonction en spécifiant d'autres propriétés.
#
function(arccon_register_package_library package_name var_name)
  find_package_handle_standard_args(${package_name} DEFAULT_MSG
    ${var_name}_LIBRARIES ${var_name}_INCLUDE_DIRS)
  set(_FOUND FALSE)
  if (${package_name}_FOUND)
    set(_FOUND TRUE)
    set(_TARGET_NAME arcconpkg_${package_name})
    set(_ALIAS_NAME arccon::${package_name})
    if (DEFINED ARCCON_EXPORT_TARGET)
      # Il n'est pas possible avec CMake de créé une cible non importée
      # qui comporte un namespace. On utilise donc un nom intermédiaire.
      add_library(${_TARGET_NAME} INTERFACE)
      install(TARGETS ${_TARGET_NAME} EXPORT ${ARCCON_EXPORT_TARGET})
      add_library(${_ALIAS_NAME} ALIAS ${_TARGET_NAME})
      message(STATUS "Registering interface target '${_TARGET_NAME}' for package '${package_name}'")
    else()
      # Note: le code suivant ne fonctionne qu'à partir de CMake 3.11
      # car avant il n'est pas possible d'avoir un alias de cible importée.
      add_library(${_TARGET_NAME} INTERFACE IMPORTED GLOBAL)
      add_library(${_ALIAS_NAME} ALIAS arcconpkg_${package_name})
      message(STATUS "Registering imported target '${_TARGET_NAME}' for package '${package_name}'")
    endif()
    if (ARCCON_REGISTER_PACKAGE_VERSION STREQUAL 2)
      # Utilise les commandes CMake au lieu de positionner directement les
      # propriétés. Regarde aussi si '${var_name}_LIBRARIES' contient les mots
      # clés 'optimized', 'debug' ou 'general'. Dans ce cas, n'ajoute pas ce mot clé
      # et on ne garde que le mode 'optimized' ou 'general' mais pas le mode debug
      # (si on garde le mode debug, CMake utilise des 'generator expression' et ces
      # derniers ne sont pas toujours bien supportés par les makefiles.
      # Voir doc de target_link_libraries()) pour plus d'infos
      set(_ALL_LIBS "${${var_name}_LIBRARIES}")
      message(STATUS "Checking: LIBS=${_ALL_LIBS}")
      list(LENGTH _ALL_LIBS _ALL_LIBS_LENGTH)
      set(_NEXT_LIB_KEEP TRUE)
      foreach(_LIB_INDEX RANGE ${_ALL_LIBS_LENGTH})
        if (_LIB_INDEX EQUAL ${_ALL_LIBS_LENGTH})
          break()
        endif()
        list(GET _ALL_LIBS ${_LIB_INDEX} _CURRENT_LIB)
        if (_CURRENT_LIB STREQUAL "debug")
          set(_NEXT_LIB_KEEP FALSE)
          continue()
        elseif (_CURRENT_LIB STREQUAL "optimized")
          set(_NEXT_LIB_KEEP TRUE)
          continue()
        elseif (_CURRENT_LIB STREQUAL "general")
          set(_NEXT_LIB_KEEP TRUE)
          continue()
        endif()
        message(STATUS "CURRENT_LIB=${_CURRENT_LIB} config=${_NEXT_LIB_KEEP}")
        if (_NEXT_LIB_KEEP)
          target_link_libraries(${_TARGET_NAME} INTERFACE "${_CURRENT_LIB}")
        endif()
        set(_NEXT_LIB_KEEP TRUE)
      endforeach()
      #target_link_libraries(${_TARGET_NAME} INTERFACE "${${var_name}_LIBRARIES}")
      target_include_directories(${_TARGET_NAME} INTERFACE "${${var_name}_INCLUDE_DIRS}")
    else()
      set_target_properties(${_TARGET_NAME}
        PROPERTIES
        # Il est possible de spécfier que les fichiers des packages doivent
        # être considérés comme des fichiers systèmes ce qui permet
        # de ne pas avoir d'avertissements dessus. Pour cela il faut ajouter
        # la ligne suivante (mais laisser INTERFACE_INCLUDE_DIRECTORIES)
        #   INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${${var_name}_INCLUDE_DIRS}"
        INTERFACE_INCLUDE_DIRECTORIES "${${var_name}_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${${var_name}_LIBRARIES}"
        )
    endif()
    arccon_internal_set_package_name_and_targets(NAME ${package_name} TARGETS ${_TARGET_NAME})
  else()
    # Le package n'est pas trouvé.
    arccon_internal_set_package_found(${package_name} FALSE)
  endif()

  # Pour compatibilité avec l'existant, définit une variable
  # ${package_name}_FOUND avec ${package_name} le nom du package en majuscule.
  string(TOUPPER ${package_name} _UPPER_TARGET_NAME)
  set(${_UPPER_TARGET_NAME}_FOUND ${_FOUND} CACHE BOOL "(COMPAT) Is Package '${package_name}' Found" FORCE)
endfunction()

# ----------------------------------------------------------------------------
# Fonction pour associer une cible importée à un package.
#
# Usage:
#
#  arccon_register_cmake_config_target(package_name CONFIG_TARGET_NAME target_name)
#
# NOTE: comme on utilise des cibles importées, il faut ajouter
# les find_dependency qui vont bien dans le fichier de *Config.cmake
#
# NOTE: L'option 'PACKAGE_NAME' n'est plus utilisée.
#
function(arccon_register_cmake_config_target package_name)

  set(options        )
  set(oneValueArgs   CONFIG_TARGET_NAME PACKAGE_NAME)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_CONFIG_TARGET_NAME)
    message(FATAL_ERROR "Missing argument CONFIG_TARGET_NAME")
  endif()

  set(_TARGET_NAME ${ARGS_CONFIG_TARGET_NAME})
  set(_ALIAS_NAME arccon::${package_name})
  # ATTENTION: Cet alias ne fonctionne que dans le 'CMakeLists.txt' appelant cette
  # fonction ou un sous-répertoire. Pour cette raison son usage est obsolète et sera
  # supprimé dans une version ultérieure de Arccon
  add_library(${_ALIAS_NAME} ALIAS ${_TARGET_NAME})

  message(STATUS "Registering CMake imported target '${_TARGET_NAME}' as target '${_ALIAS_NAME}'")
  arccon_internal_set_package_name_and_targets(NAME ${package_name} TARGETS ${_TARGET_NAME})
endfunction()

# ----------------------------------------------------------------------------
# Fonction pour associer une ou plusieurs cible importées à un package.
#
# Usage:
#
#  arccon_register_cmake_multiple_config_target(package_name CONFIG_TARGET_NAMES target_name1 [target_name2] ...)
#
function(arccon_register_cmake_multiple_config_target package_name)

  set(options        )
  set(multiValueArgs   CONFIG_TARGET_NAMES)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_CONFIG_TARGET_NAMES)
    message(FATAL_ERROR "Missing argument CONFIG_TARGET_NAMES")
  endif()

  message(STATUS "Registering CMake imported targets '${ARGS_CONFIG_TARGET_NAMES}' as package '${package_name}'")

  arccon_internal_set_package_name_and_targets(NAME ${package_name} TARGETS ${ARGS_CONFIG_TARGET_NAMES})
endfunction()

# ----------------------------------------------------------------------------
# Macro permettant de ne pas lire le reste du fichier si un package référencé
# par 'Arccon' existe déjà.
#
#   arccon_return_if_package_found(package_name)
#
# Cette macro recherche si une cible arcconpkg_${package_name} existe déjà
# et appelle return() si c'est le cas.
macro(arccon_return_if_package_found package_name)
  if (TARGET arcconpkg_${package_name})
    return()
  endif()
  if (TARGET arccon::${package_name})
    return()
  endif()
endmacro()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
