# ----------------------------------------------------------------------------
# Fonction pour créer une bibliothèque interface pour référencer facilement
# le package dans les dépendances. 'lib_name' contiend le nom
# de la bibliothèque interface et 'var_name' le prefix des variables
# contenant les répertoires d'include et les bibliothèques.
#
# Par convention la bibliothèque utilise le namespace 'arccon::${lib_name}'.
#
# Par exemple, pour la glib il suffit de mettre la macro comme suit:
#
#   arccon_register_package_library(Glib GLIB)
#
# Si la variable ARCCON_EXPORT_TARGET est définie, alors la cible
# créée par cette fonction sera exportée dans ${ARCCON_EXPORT_TARGET}.
# Sinon, la cible créée est une cible importée classique.
#
# La cible n'est créé que si la variable ${lib_name}_FOUND vaut 'TRUE'.
# Dans tous les cas, la valeur de la variable ${lib_name}_FOUND est
# conservée dans le cache.
#
# TODO: pouvoir étendre cette fonction en spécifiant d'autres propriétés.
#
function(arccon_register_package_library lib_name var_name)
  find_package_handle_standard_args(${lib_name} DEFAULT_MSG
          ${var_name}_LIBRARIES ${var_name}_INCLUDE_DIRS)
  set(_FOUND FALSE)
  if (${lib_name}_FOUND)
    set(_FOUND TRUE)
    set(_TARGET_NAME arcconpkg_${lib_name})
    set(_ALIAS_NAME arccon::${lib_name})
    if (DEFINED ARCCON_EXPORT_TARGET)
      # Il n'est pas possible avec CMake de créé une cible non importée
      # qui comporte un namespace. On utilise donc un nom intermédiaire.
      #set(_TARGET_NAME arccon::${lib_name})
      #set(_ALIAS_NAME arcconpkg_${lib_name})
      add_library(${_TARGET_NAME} INTERFACE)
      install(TARGETS ${_TARGET_NAME} EXPORT ${ARCCON_EXPORT_TARGET})
      add_library(${_ALIAS_NAME} ALIAS ${_TARGET_NAME})
      message(STATUS "Registering interface target '${_TARGET_NAME}'")
    else()
      # Note: le code suivant ne fonctionne qu'à partir de CMake 3.11
      # car avant il n'est pas possible d'avoir un alias de cible importée.
      add_library(${_TARGET_NAME} INTERFACE IMPORTED GLOBAL)
      add_library(${_ALIAS_NAME} ALIAS arcconpkg_${lib_name})
      message(STATUS "Registering imported target '${_TARGET_NAME}'")
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
  endif()
  # Force _FOUND à être dans le cache pour qu'on puisse récupérer la liste
  # des packages la propriété PACKAGES_FOUND.
  set(${lib_name}_FOUND ${_FOUND} CACHE BOOL "Is Package '${lib_name}' Found" FORCE)

  # Pour compatibilité avec l'existant, définit une variable
  # ${X}_FOUND avec ${X} le nom du package en majuscule.
  string(TOUPPER ${lib_name} _UPPER_TARGET_NAME)
  set(${_UPPER_TARGET_NAME}_FOUND ${_FOUND} CACHE BOOL "(COMPAT) Is Package '${lib_name}' Found" FORCE)
endfunction()

# ----------------------------------------------------------------------------
# Macro permettant de ne pas lire le reste du fichier si un package référencé
# par 'Arccon' existe déjà.
#
#   arccon_register_package_library(pkg_name)
#
# Cette macro recherche si une cible arcconpkg_${pkg_name} existe déjà et
# appelle return() si c'est le cas.
macro(arccon_return_if_package_found lib_name)
  if (TARGET arcconpkg_${lib_name})
    return()
  endif()
endmacro()

# ----------------------------------------------------------------------------
