set(ARCANE_HAS_SWIG FALSE)

# Pour générer automatiquement le nom de bibliothèque compatible avec
# la plateforme cible
if (POLICY CMP0122)
  cmake_policy(SET CMP0122 NEW)
endif ()

# ----------------------------------------------------------------------------

# Pour Swig, il faut vérifier la taille du type C 'long' pour que le wrapping soit
# cohérent avec la définition dans arcane/utils/ArcaneGlobal.h
find_package(SWIG)
message(STATUS "SWIG_EXECUTABLE = ${SWIG_EXECUTABLE} (found=${SWIG_FOUND} version=${SWIG_VERSION})")
if (NOT SWIG_FOUND)
  message(STATUS "Disabling .NET wrapper because Swig is not available")
  return()
endif()
if (SWIG_VERSION VERSION_LESS 4)
  message(FATAL_ERROR "Found version (${SWIG_VERSION}) of Swig is too old. Version 4+ is required")
  return()
endif()

# Il faut mettre cette politique pour que la cible générée par les
# commandes 'swig' aient le bon nom (disponible à partir de CMake 3.13)
if(POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
else()
  set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
endif()
if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif()
set(UseSWIG_MODULE_VERSION 2)
include(UseSWIG)

set(ARCANE_HAS_SWIG TRUE)

# ----------------------------------------------------------------------------
# Il faut indiquer à swig les informations pour savoir à quels types
# du C++ classique (short, int, long, long long) correspondent
# les types 'int16_t', 'int32_t' et 'int64_t'.
# Sur les architectures utilisées par Arcane, on a toujours
# 'int16_t' équivalent à 'short' et 'int32_t' équivalent à 'int'.
# Par contre, pour 'int64_t', c'est soit 'long', soit 'long long'.
# La version 3 de swig permet de définir une macro SWIGWORDSIZE64
# pour indiquer qu'un 'long' est 64 bits.
# A priori sur Linux 64 bits, c'est toujours le cas (model LP64), et sous windows
# ce n'est jamais le cas (mode LLP64).
if (UNIX)
  set(SWIG_DEFINE64 "-DSWIGWORDSIZE64")
endif()

# ----------------------------------------------------------------------------

if(ARCANE_WANT_64BIT)
  set(SWIG_DEFINE64 "${SWIG_DEFINE64} -DARCANE_64BIT")
  set(ARCANE_DOTNET_64BIT_DEFINE "/define:ARCANE_64BIT")
endif()

# ----------------------------------------------------------------------------

set(ARCANE_DOTNET_WRAPPER_OUTDIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(ARCANE_DOTNET_WRAPPER_PUBLISH_RELATIVE_DIR "lib"
  CACHE STRING "Relative Path to publish .Net binaries for wrapper" FORCE)
set(ARCANE_DOTNET_WRAPPER_INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/${ARCANE_DOTNET_WRAPPER_PUBLISH_RELATIVE_DIR}
  CACHE FILEPATH "Directory for wrapper C# libraries" FORCE)
set(ARCANE_DOTNET_WRAPPER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ----------------------------------------------------------------------------
# Fonction pour compiler un projet C#.
#
# Les arguments sont:
#
#   TARGET_NAME: nom de la cible pour CMake. Cette fonction génère une
#                cible via add_custom_target qui aura pour nom ${TARGET_NAME}
#
#   PROJECT_NAME: nom du projet C#. Un fichier ${PROJECT_NAME}.csproj.in doit
#                 se trouver dans le répertoire courant. La commande
#                 CMake configure_file() est utilisée pour faire une copie
#                 de ce projet dans le répertoire de build.
#
#   CSHARP_SOURCES: liste des fichiers sources à ajouter au projet. Ces fichiers
#                 sont aussi ajoutés à la liste des dépendances de 'cmake' ce
#                 qui permet de recompiler automatiquement le projet si
#                 un de ces fichier a changé.
#
# Le projet doit générer une assembly de nom ${PROJECT_NAME}.dll (l'extension
# doit être '.dll' même s'il s'agit d'un exécutable au sens '.net').
#
function(arcane_wrapper_add_csharp_target)
  set(options        EXECUTABLE)
  set(oneValueArgs   TARGET_NAME PROJECT_NAME)
  set(multiValueArgs CSHARP_SOURCES DOTNET_TARGET_DEPENDS)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_TARGET_NAME)
    message(FATAL_ERROR "Missing argument TARGET_NAME")
  endif()
  if (NOT ARGS_PROJECT_NAME)
    message(FATAL_ERROR "Missing argument PROJECT_NAME")
  endif()

  message(STATUS "Add C# library target_name=${ARGS_TARGET_NAME} proj=${ARGS_PROJECT_NAME}")

  set(FULL_DLL_PATH ${ARCANE_DOTNET_WRAPPER_INSTALL_DIRECTORY}/${ARGS_PROJECT_NAME}.dll)

  # Génère une variable ARCANE_CSHARP_ITEM_GROUP_FILES contenant la liste des fichiers
  # à compiler. Cette variable sera utilisé dans le configure du projet
  if (ARGS_CSHARP_SOURCES)
    list(TRANSFORM ARGS_CSHARP_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/csharp/")
    list(TRANSFORM ARGS_CSHARP_SOURCES APPEND ".cs")
    set(_CSHARP_DEPENDS ${ARGS_CSHARP_SOURCES})

    list(TRANSFORM ARGS_CSHARP_SOURCES PREPEND "    <Compile Include = \"")
    list(TRANSFORM ARGS_CSHARP_SOURCES APPEND "\"/>")
    list(JOIN ARGS_CSHARP_SOURCES "\n" _OUT_CSHARP_TXT)
    message(VERBOSE "TRANSFORM_LIST=${_OUT_CSHARP_TXT}")
    set(ARCANE_CSHARP_ITEM_GROUP_FILES "\n  <!-- The following ItemGroup is generated -->\n  <ItemGroup>\n${_OUT_CSHARP_TXT}\n  </ItemGroup>")
  endif()

  configure_file(${ARGS_PROJECT_NAME}.csproj.in
    ${ARCANE_CSHARP_PROJECT_PATH}/${ARGS_PROJECT_NAME}/${ARGS_PROJECT_NAME}.csproj @ONLY)

  arcane_add_global_csharp_target(${ARGS_TARGET_NAME}
    BUILD_DIR ${ARCANE_DOTNET_WRAPPER_INSTALL_DIRECTORY}
    ASSEMBLY_NAME ${ARGS_PROJECT_NAME}.dll
    PROJECT_PATH ${ARCANE_CSHARP_PROJECT_PATH}/${ARGS_PROJECT_NAME}
    PROJECT_NAME ${ARGS_PROJECT_NAME}.csproj
    MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS}
    DEPENDS ${support_files} ${_CSHARP_DEPENDS}
    DOTNET_TARGET_DEPENDS ${ARGS_DOTNET_TARGET_DEPENDS}
    PACK
    )

  # Installe les dll générées dans lib
  install(FILES
    ${FULL_DLL_PATH}
    ${ARCANE_DOTNET_WRAPPER_INSTALL_DIRECTORY}/${ARGS_PROJECT_NAME}.pdb
    DESTINATION lib)

  # Avec 'coreclr', il faut aussi installer les fichiers de dépendances
  # pour le mode embarqué
  if (ARCANE_DOTNET_RUNTIME STREQUAL "coreclr")
    install(FILES
      ${ARCANE_DOTNET_WRAPPER_INSTALL_DIRECTORY}/${ARGS_PROJECT_NAME}.deps.json
      DESTINATION lib)
    if (ARGS_EXECUTABLE)
      install(FILES
        ${ARCANE_DOTNET_WRAPPER_INSTALL_DIRECTORY}/${ARGS_PROJECT_NAME}.runtimeconfig.json
        DESTINATION lib)
    endif()
  endif()
endfunction()

# ----------------------------------------------------------------------------
# Fonction pour générer les cibles et commandes pour wrapper du code C++ en C#.
#
# Il faut indiquer les options suivantes:
#
# NAME: nom du wrapping. Cela est utilisé pour générer une bibliothèque
#       de nom 'arcane_dotnet_wrapper_${NAME}'
# SOURCE: nom du fichier sources pour swig (extension .i)
# NAMESPACE_NAME: nom du namespace C# dans lequel les classes sont wrappées
# DLL_NAME: nom de la DLL généré et du projet '.csproj' associé.
# CSHARP_SOURCES: liste (éventuellement vide) de fichiers C# à compiler
#  en plus de ceux générés par swig.
#
# Pour la compilation du C#, il faut un projet dans ${DLL_NAME}.csproj.in dans
# le répertoire courant et ce projet est ensuite installé dans
# ${CMAKE_BINARY_DIR}/share/csproj. C'est dans ce répertoire qu'on lance
# ensuite la compilation du C#. Les 'dlls' générées sont installées
# dans ${ARCANE_DOTNET_WRAPPER_INSTALL_DIRECTORY}
#
# Il est aussi possible de spécifier des répertoires de recherche pour swig
# via l'argument INCLUDE_DIRECTORIES
macro(arcane_wrapper_add_swig_target)
  set(options        )
  set(oneValueArgs   NAME SOURCE NAMESPACE_NAME DLL_NAME)
  set(multiValueArgs INCLUDE_DIRECTORIES CSHARP_SOURCES SWIG_TARGET_DEPENDS)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set_property(SOURCE ${ARGS_SOURCE} PROPERTY CPLUSPLUS ON)
  set_property(SOURCE ${ARGS_SOURCE} PROPERTY USE_SWIG_DEPENDENCIES TRUE)
  set_property(SOURCE ${ARGS_SOURCE} PROPERTY INCLUDE_DIRECTORIES ${Arccore_SOURCE_DIR}/src/base ${ARCANE_SRC_PATH} ${ARCANE_DOTNET_WRAPPER_SOURCE_DIR} ${ARGS_INCLUDE_DIRECTORIES})
  if (UNIX)
    # TODO: regarder si on ne peut pas mettre ce define dans un .i
    set_property(SOURCE ${ARGS_SOURCE} PROPERTY COMPILE_DEFINITIONS "SWIGWORDSIZE64")
  endif()
  set_property(SOURCE ${ARGS_SOURCE} PROPERTY COMPILE_OPTIONS -namespace ${ARGS_NAMESPACE_NAME})
  set(_TARGET_NAME arcane_dotnet_wrapper_${ARGS_NAME})
  message(STATUS "Swig source (target=${ARGS_NAME}) list='${ARGS_SOURCE}' "
    "wrapper_source='${ARCANE_DOTNET_WRAPPER_SOURCE_DIR}' arcane_src_path='${ARCANE_SRC_PATH}'")
  # Commande swig pour générer le wrapping C++ et les fichiers C#
  swig_add_library(${_TARGET_NAME}
    TYPE SHARED
    LANGUAGE CSHARP
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_SOURCE}
    OUTPUT_DIR ${ARCANE_DOTNET_WRAPPER_OUTDIRECTORY}/out_cs_${ARGS_NAME}
    OUTFILE_DIR ${ARCANE_DOTNET_WRAPPER_OUTDIRECTORY}/out_cpp_${ARGS_NAME}
    )
  # Avec cmake (au moins avec la 3.14 et avant) le module 'UseSWIG' supprime
  # le préfix 'lib' ce qui empêche de trouver correctement la dll lors de l'exécution
  if (UNIX)
    set_target_properties(${_TARGET_NAME} PROPERTIES PREFIX "lib")
  endif()

  set_target_properties(${_TARGET_NAME}
    PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH 1
    INSTALL_RPATH "$ORIGIN"
    )
  arcane_target_set_standard_path(${_TARGET_NAME})

  # Récupère la liste des fichiers générés par Swig. On s'en sert pour faire une
  # dépendance dessus pour la compilation de la partie C#.
  get_property(support_files TARGET ${_TARGET_NAME} PROPERTY SWIG_SUPPORT_FILES)
  message(STATUS "Support files for wrapper '${ARGS_NAME}' (target='${_TARGET_NAME}') files='${support_files}'")

  message(STATUS "CSHARP_SOURCES=${ARGS_CSHARP_SOURCES}")

  set(_DOTNET_TARGET_DEPENDS)
  foreach(_dtg ${ARGS_SWIG_TARGET_DEPENDS})
    list(APPEND _DOTNET_TARGET_DEPENDS dotnet_wrapper_${_dtg})
  endforeach()

  arcane_wrapper_add_csharp_target(TARGET_NAME dotnet_wrapper_${ARGS_NAME}
    PROJECT_NAME ${ARGS_DLL_NAME}
    CSHARP_SOURCES ${ARGS_CSHARP_SOURCES}
    DOTNET_TARGET_DEPENDS ${_DOTNET_TARGET_DEPENDS}
  )

  # Indique que la compilation du C# dépend des fichiers générés par SWIG.
  # NOTE: il y a probablement un bug avec CMake 3.21 sur cette partie car il indique
  # qu'il ne sait pas comment trouver les fichiers générés. Cela est peut-être du
  # au module UseSwig associé à cette version.
  # NOTE: Il semble que cela ne fonctionne pas toujours correctement si on utilise 'Make'
  # (Il y a d'ailleurs un comportement spécifique du fichier 'UseSwig.cmake' dans ce cas)
  # On n'active donc pas cette gestion si on utilise 'Make'. Dans ce cas on ajoute
  # une dépendance sur la cible générée par SWIG mais cela oblige à recompiler si une
  # des dépendances est modifiée (par exemple un '.so')
  if (CMAKE_VERSION VERSION_LESS_EQUAL 3.21 OR CMAKE_GENERATOR MATCHES "Make")
    add_dependencies(dotnet_wrapper_${ARGS_NAME} ${_TARGET_NAME})
  else()
    message(STATUS "Adding custom target with support files for wrapper '${ARGS_NAME}'")
    add_custom_target(dotnet_wrapper_${ARGS_NAME}_swig_depend ALL DEPENDS "${support_files}")
    add_dependencies(dotnet_wrapper_${ARGS_NAME} dotnet_wrapper_${ARGS_NAME}_swig_depend)
  endif()

  # Ajoute les dépendences sur les autres cibles SWIG
  foreach(_dtg ${ARGS_SWIG_TARGET_DEPENDS})
    add_dependencies(dotnet_wrapper_${ARGS_NAME} arcane_dotnet_wrapper_${_dtg})
    target_link_libraries(${_TARGET_NAME} PUBLIC arcane_dotnet_wrapper_${_dtg})
  endforeach()

  # TODO: voir si on peut supprimer cet include et sinon voir s'il ne faut pas mettre public au lieu
  # de private.
  target_include_directories(${_TARGET_NAME} PRIVATE ${ARCANE_DOTNET_WRAPPER_SOURCE_DIR} ${ARGS_INCLUDE_DIRECTORIES})

  # Compile uniquement avec l'option '-O1' pour compiler plus vite.
  # La différence de performance n'est pas perceptible pour le wrapper.
  if (UNIX)
    target_compile_options(${_TARGET_NAME} PRIVATE "-O1")
  endif()

  arcane_register_library(${_TARGET_NAME} OPTIONAL)
endmacro()

# ----------------------------------------------------------------------------
# Fonction pour ajouter au wrapper *${name}* une dépendance sur le
# wrapper *${depend_name}*
# Obsolète: utiliser 'SWIG_TARGET_DEPENDS' de la fonction 'arcane_wrapper_add_swig_target'
function(arcane_wrapper_add_swig_dependency name depend_name)
  message(STATUS "Add swig dependencies for wrapper '${name}' depend=${depend_name}")
  add_dependencies(dotnet_wrapper_${name} dotnet_wrapper_${depend_name} arcane_dotnet_wrapper_${depend_name})
  target_link_libraries(arcane_dotnet_wrapper_${name} PUBLIC arcane_dotnet_wrapper_${depend_name})
endfunction()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
