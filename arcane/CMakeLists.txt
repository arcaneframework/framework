cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

message(STATUS "CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR} CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}  CMAKE_VERSION=${CMAKE_VERSION}")

# ----------------------------------------------------------------------------
# Vérifie qu'on a bien lancé la configuration depuis le répertoire racine
# du dépot.
cmake_path(COMPARE "${CMAKE_SOURCE_DIR}" EQUAL "${CMAKE_CURRENT_SOURCE_DIR}" IS_NOT_FROM_BASE_REPO)
message(STATUS "IS_NOT_FROM_BASE_REPO=${IS_NOT_FROM_BASE_REPO}")
if (IS_NOT_FROM_BASE_REPO)
  cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH PARENT_SOURCE_DIR)
  message(FATAL_ERROR
      "Compiling in 'arcane' subdirectory is not supported."
      " Use CMakeLists.txt from repository base directory: '${PARENT_SOURCE_DIR}'"
    )
endif()

# ----------------------------------------------------------------------------
# Récupère numéro de version de Arcane à partir du fichier 'version' et
# génère un fichier 'arcane_version.h' contenant ces informations.

file(READ "version" ARCANE_VERSION_STR_FULL)
string(REPLACE "_dev" "" ARCANE_VERSION ${ARCANE_VERSION_STR_FULL})
message(STATUS "ArcaneVersion = ${ARCANE_VERSION}")

project(Arcane VERSION ${ARCANE_VERSION} LANGUAGES C CXX)

math(EXPR ARCANE_VERSION_NUMERIC "((10000 * ${Arcane_VERSION_MAJOR}) + 100 * ${Arcane_VERSION_MINOR}) + ${Arcane_VERSION_PATCH}")
configure_file(arcane_version.h.in ${CMAKE_BINARY_DIR}/arcane_version.h @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/arcane_version.h DESTINATION include)

# ----------------------------------------------------------------------------

if (NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Arcane has to be compiled with BUILD_SHARED_LIBS=TRUE. Static linking does not work")
endif()

# ----------------------------------------------------------------------------

# A partir de 3.19:
# find_program() requires permission to execute but not to read.
if(POLICY CMP0109)
  cmake_policy(SET CMP0109 NEW)
endif()

# A partir de 3.19:
# An imported target missing its location property fails during generation.
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

# A partir de 3.27:
# find_package() uses upper-case <PACKAGENAME>_ROOT variables.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Positionne CMAKE_BUILD_TYPE si ARCANE_BUILD_TYPE est défini ou
# ARCANE_BUILD_TYPE en fonction de CMAKE_BUILD_TYPE dans le cas contraire
# TODO: utiliser CMAKE_CONFIGURATION_TYPE pour modifier les valeurs possibles
# CMAKE_BUILD_TYPE et ainsi les faire correspondre à celles de Arcane.

# NOTE: a terme, ARCANE_BUILD_TYPE ne sera plus utilisé et seul
# ARCCORE_BUILD_MODE restera

# Si 'ARCCORE_BUILD_MODE' est défini, on s'en sert pour positionner
# ARCANE_BUILD_TYPE même si ce dernier est défini.

if (DEFINED ARCCORE_BUILD_MODE)
  set(ARCANE_BUILD_TYPE ${ARCCORE_BUILD_MODE})
endif()

if (NOT DEFINED ARCANE_BUILD_TYPE)
  if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  if (CMAKE_BUILD_TYPE STREQUAL Debug)
    set(ARCANE_BUILD_TYPE Debug)
  else()
    set(ARCANE_BUILD_TYPE Release)
  endif()
else()
  if (ARCANE_BUILD_TYPE STREQUAL Debug)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Debug)
    endif()
    set(ARCANE_WANT_CHECK TRUE)
  endif()

  if (ARCANE_BUILD_TYPE STREQUAL Check)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release)
    endif()
    set(ARCANE_WANT_CHECK TRUE)
  endif()

  if (ARCANE_BUILD_TYPE STREQUAL Release)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release)
    endif()
  endif()
endif()

if (NOT DEFINED ARCCORE_BUILD_MODE)
  set(ARCCORE_BUILD_MODE ${ARCANE_BUILD_TYPE})
endif()

#set(ARCCORE_BUILD_MODE ${ARCCORE_BUILD_MODE} CACHE STRING "Arccore build mode (debug,check,release)" FORCE)
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "The type of build, options are: None Debug Release RelWithDebInfo" FORCE)

message(STATUS "ARCANE_BUILD_TYPE = ${ARCANE_BUILD_TYPE}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Si CMAKE_BUILD_TYPE existe, vérifie qu'il vaut une valeur pré-définie
# Cela est nécessaire sinon certaines propriétés peuvent ne pas être valides,
# ce qui peut poser problèmes lors de la configuration (par exemple les chemins
# d'installation.

if (DEFINED CMAKE_BUILD_TYPE)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" LOWER_CMAKE_BUILD_TYPE)
  set (VALID_BUILD_TYPE_LIST "release;relwithdebinfo;debug;minsizerelease")
  if(NOT (LOWER_CMAKE_BUILD_TYPE IN_LIST VALID_BUILD_TYPE_LIST))
    message(FATAL_ERROR "Invalid value '${CMAKE_BUILD_TYPE}' for CMAKE_BUILD_TYPE."
      " Case insensitive valid values are '${VALID_BUILD_TYPE_LIST}'")
  endif()
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

SET(ARCANEBUILDROOT ${CMAKE_BINARY_DIR})
SET(ARCANESRCROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Il ne faut pas compiler la ou se trouve les sources
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "You can not do in-source compilation. You have to build in a directory distinct from the source directory")
endif()

if(VERBOSE)
  message(STATUS "Verbose mode enabled")
else()
  message(STATUS "Verbose mode disabled")
endif()

# Indique qu'on souhaite exporter dans 'ArcaneTargets' les cibles des packages
# trouvés par 'Arccon'.
option(ARCANE_WANT_ARCCON_EXPORT_TARGET "Export arccon targets in arcane." OFF)
option(ARCANE_USE_FIND_DEPENDENCIES "Use find_dependency() in ArcaneConfig.cmake." ON)

# Si on n'utilise pas find_dependency, il faut alors forcément exporter les packages
# dans 'ArcaneTargets'
if (NOT ARCANE_USE_FIND_DEPENDENCIES)
  set(ARCANE_WANT_ARCCON_EXPORT_TARGET ON)
endif()

if(ARCANE_WANT_ARCCON_EXPORT_TARGET)
  if(NOT FRAMEWORK_NO_EXPORT_PACKAGES)
    set(ARCCON_EXPORT_TARGET ArcaneTargets)
  endif()
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

option(ARCANE_ENABLE_TESTS "Enable compilation of tests" ON)
option(ARCANE_ENABLE_DOTNET_WRAPPER "Enable compilation of '.Net' wrapper" ON)
option(ARCANE_ENABLE_ALEPH "Enable the compilation of Aleph component" ON)

# Expérimental. Ne pas utiliser en dehors d'Arcane
option(ARCANE_ENABLE_DOTNET_PYTHON_WRAPPER "Enable compilation python extension using '.Net' wrapper" OFF)
if(ARCANE_ENABLE_DOTNET_PYTHON_WRAPPER)
  set(ARCANE_ENABLE_DOTNET_WRAPPER ON)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Cherche 'Arccon'

find_package(Arccon REQUIRED)
set(ARCANE_Arccon_NEEDED_VERSION "1.5.0")
message(STATUS "Arccon_VERSION = ${Arccon_VERSION}")
if (Arccon_VERSION VERSION_LESS ARCANE_AXLSTAR_NEEDED_VERSION)
  message(FATAL_ERROR "Arccon version '${Arccon_VERSION}' is too old."
    " Version ${ARCANE_Arccon_NEEDED_VERSION} required")
endif()

list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})
include(LoadBuildSystem)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Cherche 'ArcDependencies'

find_package(ArcDependencies REQUIRED)
set(ARCANE_ArcDependencies_NEEDED_VERSION "1.8.0")
message(STATUS "ArcDependencies_VERSION = ${ArcDependencies_VERSION}")
if (ArcDependencies_VERSION VERSION_LESS ARCANE_ArcDependencies_NEEDED_VERSION)
  message(FATAL_ERROR "ArcDependencies version '${ArcDependencies_VERSION}' is too old."
    " Version ${ARCANE_ArcDependencies_NEEDED_VERSION} required."
    " If you are compiling all the framework, you need to run the following command"
    " to update 'dependencies' directory:"
    "\n   cd ${CMAKE_CURRENT_SOURCE_DIR}"
    "\n   git submodule update"
    )
endif()

# ----------------------------------------------------------------------------
# Positionne les répertoires standards d'installation

include(ArcconSetInstallDirs)

set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(generated_config_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ARCANE_CMAKE_CONFIG_DIR ${config_install_dir})

# ----------------------------------------------------------------------------

set(ARCANE_AXLSTAR_NEEDED_VERSION "2.2.0")
if (NOT Axlstar_FOUND)
  find_package(Axlstar REQUIRED)
endif()
message(STATUS "Axlstar_DIR = ${Axlstar_DIR}")
message(STATUS "Axlstar_VERSION = ${Axlstar_VERSION}")
if (Axlstar_VERSION VERSION_LESS ARCANE_AXLSTAR_NEEDED_VERSION)
  message(FATAL_ERROR "Axlstar version '${Axlstar_VERSION}' is too old."
    " Version ${ARCANE_AXLSTAR_NEEDED_VERSION} required")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Autorise les tests
enable_testing()

# autorise d'ajouter une cible à une autre définie dans un autre répertoire
if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Inclut un fichier de Package si demandé.
# Il faut le faire avant de charger 'Arccon' ou 'Arccore' pour pouvoir
# prendre en compte les chemins les éventuels ajouts aux variables
# CMake telles que CMAKE_PREFIX_PATH.

# Si la variable d'environnement 'ARCANE_PACKAGE_FILE' est définie, considère que
# sa valeur indique un fichier cmake qu'il faut inclure
set(ARCANE_PACKAGE_FILE $ENV{ARCANE_PACKAGE_FILE})
message(STATUS "ARCANE_PACKAGE_FILE VALUE IS '${ARCANE_PACKAGE_FILE}'")
if(ARCANE_PACKAGE_FILE)
  if(EXISTS ${ARCANE_PACKAGE_FILE})
    message(STATUS "Adding ARCANE_PACKAGE_FILE named '${ARCANE_PACKAGE_FILE}'")
    include(${ARCANE_PACKAGE_FILE})
  else()
    MESSAGE(FATAL_ERROR "File '${ARCANE_PACKAGE_FILE}' is missing")
  endif()
endif()

# ----------------------------------------------------------------------------
# Ajoute 'cmake/Modules' à la CMAKE_MODULE_PATH pour la recherche de packages

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

# ----------------------------------------------------------------------------
# Affiche les infos sur le compilateur
message(STATUS "CXX Compiler             = ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX CompilerId           = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler Version     = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_FLAGS          = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG    = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE  = ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS   = ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR   = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_VERSION     = ${CMAKE_SYSTEM_VERSION}")

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Définit une cible fictive 'arcane_export_compile_flags' contenant les options du
# compilateur pour les bibliothèques et exécutables liés à Arcane et qui
# doivent être utilisés par le code utilisant Arcane. Cette cible contient par
# exemple les options pour spécifier la version du C++, la vectorisation.
# les avertissements de compilations.1

# Définit une cible fictive 'arcane_build_compile_flags' similaire à
# 'arcane_export_compile_flags' mais qui contient les options de compilation internes
# à la compilation des bibliothèques générées par Arcane. Ce sont par exemple
# les avertissements de compilation ou les options sur la visibilité des symboles.

# Ces deux cibles sont utilisées par 'arcane_utils' et donc toute bibliothèque
# dépendant de cette cible les utiliseront aussi.

add_library(arcane_export_compile_flags INTERFACE)
install(TARGETS arcane_export_compile_flags EXPORT ArcaneTargets)
add_library(arcane_build_compile_flags INTERFACE)
install(TARGETS arcane_build_compile_flags EXPORT ArcaneTargets)

# ----------------------------------------------------------------------------

# NOTE: il est important de spécifier NO_SYSTEM_FROM_IMPORTED
# pour que les .h des cibles importées ne soient pas inclus avec -isystem.
# En effet, avec par exemple le compilateur GCC, les chemins spécifiés
# par -isystem sont pris en compte après les valeurs de la variable
# d'environnement CPATH ce qui peut poser problème si cette dernière est
# positionnée dans l'environnement de l'utilisateur car alors on
# n'utilise pas forcément le bon package.
# TODO: il faudrait voir comment corriger ce problème notamment car un des avantages
# de l'option '-isystem' est que cela enlève les avertissements de compilation
# dans les fichiers des répertoires spécifiés par cette option.

set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)

# ----------------------------------------------------------------------------
# Il ne faut surtout pas inclure le répertoire courant automatiquement
# car cela pose des problèmes avec certains fichiers tels que 'String.h' qui
# peut être inclus à la place de 'string.h' sous windows par exemple.

set(CMAKE_INCLUDE_CURRENT_DIR FALSE)

# ----------------------------------------------------------------------------

# Note: depuis mai 2017, les threads sont toujours actifs
set(ARCANE_CONFIGURE_WANT_THREAD TRUE)

set(ARCANE_CONFIGURE_WANT_MPC $ENV{ARCANE_CONFIGURE_WANT_MPC})
message(STATUS "MPC config: ${ARCANE_CONFIGURE_WANT_MPC}")
if (ARCANE_CONFIGURE_WANT_MPC)
  set(ARCANE_USE_MPC TRUE CACHE BOOL "true if MPC is enabled" FORCE)
  message(STATUS "configure with MPC")
endif (ARCANE_CONFIGURE_WANT_MPC)

set(ARCANE_USE_THREAD TRUE CACHE BOOL "true if threads are enabled" FORCE)
message(STATUS "Threads are always enabled")

# ----------------------------------------------------------------------------

# Si on utilise gcc, supprime les extensions pour forcer le '-std=c++11'
# car sinon cmake va ajouter lui même l'option et dans ce cas il met '-std=gnu++11'.
# Cela pose problème car ce mode gnu++11 n'est pas strictement équivalent à c++11,
# notamment par exemple pour le noexcept(true) qui est valide en c++11 et pas
# en gnu++11
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_EXTENSIONS FALSE)
endif ()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Ajoute les infos de Arccore

if (ARCANE_WANT_CHECK)
  set(ARCCORE_CHECK TRUE)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Charge 'Arccore' depuis le répertoire de base.
if (NOT Arccore_FOUND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../arccore arccore)
  set(Arccore_FOUND YES)
  set(Arccore_FOUND YES PARENT_SCOPE)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "[arcane] Is Check mode enabled ? ${ARCANE_WANT_CHECK} (from arccore: ${ARCCORE_CHECK})")
message(STATUS "[arcane] Is Debug mode enabled ? ${ARCANE_WANT_DEBUG} (from arccore: ${ARCCORE_DEBUG})")

# ----------------------------------------------------------------------------

# Maintenant, on considère qu'on a toujours le support du C++20
set(ARCANE_USE_CXX11 TRUE CACHE BOOL "true if C++11 is supported and used" FORCE)
set(ARCANE_USE_CXX14 TRUE CACHE BOOL "true if C++14 is supported and used" FORCE)
set(ARCANE_HAS_CXX20 TRUE CACHE BOOL "True if C++20 is used (should be always true)" FORCE)
set(ARCANE_HAS_ACCELERATOR_API TRUE CACHE BOOL "True Arcane has accelerator API" FORCE)

set(ARCANE_HAS_CXX23 FALSE)
if (${ARCCORE_CXX_STANDARD} VERSION_GREATER_EQUAL 23)
  set(ARCANE_HAS_CXX23 TRUE)
endif()
set(ARCANE_HAS_CXX23 ${ARCANE_HAS_CXX23} CACHE BOOL "True if C++23 is used" FORCE)

# ----------------------------------------------------------------------------
# Utilise les mêmes options de compilation que 'Arccore'.

target_link_libraries(arcane_build_compile_flags INTERFACE Arccore::arccore_build_compile_flags)

# ----------------------------------------------------------------------------
# Regarde la taille des différents types (short,int,long,long long) du C++

set(ARCANE_TYPE_INT16 ${ARCCORE_TYPE_INT16})
set(ARCANE_TYPE_INT32 ${ARCCORE_TYPE_INT32})
set(ARCANE_TYPE_INT64 ${ARCCORE_TYPE_INT64})
set(ARCANE_INT32_MAX ${ARCCORE_INT32_MAX})
set(ARCANE_INT64_MAX ${ARCCORE_INT64_MAX})

# ----------------------------------------------------------------------------

include(${CMAKE_CURRENT_LIST_DIR}/cmake/ArcanePackageInfo.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Functions.cmake)

# ----------------------------------------------------------------------------

set(ARCANE_COMPATIBILITY_DIRECTORY "${CMAKE_BINARY_DIR}/compatibility/include"
  CACHE STRING "Path to generated headers for compatibilit" FORCE)
file(MAKE_DIRECTORY "${ARCANE_COMPATIBILITY_DIRECTORY}")

# ----------------------------------------------------------------------------

set(ARCANE_WANT_GEOMETRY TRUE)

message(STATUS "Using AVX Simd instructions ? -> ${ARCANE_WANT_AVX}")
message(STATUS "Using AVX512 Simd instructions ? -> ${ARCANE_WANT_AVX512}")

# Utilisation de purify.
if($ENV{ARCANE_WANT_PURIFY})
  MESSAGE(STATUS "Activating Purify")
  set(ARCANE_PURIFY_OPTIONS "-best-effort")
  SET(CMAKE_CXX_LINK_EXECUTABLE "purify ${ARCANE_PURIFY_OPTIONS} ${CMAKE_CXX_LINK_EXECUTABLE}")
  SET(CMAKE_C_LINK_EXECUTABLE "purify ${ARCANE_PURIFY_OPTIONS} ${CMAKE_C_LINK_EXECUTABLE}")
endif()

MESSAGE(STATUS "Installation dir is ${CMAKE_INSTALL_PREFIX}")
MESSAGE(STATUS "ARCANEBUILDROOT dir is ${ARCANEBUILDROOT}")

# Il faut positionner ces variables avant d'inclure le fichiers 'srcs.cmake.txt'
# car il les utilise
SET(ARCANE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
SET(ARCANE_INSTALL_SHR ${CMAKE_INSTALL_PREFIX}/share)
SET(ARCANE_INSTALL_LIB ${CMAKE_INSTALL_PREFIX}/lib)
SET(ARCANE_INSTALL_BIN ${CMAKE_INSTALL_PREFIX}/bin)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/share)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# permet d'ajouter des services se trouvant dans des lib compilées en dehors des sources d'Arcane
if(ARCANE_WANT_EXTRA_LIBS)
  list(INSERT ARCANE_EXTERNAL_LIBRARIES 0 $ENV{ARCANE_EXTRA_LIBS})
  message(STATUS     "ARCANE_EXTERNAL_LIBRARIES IS '${ARCANE_EXTERNAL_LIBRARIES}'")
endif()

if(NOT CMAKE_FIND_DIR)
  set(CMAKE_FIND_DIR "CMake")
endif()

# Ajoute à CMAKE_MODULE_PATH nos chemins contenant les scripts de recherche de CMake.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_FIND_DIR})

# ----------------------------------------------------------------------------
# OBSOLÈTE (09/2022): Utiliser arccon_register_package_library() à la place
#
# Macro pour créer une bibliothèque interface pour référencer facilement
# le package dans les dépendances. 'lib_name' contiend le nom
# de la bibliothèque interface et 'var_name' le prefix des variables
# contenant les répertoires d'include et les bibliothèques.
# Par convention la bibliothèque utilise le namespace 'arcane'.
# Par exemple, pour la glib il suffit de mettre la macro comme suit:
#   arcane_add_package_library(glib GLIB)
macro(arcane_add_package_library lib_name var_name)
  add_library(arcanepkg_${lib_name} INTERFACE)
  set_target_properties(arcanepkg_${lib_name}
    PROPERTIES
    # Il est possible de spécfier que les fichiers des packages doivent
    # être considérés comme des fichiers systèmes ce qui permet
    # de ne pas avoir d'avertissements dessus. Pour cela il faut ajouter
    # la ligne suivante (mais laisser INTERFACE_INCLUDE_DIRECTORIES)
    #   INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${${var_name}_INCLUDE_DIRS}"
    INTERFACE_INCLUDE_DIRECTORIES "${${var_name}_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${${var_name}_LIBRARIES}"
    )
  install(TARGETS arcanepkg_${lib_name} EXPORT ArcaneTargets)
  add_library(arcane::${lib_name} ALIAS arcanepkg_${lib_name})
  add_library(arccon::${lib_name} ALIAS arcanepkg_${lib_name})
  set(${lib_name}_FOUND TRUE CACHE BOOL "Is Package '${lib_name}' Found" FORCE)
endmacro()

# ----------------------------------------------------------------------------
# Créé une bibliothèque interface qui s'appelle 'arcane_full' et qui contient
# la liste des bibliothèques que doit utiliser un code utilisant Arcane.
# La macro 'arcane_register_library' permet d'enregistrer des
# cibles à 'arcane_full'.
add_library(arcane_full INTERFACE)
install(TARGETS arcane_full EXPORT ArcaneTargets)

# Ceci est nécessaire avec certaines versions de Linux (ubuntu 17.04 par exemple)
# qui n'ajoutent pas une bibliothèque si aucun symbole de cette
# bibliothèque n'est explictement utilisé (ce qui est souvent le cas avec les
# mécanismes de service)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # TODO: Vérifier que tous les compilateurs Linux (GCC,CLANG,Intel) supportent
  # cette option.
  message(STATUS "Adding 'no-as-needed' to link option for 'arcane_full'")
  target_link_options(arcane_full INTERFACE "-Wl,--no-as-needed")
  # Normalement ce n'est pas nécessaire par défaut mais cela peut permettre de
  # contourner des problèmes de RPATH ou RUNPATH pour les bibliothèques
  # pas directement utilisées par Arcane pour lequel le RPATH ou le RUNPATH
  # n'est pas bien géré (cela peut-être du à un bug ou un fichier
  # de configuration qui n'utilise pas cela)
  if (ARCANE_FORCE_NOASNEEDED)
    message(STATUS "Adding 'no-as-needed' to link option for 'arcane_build_compile_flags'")
    target_link_options(arcane_build_compile_flags INTERFACE "-Wl,--no-as-needed")
  endif()
endif()

# ----------------------------------------------------------------------------
# Liste des toutes les cibles bibliothèques définies par Arcane.
# On garde cette liste pour des raisons de compatibilité mais par la suite
# il faudra la supprimer et n'utiliser que des cibles interfaces.
set(ARCANE_LIBRARIES)

# Liste des cibles exportées par Arcane.
# Cette liste permet de faire automatiquement dans le fichier de configuration
# un Alias entre les noms sans namespace et avec namespace
# (par exemple 'arcane_utils' et 'Arcane::arcane_utils')
set(ARCANE_EXPORTED_TARGETS "" CACHE STRING "List of exported targets" FORCE)

set(ARCANE_PUBLIC_FOUND_PACKAGES)
set(ARCANE_PUBLIC_FOUND_PACKAGES "" CACHE STRING "List of public external packages founds" FORCE)

# ----------------------------------------------------------------------------
# Ajoute à ARCANE_LIBRARIES les cibles issues de 'Arccore' pour qu'elles soient
# dans le pkglist.xml
# TODO: regarder si cela est toujours utile
set(ARCCORE_ALLOWED_TARGETS base collections trace concurrency messagepassing messagepassingmpi)
foreach(pkg ${ARCCORE_ALLOWED_TARGETS})
  if(TARGET Arccore::arccore_${pkg})
    list(APPEND ARCANE_LIBRARIES Arccore::arccore_${pkg})
  endif()
endforeach()

# ----------------------------------------------------------------------------
# Recherche les packages disponibles
message(STATUS "Infos from package finders")
message(STATUS "----------------------------------------------------")
include(FindPkgConfig)
find_package(Glib REQUIRED)

# ----------------------------------------------------------------------------
# Choix du lecteur XML.
# Le seul lecteur maintenant disponible est 'LibXml2'. L'ancien lecteur
# basé sur XercesC n'existe plus. On affiche une erreur si ce dernier est
# demandé.
if (ARCANE_WANT_XERCESC)
  message(FATAL_ERROR "Option 'ARCANE_WANT_XERCESC' is no longer available.")
endif()

set(ARCANE_XMLREADER_PACKAGE "LibXml2" CACHE STRING "Package to handle XML files" FORCE)
message(STATUS "XMLReader package = ${ARCANE_XMLREADER_PACKAGE}")
find_package(${ARCANE_XMLREADER_PACKAGE} REQUIRED)

# ----------------------------------------------------------------------------
# Ne recherche pas MPI si on le demande explicitement
# TODO: supprimer cela et faire le find_package uniquement dans les
# composantes qui en ont besoin.
if (NOT ARCANE_WANT_NOMPI)
  arcane_find_package(MPI)
endif()
arcane_find_package(Valgrind)

# Définir la variable PROTECTION comme valeur par défaut : FLEXLM
set(PROTECTION "FLEXLM" CACHE STRING "Specify the license system (FLEXLM or RLM)")

# Vérifiez la valeur de PROTECTION
if(PROTECTION STREQUAL "FLEXLM")
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/FindFlexLM.cmake)
elseif(PROTECTION STREQUAL "RLM")
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/FindRLM.cmake)
else()
    message(FATAL_ERROR "Invalid value for PROTECTION: ${PROTECTION}. Please use FLEXLM or RLM.")
endif()

message(STATUS "----------------------------------------------------")

# default package list
set(ARCANE_PACKAGE_LIST ${ARCANE_ADDITIONAL_PACKAGES})

# optional package list
set(ARCANE_OPTIONAL_PACKAGE_LIST FLEXLM ${ARCANE_OPTIONAL_PACKAGE_LIST})

message(STATUS "**** CHECK PACKAGE LIST ****")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# TODO: Supprimer le support des packages disabled/enabled.
# Quand on utilisera find_package() partout, il sera possible de désactiver
# un package en positionnant la variable CMAKE_DISABLE_FIND_PACKAGE_<PackageName>
#
# Positionne une variable 'DISABLE_PKG_*' pour chaque package supprimé via l'option --disable du configure
set(ARCANE_CONFIG_DISABLED_PACKAGES $ENV{ARCANE_CONFIG_DISABLED_PACKAGES})
foreach(package ${ARCANE_CONFIG_DISABLED_PACKAGES})
  set(DISABLE_PKG_${package} "YES")
endforeach()

# Positionne une variable ENABLE_PKG_*' pour chaque package activé via l'option --enable du configure
set(ARCANE_CONFIG_ENABLED_PACKAGES $ENV{ARCANE_CONFIG_ENABLED_PACKAGES})
foreach(package ${ARCANE_CONFIG_ENABLED_PACKAGES})
  set(ENABLE_PKG_${package} "YES")
endforeach()

# Parcours tous les packages et supprime ceux que l'utilisateur a spécifié comme supprimé
foreach(package ${ARCANE_PACKAGE_LIST})
  set(_TMP_PKG ${DISABLE_PKG_${package}})
  message(STATUS "IS DISABLED? ${package} ${_TMP_PKG}")
  IF(_TMP_PKG STREQUAL "YES")
    message(STATUS "Packages '${package}' disabled from configure")
    set(DISABLE_PKG_${package} "YES" CACHE STRING "Disabled package from configure" FORCE)
    set(${package}_FOUND "NO")
    set(${package}_INCLUDE_DIRS)
    set(${package}_LIBRARIES   )
  else()
    if(ENABLE_PKG_${package} STREQUAL "YES")
      if(${package}_FOUND STREQUAL "NO")
        message(FATAL_ERROR "Requested package ${package} from configure but not found")
      endif()
    endif()
  endif()
endforeach()

# Parcours tous les packages et active ceux que l'utilisateur a spécifier comme activé
foreach(package ${ARCANE_OPTIONAL_PACKAGE_LIST})
  SET(_TMP_PKG ${ENABLE_PKG_${package}})
  MESSAGE(STATUS "IS ENABLED?  ${package} ${_TMP_PKG}")
  if(_TMP_PKG STREQUAL "YES")
    MESSAGE(STATUS "Packages '${package}' enabled from configure")
    SET(ENABLE_PKG_${package} "YES" CACHE STRING "Enabled package from configure" FORCE)
    if(${package}_FOUND STREQUAL "NO")
	    MESSAGE(FATAL_ERROR "Requested package ${package} but not found")
    endif(${package}_FOUND STREQUAL "NO")
	else()
    SET(${package}_FOUND "NO")
    SET(${package}_INCLUDE_DIRS)
    SET(${package}_LIBRARIES   )
  endif(_TMP_PKG STREQUAL "YES")
endforeach()

set(ARCANE_PACKAGE_LIST ${ARCANE_PACKAGE_LIST} ${ARCANE_OPTIONAL_PACKAGE_LIST})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

include(FindDoxygen)

# Avec le compilateur Microsoft, spécifie qu'on souhaite les exceptions classique ISO C++
# (il semble en avoir besoin avec VS2013 (MSVC12) mais pas VS2010)
# TODO: verifier si cela est toujours utile et si c'est le cas le mettre dans arcane_compile_flags.
if(MSVC12)
 add_definitions(/EHs)
endif()

# Oblige à declarer les symboles externes des bibliotheques
set(ARCANE_USE_HIDDEN_VISIBILITY TRUE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion du C#.

include(ArcconDotNet)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/GlobalCSharpTarget.cmake)

# ----------------------------------------------------------------------------

SET(ARCANE_EXTERNAL_LIBRARY_DIRS)

# Desactive le wrapping C# si le type 'Real' est 'long double'
# car cela ne fonctionne pas (pas encore)
if(ARCANE_WANT_LONGDOUBLE)
  set(ARCANE_DOTNET FALSE)
endif()

IF(NOT FLEXLM_FOUND)
  MESSAGE(STATUS "PACKAGE: ${FLEXLM_PROTECTION_NAME}         : not found")
ELSE(NOT FLEXLM_FOUND)
  MESSAGE(STATUS "PACKAGE: ${FLEXLM_PROTECTION_NAME}         : ${FLEXLM_LIBRARIES}")
ENDIF(NOT FLEXLM_FOUND)

IF(ARCANE_CONFIGURE_WANT_MPC)
  IF(NOT MPC_FOUND)
    MESSAGE(FATAL_ERROR "PACKAGE: MPC enabled in configure but not found")
  ENDIF(NOT MPC_FOUND)
ENDIF(ARCANE_CONFIGURE_WANT_MPC)

set(ARCANE_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ARCANE_ARC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion des 'axl'.

set(ARCANE_AXL2CC ${AXLSTAR_AXL2CC})
message(STATUS "Arcane: axl2cc is '${ARCANE_AXL2CC}'")

# Liste des dépendances pour la génération des axl en C#.
set(ARCANE_AXLSTAR_DEPEND "${AXLSTAR_AXL2CC}")
if (TARGET dotnet_axl_depend)
  list(APPEND ARCANE_AXLSTAR_DEPEND dotnet_axl_depend)
endif()

# ----------------------------------------------------------------------------
# Recherche 'googletest' pour les tests unitaires
find_package(GTest)
if (GTEST_FOUND)
  include(GoogleTest)
endif()

# ----------------------------------------------------------------------------
# Disabled in arcane.pc (check it if useful)
set(ARCANE_INCLUDE_LIST ${ARCANE_INCLUDE_LIST} ${GLIB_INCLUDE_DIRS})

set(ARCANE_CORE_LIBRARY arcane_core)
set(ARCANE_BASE_LIBRARIES arcane_mesh arcane_impl arcane_core arcane_utils)

# Ajoute répertoire ${name}_INCLUDE_DIRS à la liste des -I si le package $name est disponible
# Si un deuxième argument est présent, il s'agit de la cible à ajouter.
macro(add_include_directories_if_found name)
  if(${name}_FOUND)
    message(STATUS "Including ${name} path for next component parsing: ${${name}_INCLUDE_DIRS}")
    if(${ARGC} GREATER 1)
      target_include_directories(${ARGV1} PRIVATE ${${name}_INCLUDE_DIRS})
    else()
      include_directories(${${name}_INCLUDE_DIRS})
    endif()
  endif()
endmacro()

add_subdirectory(src/arcane/ios)

if(NOT ARCANE_DISABLE_HYODA)
  add_subdirectory(src/arcane/hyoda)
  if(TARGET arcane_hyoda)
    list(APPEND ARCANE_ADDITIONAL_COMPONENTS arcane_hyoda)
  endif()
endif()

add_subdirectory(src/arcane/dotnet/coreclr)

add_subdirectory(src/arcane/parallel/mpi)
add_subdirectory(src/arcane/parallel/thread)

if(TARGET arcane_mpi)
  set(ARCANE_HAS_ARCANEMPI TRUE)
endif()

set(ARCANE_THREAD_LIBRARY arcane_thread)
if(NOT WIN32)
  list(APPEND ARCANE_EXTERNAL_LIBRARIES pthread)
endif()
if(MPI_FOUND)
  if(NOT ARCANE_USE_MPC)
    add_subdirectory(src/arcane/parallel/mpithread)
    set(ARCANE_USE_MPITHREAD 1)
  endif()
endif()

if(ARCANE_WANT_GEOMETRY)
  list(APPEND ARCANE_LIBRARIES arcane_geometry)
endif()

# Avec MPC, il faut ajouter des arguments lors de la compilation
# Ces arguments sont positionnes dans FindMPC.cmake
# Il faut mettre ces arguments au debut pour etre sur que les includes
# MPC sont au debut des chemins recherches par le compilateur
if(ARCANE_USE_MPC)
  target_compile_options(arcane_build_compile_flags INTERFACE ${MPC_CFLAGS})
endif()

if(FLEXLM_FOUND)
  # TODO: mettre cela dans un 'target_compile_definitions'
  ADD_DEFINITIONS(-DARCANE_FLEXLM)
  set(ARCANE_INCLUDE_LIST ${ARCANE_INCLUDE_LIST} ${FLEXLM_INCLUDE_DIR})
  list(APPEND ARCANE_EXTERNAL_LIBRARIES ${FLEXLM_LIBRARIES})
  # TODO: a priori cette variable n'est plust utilisée donc on devrait pouvoir la supprimer
  list(APPEND ARCANE_EXTERNAL_LIBRARY_DIRS ${FLEXLM_LIBRARY_DIRS})
endif()

if(ARCANE_USE_MPC)
  list(APPEND ARCANE_LIBRARIES ${MPC_LDFLAGS})
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

message(STATUS "ARCANE_EXTERNAL_LIBRARY_DIRS is ${ARCANE_EXTERNAL_LIBRARY_DIRS}")

# ----------------------------------------------------------------------------

# TODO: ajouter en INTERFACE 'ARCANE_NO_USING_FOR_STREAM' mi-2022 lorsque toutes
# les branches de développement de Arcane auront été fusionnées
# target_compile_definitions(arcane_build_compile_flags INTERFACE ARCANE_NO_USING_FOR_STREAM)

# Force à rendre privé les structures gérant les connectivités pour préparer leur suppression
target_compile_definitions(arcane_build_compile_flags INTERFACE ARCANE_FORCE_HIDE_ITEM_CONNECTIVITY_STRUCTURE)

if (UNIX)
  # Sur certaines plateformes (ex: centos7), il faut ajouter ces options
  # pour la compilation.
  target_compile_options(arcane_build_compile_flags INTERFACE -pthread)
  target_link_options(arcane_build_compile_flags INTERFACE -pthread)
endif()

# ----------------------------------------------------------------------------
# Meme si on n'est pas en mode verif, il faut toujours avoir les symboles de debug
# pour le profiling ou pour les debuggers
if (UNIX)
  target_compile_options(arcane_build_compile_flags INTERFACE -g)
endif ()
if (MSVC)
  # '/Zi' permet de créér des fichiers '.pdb' mais n'est pas compatible avec 'ccache'.
  # On utilise donc '/Z7' qui place les informations de debug dans les '.obj'.
  target_compile_options(arcane_build_compile_flags INTERFACE /Z7)
  target_link_options(arcane_build_compile_flags INTERFACE /debug)
  # Indique qu'on souhaite utiliser le pré-processeur standard
  # C'est nécessaire pour la macro __VA_OPT__ du C++20.
  target_compile_options(arcane_build_compile_flags INTERFACE /Zc:preprocessor)
endif()

# ----------------------------------------------------------------------------

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID STREQUAL Clang))
  target_compile_options(arcane_build_compile_flags INTERFACE -Wall -Wdeprecated ${_CONVERSION_WARNING})
  if (ARCANE_DISABLE_DEPRECATED_WARNINGS)
    target_compile_options(arcane_build_compile_flags INTERFACE -Wno-deprecated -Wno-deprecated-declarations)
  endif()
  if(ARCANE_WANT_AVX)
    target_compile_options(arcane_export_compile_flags INTERFACE -mavx)
  endif()
  if(ARCANE_WANT_AVX512)
    target_compile_options(arcane_export_compile_flags INTERFACE -mavx512f -mavx512cd)
  endif()
  message(STATUS "Compiler is g++")
  if(ARCANE_USE_HIDDEN_VISIBILITY)
    target_compile_options(arcane_build_compile_flags INTERFACE -fvisibility-inlines-hidden -fvisibility=hidden)
  endif()
endif()

# Sous windows ou avec NVHPC, désactive les '#warning'
if (WIN32 OR (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC))
  target_compile_definitions(arcane_build_compile_flags INTERFACE NO_USER_WARNING)
endif ()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Positionne les listes qui peuvent être remplies par les répertoires
# additionnels.

set(ARCANE_OPTIONAL_LIBRARIES)

# ----------------------------------------------------------------------------
# Regarde si les compteurs 'perf' sont disponibles

if (UNIX)
  try_compile(ARCANE_HAS_LINUX_PERF_COUNTERS ${CMAKE_CURRENT_BINARY_DIR} ${Arcane_SOURCE_DIR}/cmake/test_linux_perf_syscall.cc)
  message(STATUS "Check: perf_event syscalls are available? = ${ARCANE_HAS_LINUX_PERF_COUNTERS}")
  set(ARCANE_HAS_LINUX_PERF_COUNTERS ${ARCANE_HAS_LINUX_PERF_COUNTERS} CACHE BOOL "Is linux perf counters are availables" FORCE)
endif()

# Regarde si les 'hooks' mémoire de 'malloc.h' de la glibc sous linux
# sont disponibles
if (UNIX)
  try_compile(ARCANE_HAS_MALLOC_HOOKS ${CMAKE_CURRENT_BINARY_DIR} ${Arcane_SOURCE_DIR}/cmake/test_glibc_malloc_hooks.cc)
  message(STATUS "Check: malloc hooks are available? = ${ARCANE_HAS_MALLOC_HOOKS}")
  set(ARCANE_HAS_MALLOC_HOOKS ${ARCANE_HAS_MALLOC_HOOKS} CACHE BOOL "Is glibc 'malloc hooks' are availables" FORCE)
endif()

# Ajoute option pour désactiver les tests utilisant les compteurs hardware car ils ne sont pas
# forcément accessibles notamment dans les dockers des machines du CI
option(ARCANE_DISABLE_PERFCOUNTER_TESTS "True if we disable tests using perf counters" OFF)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Option pour indiquer si on veut utiliser la nouvelle implémentation
# de ItemInternalMap. Cette nouvelle implémentation est plus rapide et consomme
# moins de mémoire que l'ancienne.
# A partir de la version 4.0 de Arcane, seule la nouvelle implémentation est disponible.
set(ARCANE_USE_HASHTABLEMAP2_FOR_ITEMINTERNALMAP TRUE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "ARCANE_LIBRARIES '${ARCANE_LIBRARIES}'")

add_subdirectory(src/arcane/utils)
add_subdirectory(src/arcane/impl)
add_subdirectory(src/arcane/mesh)
add_subdirectory(src/arcane/launcher)
add_subdirectory(src/arcane/core)

add_subdirectory(src/arcane/geometry)
# TODO: regarder si toujours utile de positionner cela.
# (pour l'instant cela n'est utilisé que pour les tests)
set(ARCANE_HAS_GEOMETRY YES)

add_subdirectory(src/arcane/corefinement)

if(ARCANE_WANT_TOTALVIEW)
  add_subdirectory(src/arcane/totalview)
  set(ARCANE_ADDITIONAL_COMPONENTS arcane_totalview ${ARCANE_ADDITIONAL_COMPONENTS})
endif()

if (ARCANE_WANT_64BIT)
  SET(ARCANE_DOTNET_64BIT_DEFINE "/define:ARCANE_64BIT")
endif()

add_subdirectory(src/arcane/hdf5)
add_subdirectory(src/arcane/std)
if (ARCANE_ENABLE_ALEPH)
  add_subdirectory(src/arcane/aleph)
endif()
add_subdirectory(src/arcane/driverlib)
add_subdirectory(src/arcane/driver)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion du support des accélérateurs.
# Le choix du runtime accélérateur est maintenant dans Arccore (2025/11)

if (ARCANE_HAS_ACCELERATOR)
  # Les macros ARCANE_WANT* sont utilisées uniquement pour spécifier
  # le runtime utilisé dans 'arcane_core_config'. A terme il faudra
  # utiliser ce qui est dans Arccore.
  if (ARCANE_ACCELERATOR_MODE STREQUAL "ROCM")
    enable_language(HIP)
    set (ARCANE_WANT_HIP TRUE)
  elseif(ARCANE_ACCELERATOR_MODE STREQUAL "CUDA")
    if (NOT CMAKE_CUDA_HOST_COMPILER)
      set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()
    enable_language(CUDA)
    set (ARCANE_WANT_CUDA TRUE)
  elseif (ARCANE_ACCELERATOR_MODE STREQUAL "SYCL" )
    set (ARCANE_WANT_SYCL TRUE)
  endif()
endif()

add_subdirectory(src/arcane/accelerator/core)
add_subdirectory(src/arcane/accelerator)

if (ARCANE_HAS_ACCELERATOR)
  add_subdirectory(src/arcane/accelerator/${ARCANE_ACCELERATOR_RUNTIME}/runtime)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion des outils 'C#' et du wrapper 'Swig'

add_subdirectory(tools)

# ----------------------------------------------------------------------------
# Gestion du wrapper python (nécessite le wrapper C#)
if (ARCANE_ENABLE_DOTNET_PYTHON_WRAPPER)
  add_subdirectory(python)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Détection ou choix du partitionneur par défaut

if (NOT ARCANE_DEFAULT_PARTITIONER)
  set (ARCANE_DEFAULT_PARTITIONER "Auto")
endif()

message(STATUS "Wanted Default Partitioner is '${ARCANE_DEFAULT_PARTITIONER}'")
if(MPI_FOUND)
  set(PARTITIONER_FOUND FALSE)
  if(ARCANE_DEFAULT_PARTITIONER STREQUAL "Auto")
    message(STATUS "Automatic detection of Default Partitioner in the following order: 'Metis', 'Zoltan', 'PTScotch'")
    # Si on chosit le mode automatiquement, on prend en priorité 'Parmetis' puis
    # 'Zoltan' puis 'Scotch'.
    if(Parmetis_FOUND)
      set(ARCANE_DEFAULT_PARTITIONER "Metis")
    elseif(Zoltan_FOUND)
      set(ARCANE_DEFAULT_PARTITIONER "Zoltan")
    elseif(PTScotch_FOUND)
      set(ARCANE_DEFAULT_PARTITIONER "PTScotch")
    endif()
    message(STATUS "'${ARCANE_DEFAULT_PARTITIONER}' is your default partitioner")
  elseif(ARCANE_DEFAULT_PARTITIONER STREQUAL "Metis")
    if(NOT Parmetis_FOUND)
      message(FATAL_ERROR "Default Partitioner '${ARCANE_DEFAULT_PARTITIONER}' is not available")
    endif()
  elseif(ARCANE_DEFAULT_PARTITIONER STREQUAL "PTScotch")
    if(NOT PTScotch_FOUND)
      MESSAGE(FATAL_ERROR "Default Partitioner '${ARCANE_DEFAULT_PARTITIONER}' is not available") 
    endif()
  elseif(ARCANE_DEFAULT_PARTITIONER STREQUAL "Zoltan")
    if(NOT Zoltan_FOUND)
      MESSAGE(FATAL_ERROR "Default Partitioner '${ARCANE_DEFAULT_PARTITIONER}' is not available") 
    endif()
  else()
    if(Parmetis_FOUND OR PTScotch_FOUND OR Zoltan_FOUND)
      message(FATAL_ERROR "Default Partitioner '${ARCANE_DEFAULT_PARTITIONER}' is not allowed : choose between Metis, PTScotch, Zoltan or Auto")
    endif()
  endif()
  if(Parmetis_FOUND OR TARGET PTScotch_FOUND OR Zoltan_FOUND)
    message(STATUS "'${ARCANE_DEFAULT_PARTITIONER}' is your default partitioner")
    message(STATUS "Setting PARTITIONER_FOUND to TRUE")
    set(PARTITIONER_FOUND TRUE)
    set(ARCANE_DEFAULT_PARTITIONER "${ARCANE_DEFAULT_PARTITIONER}" CACHE STRING "Default partitioner" FORCE)
    set(ARCANE_DEFAULT_PARTITIONER_IS_METIS FALSE)
    if(ARCANE_DEFAULT_PARTITIONER STREQUAL Metis)
      set(ARCANE_DEFAULT_PARTITIONER_IS_METIS TRUE)
    endif()
    set(ARCANE_DEFAULT_PARTITIONER_IS_METIS "${ARCANE_DEFAULT_PARTITIONER_IS_METIS}" CACHE STRING "Is Metis the default partitioner?" FORCE)
  endif()
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Les tests ont besoin d'infos comme le partitionneur utilisé  donc il
# faut ajouter le répertoire de test en dernier.

if (ARCANE_ENABLE_TESTS)
  add_subdirectory(src/arcane/tests)
endif()

# ----------------------------------------------------------------------------

# Gestion des exemples
add_subdirectory(samples_build)

# ----------------------------------------------------------------------------
# Ajoute les répertoires 'matériaux' et 'cartesianmesh'
# Pour l'instant on le fait après les tests pour cohérence avec l'existant
# car avant ces répertoires étaientt dans ceapart.
add_subdirectory(src/arcane/cartesianmesh)
if (ARCANE_HAS_ACCELERATOR_API)
  add_subdirectory(src/arcane/materials)
endif()

# ----------------------------------------------------------------------------
# Ajoute les sources spécifiques CEA.

include(${CMAKE_CURRENT_SOURCE_DIR}/ceapart/Arcane.cmake)
add_subdirectory(ceapart ${CMAKE_BINARY_DIR}/cea)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "ARCANE_LIBRARIES is ${ARCANE_LIBRARIES}")
message(STATUS "ARCANE_EXTERNAL_LIBRARIES is ${ARCANE_EXTERNAL_LIBRARIES}")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

function(GENERATE_DOC_TARGET docname)
  set(DOC_OUTPUT_DIR ${CMAKE_BINARY_DIR})
  configure_file(${ARCANESRCROOT}/doc/${docname}.doxyfile ${ARCANEBUILDROOT}/share/${docname}.doxyfile @ONLY)
  add_custom_target(${docname} COMMAND
      ${DOXYGEN} ${ARCANEBUILDROOT}/share/${docname}.doxyfile &&
      echo "Doc index file : file://${ARCANEBUILDROOT}/share/${docname}/html/index.html")

  set(DOC_OUTPUT_DIR ${CMAKE_INSTALL_PREFIX})
  configure_file(${ARCANESRCROOT}/doc/${docname}.doxyfile ${ARCANEBUILDROOT}/share/${docname}_install.doxyfile @ONLY)
  add_custom_target(install_${docname} COMMAND ${DOXYGEN} ${ARCANEBUILDROOT}/share/${docname}_install.doxyfile)
endfunction()

# ----------------------------------------------------------------------------
# Cibles pour generer la documentation

# Variable permettant de choisir le thème des documentations user et dev.
if(ARCANEDOC_LEGACY_THEME STREQUAL ON)
  set(ARCANEDOC_HEADER_USER "header_no_theme.html")
  set(ARCANEDOC_HEADER_DEV "header_no_theme.html")
  set(ARCANEDOC_AXLDOC_LEGACY "--legacy")
  message(STATUS "User and dev documentations with legacy theme")
else()
  set(ARCANEDOC_HEADER_USER "header.html")
  set(ARCANEDOC_HEADER_DEV "header.html")
  set(ARCANEDOC_AXLDOC_LEGACY "")
endif()

# Variable pour savoir si la doc aura accès à internet ou non.
if(ARCANEDOC_OFFLINE STREQUAL ON)
  set(ARCANEDOC_MATHJAX "NO")
  message(STATUS "Offline mode for user and dev documentations")
else()
  set(ARCANEDOC_MATHJAX "YES")
endif()

# ----------------------------------------------------------------------------

set(IS_FOR User)
configure_file(doc/generate_axl_infos.sh.in lib/generate_axl_infos.sh @ONLY)
set(IS_FOR Dev)
configure_file(doc/generate_axl_infos.sh.in lib/generate_axl_infos_dev.sh @ONLY)

file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/share/axl)

# ----------------------------------------------------------------------------

message(STATUS "DOXYGEN = ${DOXYGEN}")
if(DOXYGEN)
  GENERATE_DOC_TARGET(devdoc)
  GENERATE_DOC_TARGET(userdoc)
  if (UNIX)
    # Cible pour générer la documentation des axl
    # TODO: pouvoir changer le shell
    add_custom_target(generate_axl_doc COMMAND "/bin/sh" ${CMAKE_CURRENT_BINARY_DIR}/lib/generate_axl_infos.sh)
    add_custom_target(generate_axl_doc_dev COMMAND "/bin/sh" ${CMAKE_CURRENT_BINARY_DIR}/lib/generate_axl_infos_dev.sh)
    add_dependencies(userdoc generate_axl_doc)
    add_dependencies(devdoc generate_axl_doc_dev)
  endif()
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "BUILD TYPE ${CMAKE_BUILD_TYPE}")
message(STATUS "DEBUG FLAGS TYPE ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "RELEASE FLAGS TYPE ${CMAKE_CXX_FLAGS_RELEASE}")

# ----------------------------------------------------------------------------
# Configuration pour les connectivités.
#
# Il n'existe plus qu'un seul mode pour les connectivités:
# ARCANE_ITEM_CONNECTIVITY_SIZE_MODE = 2
# A terme il faudra supprimer cette variable
set(ARCANE_ITEM_CONNECTIVITY_SIZE_MODE 2 CACHE STRING "Item connectivity size mode")

# ----------------------------------------------------------------------------

macro(ARCANE_WRITE_ONE_CONFIG varname configname)
  if(${varname})
    set(_STR "${_STR}#ifndef ${configname}\n#define ${configname}\n#endif\n")
  else()
    set(_STR "${_STR}//#define ${configname}\n")
  endif()
endmacro()

macro(ARCANE_WRITE_ONE_CONFIG_VALUE varname configname)
  if(${varname})
    set(_STR "${_STR}#ifndef ${configname}\n#define ${configname} ${${configname}}\n#endif\n")
  else()
    set(_STR "${_STR}//#define ${configname}\n")
  endif()
endmacro()

function(arcane_write_config_str str filename)
  file(WRITE ${CMAKE_BINARY_DIR}/lib/${filename}.gen ${str})
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/lib/${filename}.gen ${CMAKE_BINARY_DIR}/${filename})
  install(FILES ${CMAKE_BINARY_DIR}/${filename} DESTINATION include)
endfunction()

# Fonction pour générer le fichier 'arcane_config_core.h' indiquant la disponibilité
# de telle ou telle option
function(arcane_gen_core_config_file filename)
  set(_STR "// # File generated from configure. Do not edit\n")
  set(_STR "${_STR}// # Options\n")
  set(_STR "${_STR}#ifndef ARCANE_CORE_CONFIG_H\n")
  set(_STR "${_STR}#define ARCANE_CORE_CONFIG_H\n")
  set(_STR "${_STR}#define ARCANE_TYPE_INT16 ARCCORE_TYPE_INT16\n")
  set(_STR "${_STR}#define ARCANE_TYPE_INT32 ARCCORE_TYPE_INT32\n")
  set(_STR "${_STR}#define ARCANE_TYPE_INT64 ARCCORE_TYPE_INT64\n")
  set(_STR "${_STR}#define ARCANE_INTEGER_MAX ARCCORE_INTEGER_MAX\n")
  set(_STR "${_STR}#define ARCANE_INT32_MAX ${ARCANE_INT32_MAX}\n")
  set(_STR "${_STR}#define ARCANE_INT64_MAX ${ARCANE_INT64_MAX}\n")
  set(_STR "${_STR}#define ARCANE_ITEM_CONNECTIVITY_SIZE_MODE_LEGACY 1\n")
  set(_STR "${_STR}#define ARCANE_ITEM_CONNECTIVITY_SIZE_MODE_NEW 2\n")
  set(_STR "${_STR}#define ARCANE_ITEM_CONNECTIVITY_SIZE_MODE_DYNAMIC 3\n")
  set(_STR "${_STR}#define ARCANE_ITEMENUMERATOR_VERSION ${ARCANE_ITEMENUMERATOR_VERSION}\n")
  ARCANE_WRITE_ONE_CONFIG(ARCANE_USE_THREAD ARCANE_HAS_THREAD)
  ARCANE_WRITE_ONE_CONFIG(ARCCORE_CHECK ARCANE_CHECK)
  ARCANE_WRITE_ONE_CONFIG(ARCCORE_DEBUG ARCANE_DEBUG)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_CUDA ARCANE_HAS_CUDA)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_HIP ARCANE_HAS_HIP)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_SYCL ARCANE_HAS_SYCL)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_USE_CXX11 ARCANE_HAS_CXX11)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_USE_CXX14 ARCANE_HAS_CXX14)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_HAS_CXX20 ARCANE_HAS_CXX20)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_HAS_ACCELERATOR_API ARCANE_HAS_ACCELERATOR_API)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_HAS_ACCELERATOR ARCANE_HAS_ACCELERATOR)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_64BIT ARCANE_64BIT)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_LONGDOUBLE ARCANE_REAL_LONG)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_AVX ARCANE_HAS_AVX)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_AVX512 ARCANE_HAS_AVX512)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_WANT_LIBXML2 ARCANE_USE_LIBXML2)
  ARCANE_WRITE_ONE_CONFIG_VALUE(ARCANE_ITEM_CONNECTIVITY_SIZE_MODE ARCANE_ITEM_CONNECTIVITY_SIZE_MODE)
  ARCANE_WRITE_ONE_CONFIG_VALUE(ARCANE_USE_HASHTABLEMAP2_FOR_ITEMINTERNALMAP ARCANE_USE_HASHTABLEMAP2_FOR_ITEMINTERNALMAP)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_USE_MPC ARCANE_USE_MPC)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_USE_HIDDEN_VISIBILITY ARCANE_USE_HIDDEN_VISIBILITY)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_USE_STD_UNORDEREDMAP ARCANE_USE_STD_UNORDEREDMAP)
  ARCANE_WRITE_ONE_CONFIG(ARCANE_HAS_ARCANEMPI ARCANE_HAS_ARCANEMPI)

  SET(_STR "${_STR}#define METIS_DEFAULT_PARTITIONER 1\n")
  SET(_STR "${_STR}#define PTSCOTCH_DEFAULT_PARTITIONER 2\n")
  SET(_STR "${_STR}#define ZOLTAN_DEFAULT_PARTITIONER 3\n")
  Set(_STR "${_STR}#define ARCANE_DEFAULT_PARTITIONER_STR \"${ARCANE_DEFAULT_PARTITIONER}\"\n")
  STRING(TOUPPER ${ARCANE_DEFAULT_PARTITIONER} ARCANE_DEFAULT_PARTITIONER)
  SET(_STR "${_STR}#ifndef ARCANE_DEFAULT_PARTITIONER\n#define ARCANE_DEFAULT_PARTITIONER ${ARCANE_DEFAULT_PARTITIONER}_DEFAULT_PARTITIONER\n#endif\n")
  SET(_STR "${_STR}#endif // ARCANE_CORE_CONFIG_H\n")
  ARCANE_WRITE_CONFIG_STR(${_STR} ${filename})
endfunction()

# Fonction pour générer le fichier 'arcane_packages.h' indiquant la disponibilité
# de tel ou tel package ou composante
function(arcane_gen_package_config_file filename)
  SET(_STR "// # File generated from configure. Do not edit\n")
  SET(_STR "${_STR}// # Packages list\n")
  foreach(package ${ARCANE_PACKAGE_LIST})
    # S'assure que le package n'a pas de '-' dans le nom sinon cela
    # génère une macro invalide
    string(REPLACE "-" "_" package "${package}")
    # Si une cible 'arcane::${package} existe, alors le package est disponible.
    string(TOLOWER ${package} lower_pkg_name)
    string(TOUPPER ${package} upper_pkg_name)
    if (TARGET arcane::${lower_pkg_name})
      ARCANE_WRITE_ONE_CONFIG(TRUE ARCANE_HAS_PACKAGE_${package})
      ARCANE_WRITE_ONE_CONFIG(TRUE ARCANE_HAS_PACKAGE_${upper_pkg_name})
    elseif (TARGET arccon::${package})
      ARCANE_WRITE_ONE_CONFIG(TRUE ARCANE_HAS_PACKAGE_${package})
      ARCANE_WRITE_ONE_CONFIG(TRUE ARCANE_HAS_PACKAGE_${upper_pkg_name})
    else()
      ARCANE_WRITE_ONE_CONFIG(${package}_FOUND ARCANE_HAS_PACKAGE_${package})
      ARCANE_WRITE_ONE_CONFIG(${package}_FOUND ARCANE_HAS_PACKAGE_${upper_pkg_name})
    endif()
  endforeach()
  ARCANE_WRITE_CONFIG_STR(${_STR} ${filename})
endfunction()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/arcane_internal_config.h.in ${CMAKE_BINARY_DIR}/arcane_internal_config.h)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Ces propriétés sont utilisées dans le fichier de configuration généré et il ne
# faut pas les supprimer.
get_property(ARCANE_CMAKE_PACKAGE_LIST GLOBAL PROPERTY PACKAGES_FOUND)
message(STATUS "PackageList registered by CMake: ${ARCANE_CMAKE_PACKAGE_LIST}")
list(APPEND ARCANE_PACKAGE_LIST ${_CMAKE_PACKAGE_LIST} ${ARCANE_CMAKE_PACKAGE_LIST})
message(STATUS "ArcanePackageList: ${ARCANE_PACKAGE_LIST}")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Fichier qui contiendra l'ensemble des chemins des bibliothèques
# dynamiques. Elle sera utilisée par file(GET_RUNTIME_DEPENDENCIES) pour
# recopier ces bibliothèques dans le répertoire contenant les bibliothèques
# Arcane
set(ARCANE_COPY_DLLS_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/copy_dlls.srcs.cmake)
file(WRITE ${ARCANE_COPY_DLLS_CONFIG_FILE} "# This file is generated. Do not edit\n")

# ----------------------------------------------------------------------------
# Récapitule les infos sur les packages
foreach(package ${ARCANE_PACKAGE_LIST})
  set(FORMATTED_PACKAGE_NAME "${package}                    ")
  string(SUBSTRING "${FORMATTED_PACKAGE_NAME}" 0 20 FORMATTED_PACKAGE_NAME)
  if(NOT ${package}_FOUND)
    message(STATUS   "PACKAGE: ${FORMATTED_PACKAGE_NAME} : not found")
    continue()
  endif()

  string(TOLOWER ${package} target_name)
  set(_TARGETNAME)
  if (TARGET arcane::${target_name})
    set(_TARGETNAME arcane::${target_name})
  elseif (TARGET arccon::${target_name})
    set(_TARGETNAME arccon::${target_name})
  elseif (TARGET arccon::${package})
    set(_TARGETNAME arccon::${package})
  endif()
  if (_TARGETNAME)
    message(STATUS     "${package} (target: ${_TARGETNAME})")
    get_target_property(_INCS ${_TARGETNAME} INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_LIBS ${_TARGETNAME} INTERFACE_LINK_LIBRARIES)
    if (_INCS)
      message(STATUS   "  includes:")
      foreach(_inc ${_INCS})
        message(STATUS   "    ${_inc}")
      endforeach()
    endif()
    if (_LIBS)
      message(STATUS   "  libraries:")
      foreach(_lib ${_LIBS})
        file(APPEND ${ARCANE_COPY_DLLS_CONFIG_FILE} "list(APPEND LIBS_LIST \"$<TARGET_GENEX_EVAL:${_TARGETNAME},${_lib}>\")\n")
        message(STATUS   "    ${_lib}")
      endforeach()
    endif()
  else()
    # Si le package contient des includes où des bibliothèques,
    # vérifie qu'il y a bien une cible arcane::${package} associée.
    # Certains packages (par exemple valgrind) ne contiennent que des
    # exécutables et dans ce cas il n'y a pas de cible associé.
    if (${package}_INCLUDE_DIRS AND ${package}_LIBRARIES)
      message(WARNING  "No target arcane::${target_name} or arccon::${target_name} for packages '${package}'")
    endif()
    message(STATUS   "PACKAGE: ${FORMATTED_PACKAGE_NAME} : incs: ${${package}_INCLUDE_DIRS}")
    message(STATUS   "PACKAGE: ${FORMATTED_PACKAGE_NAME} : libs: ${${package}_LIBRARIES}")
  endif()
endforeach()

# ----------------------------------------------------------------------------
# Affiche les packages non trouvés

get_property(_NOT_FOUND_PACKAGE_LIST GLOBAL PROPERTY PACKAGES_NOT_FOUND)
if (_NOT_FOUND_PACKAGE_LIST)
  message(STATUS "")
  message(STATUS "")
  message(STATUS "WARNING: The following packages were NOT FOUND:")
  foreach(pkg ${_NOT_FOUND_PACKAGE_LIST})
    message(STATUS "    ${pkg}")
  endforeach()
  message(STATUS "\n")
endif()

# ----------------------------------------------------------------------------
# Gestion des DLLs pour Windows:
# 1. Ajoute une commande et une cible pour copier les DLLs dans le répertoire 'lib'
# 2. Si on utilise IntelMPI, copie 'libfabric.dll' dans ce même répertoire. Cette
# bibliothèque n'est pas utilisée lors de l'édition de lien mais est nécessaire
# lors de l'exécution pour ne pas planter lors de l'appel à MPI_Init()
#
# TODO: ne pas mettre en dur l'exécutable de test mais laisser aux sous-répertoires
# la possibilité d'ajouter des exécutables.
#
# TODO: pour 'libfabric.dll', plutôt modifier le package 'intel-mpi' de 'vcpkg' pour
# indiquer explicitement où se trouve cette DLL.
#
if (TARGET arcane_tests_exec)
  file(APPEND ${ARCANE_COPY_DLLS_CONFIG_FILE} "list(APPEND EXE_LIST \"$<TARGET_FILE:arcane_tests_exec>\")\n")
endif()
# Pour copier 'nethost.dll' sous Windows.
if (TARGET arcane_dotnet_coreclr)
  file(APPEND ${ARCANE_COPY_DLLS_CONFIG_FILE} "list(APPEND EXE_LIST \"$<TARGET_FILE:arcane_dotnet_coreclr>\")\n")
endif()
foreach(pkg ${ARCANE_LIBRARIES})
  if (TARGET ${pkg})
    file(APPEND ${ARCANE_COPY_DLLS_CONFIG_FILE} "list(APPEND EXE_LIST \"$<TARGET_FILE:${pkg}>\")\n")
  endif()
endforeach()
foreach(pkg ${ARCANE_OPTIONAL_LIBRARIES})
  if (TARGET ${pkg})
    file(APPEND ${ARCANE_COPY_DLLS_CONFIG_FILE} "list(APPEND EXE_LIST \"$<TARGET_FILE:${pkg}>\")\n")
  endif()
endforeach()
find_path(INTEL_LIBFABRIC NAMES "libfabric.dll" PATH_SUFFIXES bin)
if (INTEL_LIBFABRIC)
  message (STATUS "Found 'libfabric.dll' in = '${INTEL_LIBFABRIC}'")
  file(COPY ${INTEL_LIBFABRIC}/libfabric.dll DESTINATION "${LIBRARY_OUTPUT_PATH}")
endif()

set(ARCANE_COPY_DLL_GENERATED_FILE ${CMAKE_CURRENT_BINARY_DIR}/copy_dlls_config.$<CONFIG>.cmake)
file(GENERATE OUTPUT ${ARCANE_COPY_DLL_GENERATED_FILE} INPUT ${ARCANE_COPY_DLLS_CONFIG_FILE})
add_custom_target(copy_dlls
  COMMAND ${CMAKE_COMMAND} -D CONFIG_INCLUDE_FILE=${ARCANE_COPY_DLL_GENERATED_FILE} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ArcaneCopyDLLs.cmake )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

arcane_gen_core_config_file(arcane_core_config.h)
arcane_gen_package_config_file(arcane_packages.h)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
## Création et génération de ArcaneConfig.cmake ##

set(ARCANE_EXPORT_PACKAGES TRUE)
if (FRAMEWORK_NO_EXPORT_PACKAGES)
  set(ARCANE_EXPORT_PACKAGES FALSE)
endif()

# ----------------------------------------------------------------------------
# Créé une bibliothèque interface qui s'appelle 'arcane_full' et qui contient
# la liste des bibliothèques utilisées par un code.
# Pour créer cette liste, on prend la variable ARCANE_LIBRARIES
# qui contient l'ensemble des bibliothèques et on supprime de cette
# liste les bibliothèques qu'on qualifie d'optionnelle, c'est à dire
# que le code utilisant Arcane n'est pas obligé de les inclure.
# Cette liste est définie par la variable ARCANE_OPTIONAL_LIBRARIES

message(STATUS "ARCANE_LIBRARIES: ${ARCANE_LIBRARIES}")
message(STATUS "ARCANE_OPTIONAL_LIBRARIES: ${ARCANE_OPTIONAL_LIBRARIES}")

# ----------------------------------------------------------------------------

# Finalise la gestion du C# en créant la commande qui compile le projet
# On doit faire cela à la fin car il n'y a qu'à ce moment là qu'on connait
# toutes les dépendences (en effet CMake ne propage pas les dépendences sur
# add_custom_command)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/GlobalCSharpCommand.cmake)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Création des fichiers ArcaneConfig.cmake et ArcaneTargets.cmake
#
# NOTE: la commande configure_package_config_file ne fonctionne
# que si le répertoire d'installation (CMAKE_INSTALL_PREFIX) est
# défini. Pour que cela fonctionne toujours, si cette variable
# n'est pas définie, on prend comme répertoire d'installation
# ${CMAKE_CURRENT_BINARY_DIR}/install

configure_file(cmake/ArcaneConfigCache.cmake.in ${generated_config_dir}/ArcaneConfigCache.cmake @ONLY)
configure_file(cmake/ArcanePathHints.cmake.in ${generated_config_dir}/ArcanePathHints.cmake @ONLY)
configure_file(cmake/ArcanePathHints-install.cmake.in ${generated_config_dir}/install/ArcanePathHints.cmake @ONLY)
install(FILES ${generated_config_dir}/install/ArcanePathHints.cmake DESTINATION ${ARCANE_CMAKE_CONFIG_DIR})

install(EXPORT ArcaneTargets DESTINATION ${ARCANE_CMAKE_CONFIG_DIR} NAMESPACE "Arcane::")
export(EXPORT ArcaneTargets FILE ${ARCANE_CMAKE_CONFIG_DIR}/ArcaneTargets.cmake)

set(ARCANE_INSTALL_INCLUDE_DIR include)
set(ARCANE_INSTALL_LIB_DIR lib)

message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
set(_CONFIGURE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
if (NOT _CONFIGURE_INSTALL_PREFIX)
  set(_CONFIGURE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
endif ()

include(GenerateExportHeader)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/ArcaneConfig.cmake.in
  ${generated_config_dir}/ArcaneConfig.cmake
  INSTALL_DESTINATION ${ARCANE_CMAKE_CONFIG_DIR}
  INSTALL_PREFIX ${_CONFIGURE_INSTALL_PREFIX}
  PATH_VARS ARCANE_INSTALL_INCLUDE_DIR ARCANE_INSTALL_LIB_DIR)
write_basic_package_version_file(
  ${generated_config_dir}/ArcaneConfigVersion.cmake
  COMPATIBILITY AnyNewerVersion)
install(FILES
  ${generated_config_dir}/ArcaneConfig.cmake
  ${generated_config_dir}/ArcaneConfigCache.cmake
  ${generated_config_dir}/ArcaneConfigVersion.cmake
  DESTINATION ${ARCANE_CMAKE_CONFIG_DIR})

# Installe les 'Find*' fournis par Arcane
arccon_install_directory(NAMES cmake/Modules DESTINATION "${ARCANE_CMAKE_CONFIG_DIR}")

# ----------------------------------------------------------------------------

set(_ARCANE_PACKAGES_ROOTINFO)
foreach(package ${ARCANE_PACKAGE_LIST})
  if (${package}_DIR)
    set(_ARCANE_PACKAGES_ROOTINFO "${_ARCANE_PACKAGES_ROOTINFO}arcane_set_if_unset(${package}_DIR \"${${package}_DIR}\")\n")
  elseif (${package}_ROOT)
    set(_ARCANE_PACKAGES_ROOTINFO "${_ARCANE_PACKAGES_ROOTINFO}arcane_set_if_unset(${package}_ROOT \"${${package}_ROOT}\")\n")
  endif()
endforeach()
file(APPEND ${generated_config_dir}/ArcaneConfigCache.cmake ${_ARCANE_PACKAGES_ROOTINFO})

# ----------------------------------------------------------------------------
# Installation du modèle coverity

install(FILES extras/coverity/ArcaneModel.cc DESTINATION share/coverity)

# ----------------------------------------------------------------------------
# Indique que Arcane est trouvé pour les autres éléments du mono-dépôt qui en dépendent (Alien/ArcaneInterface)
# Donne accès aux fonctions cmake d'Arcane aux autres éléments du mono-dépôt (Alien/ArcaneInterface)
set(Arcane_FOUND YES PARENT_SCOPE)
set(ARCANE_CMAKE_COMMANDS ${ARCANESRCROOT}/cmake/Functions.cmake PARENT_SCOPE)
# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
