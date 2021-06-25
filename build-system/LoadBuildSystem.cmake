cmake_minimum_required(VERSION 3.4.1)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# outillage cmake
include(CMakeParseArguments)
include(FindPackageHandleStandardArgs)
include(GenerateExportHeader)  
include(CheckCXXCompilerFlag)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

get_filename_component(BUILD_SYSTEM_PATH ${CMAKE_CURRENT_LIST_FILE} PATH)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
 
find_program(PKGLIST_LOADER 
  NAMES PkgListLoader.exe 
  PATHS ${BUILD_SYSTEM_PATH}/bin
  NO_DEFAULT_PATH
  )
find_program(WHOLEARCHIVE_VCPROJ_TOOL
  NAMES WholeArchiveVCProj.exe
  HINTS ${BUILD_SYSTEM_PATH}/bin
  NO_DEFAULT_PATH	
  )
find_program(CMAKELIST_GENERATOR
  NAMES CMakeListGenerator.exe
  HINTS ${BUILD_SYSTEM_PATH}/bin
  NO_DEFAULT_PATH 
  )
find_program(ECLIPSECDT_GENERATOR
  NAMES EclipseCDTSettings.exe
  HINTS ${BUILD_SYSTEM_PATH}/bin
  NO_DEFAULT_PATH 
  )
if(WIN32)
  find_program(WINDOWS_PATH_RESOLVER_TOOL
    NAMES WindowsPathResolver.exe
    HINTS ${BUILD_SYSTEM_PATH}/bin
    NO_DEFAULT_PATH 
    )
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
 
if(NOT CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install/${PROJECT_VERSION} 
      CACHE PATH "Default install path" FORCE)
endif()

# compilation dans les sources interdites
if(${PROJECT_BINARY_DIR} STREQUAL ${PROJECT_SOURCE_DIR})
  logFatalError("You can not do in-source compilation. You have to build in a directory distinct from the source directory")
endif()

# où sont placés le exe  et les libs
# NB: en fait, ne retire pas les répertoires des configurations avec Visual/XCode
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Répertoire de copie des dlls pour windows
if(NOT BUILDSYSTEM_DLL_COPY_DIRECTORY)
  set(BUILDSYSTEM_DLL_COPY_DIRECTORY ${PROJECT_BINARY_DIR}/bin/sys_dlls)
endif()

file(MAKE_DIRECTORY ${BUILDSYSTEM_DLL_COPY_DIRECTORY})

if(BUILDSYSTEM_NO_CONFIGURATION_OUTPUT_DIRECTORY)
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} CONFIG) 
    # les dlls/libs/exe ne sont pas placés dans des répertoires de configuration
    # Pour Visual/XCode, pas de chemin bin/Release ou lib/Release (par exemple)
    # NB: Sous linux, cela ne change rien
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG} ${PROJECT_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG} ${PROJECT_BINARY_DIR}/bin)
  endforeach()
endif()

# rpath 
SET(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

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

set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)

# pas de suffixe aux executables
# NOTE GG: il ne faut supprimer le suffixe sous Windows car cela planter
# pas mal de tests pour trouver les packages et cela risque de poser
# d'autre problèmes.
#set(CMAKE_EXECUTABLE_SUFFIX "")

# inclusion automatique des répertoires
# CMAKE_CURRENT_BINARY_DIR et CMAKE_CURRENT_SOURCE_DIR
# dans les chemins pour la compilation
# NOTE GG: il ne faut pas inclure par défaut le répertoire courant aux
# sources car cela pose des problèmes si certains fichiers ont le même
# nom que ceux d'autres packages. Si nécessaire, si pour une cible cela
# est nécessaire elle doit l'ajouter elle-même
#if(NOT CMAKE_INCLUDE_CURRENT_DIR)
#  set(CMAKE_INCLUDE_CURRENT_DIR ON)
#endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message_separator()
logStatus(" **  Project name      : ${BoldRed}${PROJECT_NAME}${ColourReset}")
logStatus(" **          version   : ${BoldRed}${PROJECT_VERSION}${ColourReset}")
message_separator()
logStatus(" **  System name       : ${CMAKE_SYSTEM_NAME}")
logStatus(" **         version    : ${CMAKE_SYSTEM_VERSION}")
logStatus(" **         processor  : ${CMAKE_SYSTEM_PROCESSOR}")
if(EXISTS "/etc/redhat-release")
  file(READ "/etc/redhat-release" REDHAT_RELEASE)
  string(REPLACE "\n" "" REDHAT_RELEASE ${REDHAT_RELEASE})
  logStatus(" **         vendor     : ${REDHAT_RELEASE}")
endif()
message_separator()
site_name(BUILD_SITE_NAME)
logStatus(" ** Build site name    : ${BUILD_SITE_NAME}")
message_separator()
logStatus(" **  Generator         : ${CMAKE_GENERATOR}")
message_separator()
logStatus(" **  Build System path : ${BUILD_SYSTEM_PATH}")
logStatus(" **       Install path : ${CMAKE_INSTALL_PREFIX}")
logStatus(" **     Dlls copy path : ${BUILDSYSTEM_DLL_COPY_DIRECTORY}")
if(BUILDSYSTEM_NO_CONFIGURATION_OUTPUT_DIRECTORY)
  logStatus(" ** No configuration in output directories lib/bin")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# désactivation des packages par -DDisablePackages=xx;yy;zz...
# et activation par -DEnablePackages=xx;yy;zz...
managePackagesActivation()

# activation des métas par -DEnableMetas=xx;yy;zz...
manageMetasActivation()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
