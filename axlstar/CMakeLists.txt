cmake_minimum_required(VERSION 3.21.0)

file(READ "version" AXLSTAR_VERSION_STR_FULL)
string(REPLACE "_dev" "" AXLSTAR_VERSION ${AXLSTAR_VERSION_STR_FULL})
message(STATUS "AxlstarVersion = ${AXLSTAR_VERSION}")

# Positionne le projet. Il est important d'indiquer qu'on
# n'utilise aucune langage pour ne pas rechercher les compilateurs
project(Axlstar
  LANGUAGES NONE
  VERSION ${AXLSTAR_VERSION})

# TODO: a supprimer, utiliser éventuellement 'Axlstar_ROOT'
set(AXLSTAR_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Arccon "1.5.0" REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH})
include(${ARCCON_CMAKE_COMMANDS})

include(ArcconSetInstallDirs)
include(ArcconDotNet)

find_package(ArcDependencies "1.5.0" REQUIRED)

message(STATUS "ARCDEPENDENCIES_NUGET_SOURCES_DIR = '${ARCDEPENDENCIES_NUGET_SOURCES_DIR}'")

# ----------------------------------------------------------------------------
# Pour le support de .NET, on ne supporte maintenant que 'coreclr' et plus 'mono'.
# Si non spécifié, on prend 'coreclr' par défaut.
if (NOT AXLSTAR_DOTNET_RUNNER)
  set(AXLSTAR_DOTNET_RUNNER coreclr)
endif()
message(STATUS "AXLSTAR_DOTNET_RUNNER is '${AXLSTAR_DOTNET_RUNNER}'")

if (AXLSTAR_DOTNET_RUNNER STREQUAL mono)
  message(FATAL_ERROR "Building with 'mono' is no longer supported. Only 'coreclr' is supported")
elseif (AXLSTAR_DOTNET_RUNNER STREQUAL coreclr)
  # Fichier pour rechercher .NetCore.
  include(CMake/FindDotNetCore.cmake)
  set(DOTNET_RUNNER_EXEC ${DOTNET_EXEC})
  set(DOTNET_BUILDER_EXEC ${DOTNET_EXEC})
  set(DLL_EXE_EXTENSION .dll)
else()
  message(FATAL_ERROR "Invalid value '${AXLSTAR_DOTNET_RUNNER}' for AXLSTAR_DOTNET_RUNNER. Only 'coreclr' is supported")
endif()

# ----------------------------------------------------------------------------
# Regarde s'il faut prendre axl2cc ou axl2ccT4
# On utilise toujours axl2ccT4 mais maintenant les deux scripts 'axl2cc' et
# 'axl2ccT4' font la même chose.
if (WIN32)
  set(AXLSTAR_CLR_EXE_EXTENSION ".bat")
endif()
set(AXLSTAR_AXL2CC_NAME "axl2ccT4")
set(AXLSTAR_PUBLISH_DIR ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/axlstar_dlls)
set(AXLSTAR_AXL2CC_EXE ${AXLSTAR_PUBLISH_DIR}/Axlstar.Axl2ccT4${DLL_EXE_EXTENSION})
set(AXLSTAR_AXL2CC ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${AXLSTAR_AXL2CC_NAME}${AXLSTAR_CLR_EXE_EXTENSION})
set(AXLSTAR_AXL2CC_TARGETDEPEND ${AXLSTAR_AXL2CC_EXE})
# Pour compatibilité
set(ARCANE_AXL2CC ${AXLSTAR_AXL2CC})
message(STATUS "Axlstar: Axl2cc is '${AXLSTAR_AXL2CC}'")
set(AXLSTAR_AXL2CC ${AXLSTAR_AXL2CC} CACHE PATH "Script to generate C++/C# code from 'axl' files" FORCE)
set(AXLSTAR_AXL2CC_EXE ${AXLSTAR_AXL2CC_EXE} CACHE PATH ".NET exe to generate C++/C# code from 'axl' files" FORCE)
# Nom de la dépendence à 'axl2cc' pour la commande 'generateAxl'.
# Cela permet de regénérer les _axl.h si le projet Arcane.Axl.csproj est recompilé.
# Normalement, il s'agit de AXLSTAR_AXL2CC_EXE sauf si on utilise mkbundle, auquel
# cas il n'y a pas l'extension '.exe'.
set(AXLSTAR_AXL2CC_TARGETDEPEND ${AXLSTAR_AXL2CC_EXE} CACHE PATH "Dependencies for command 'generateAxl'" FORCE)

# ----------------------------------------------------------------------------

set(AXL_SOLUTION_FILE Arcane.Axl.sln)

# Il est possible de définir une variable 'AXLSTAR_MSBUILD_RESTORE_ARGS' permettant
# d'ajouter des arguments lors de la phase de restauration comme par exemple
# le chemin d'un fichier de configuration Nuget pour spécifier les sources NuGet si on n'a
# pas accès à internet.
message(STATUS "Value of AXLSTAR_MSBUILD_RESTORE_ARGS=${AXLSTAR_MSBUILD_RESTORE_ARGS}")

# Spécifie le répertoire contenant les sources nuget
if (ARCDEPENDENCIES_NUGET_SOURCES_DIR)
  set(_RESTORE_SOURCE_ARGS /p:RestoreSources="${ARCDEPENDENCIES_NUGET_SOURCES_DIR}")
endif()

set(AXLSTAR_MSBUILD_COMMON_COMMAND_ARGS /p:Configuration=Debug /p:Platform=\"Any CPU\" /p:RestoreNoCache=true ${_RESTORE_SOURCE_ARGS} /p:BinDir=${CMAKE_BINARY_DIR}/axlstar_obj/ ${AXLSTAR_MSBUILD_RESTORE_ARGS})

# Avec 'dotnet', la version du framework utilisée est 'netcoreapp6' (ou 'net6' c'est la même chose)
if (AXLSTAR_DOTNET_RUNNER STREQUAL coreclr)
  # Il semble que cette commande peut potentiellement bloquer (ne pas rendre la main à la fin de l'exécution)
  # Cela semble le cas uniquement si on utiliser 'publish' (avec 'build' cela ne pose pas de problèmes)
  # Pour éviter cela, on peut tester:
  #  1. /nodeReuse=false (pour ne pas laisser de processus dotnet qui tournent après la compilation)
  #  2. /m:1 pour compiler un seul projet à la fois
  #  3. /p:UseSharedCompilation=false (pour ne pas garder la compilateur actif entre deux compilations)
  # La méthode 1. évite de laisser des processus tourner et n'ajoute pas de temps à la compilation
  # donc on l'utilise par défaut.
  set(AXLSTAR_BUILD_COMMAND_ARGS publish /nodeReuse:false /p:UseSharedCompilation=false ${AXL_SOLUTION_FILE} /p:AxlToolsExeFramework=netcoreapp${CORECLR_VERSION} /p:PublishDir=${AXLSTAR_PUBLISH_DIR}/)
endif()
set(AXLSTAR_BUILD_COMMAND_ARGS ${AXLSTAR_BUILD_COMMAND_ARGS} ${AXLSTAR_MSBUILD_COMMON_COMMAND_ARGS})

# Pour que la compilation fonctionne avec 'coreclr' (version .NetCore 2.2), il faut
# que le répertoire de travail soit celui où se trouve la solution. Il faut donc ne
# spécifier que le nom de la solution et pas le chemin complet.
add_custom_command(OUTPUT ${AXLSTAR_AXL2CC_EXE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${DOTNET_BUILDER_EXEC}
  ARGS ${AXLSTAR_BUILD_COMMAND_ARGS}
  DEPENDS ${AXL_SOLUTION_FILE}
  )
add_custom_target(dotnet_axl_depend ALL DEPENDS ${AXLSTAR_AXL2CC_EXE})

# Créé une cible pour forcer la recompilation
add_custom_target(dotnet_axl
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${DOTNET_BUILDER_EXEC} ${AXLSTAR_BUILD_COMMAND_ARGS}
  )

configure_file(${AXLSTAR_ROOT}/Arcane.Axl/axl.xsd share/axl.xsd)
install(FILES ${AXLSTAR_ROOT}/Arcane.Axl/axl.xsd DESTINATION share)

# Copie le répertoire de publication dans le répertoire d'installation
# Attention à bien mettre les droits SETUID pour le groupe pour que
# les répertoires créés aient les bons groupes.

arccon_dotnet_install_publish_directory(DIRECTORY ${AXLSTAR_PUBLISH_DIR}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Function pour configurer un script permettant de lancer une commande C#.
# Le script préfixe la commande par le lanceur utilisé
function(axlstar_configure_runner_script file_name command_name)
  set(DLL_NAME Axlstar.Driver${DLL_EXE_EXTENSION})
  set(COMMAND_NAME ${command_name})
  set(AXLSTAR_INSTALL_BIN ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

  set(LAUNCHER_NAME launcher.sh.in)
  if (WIN32)
    set(LAUNCHER_NAME launcher.bat.in)
    set(file_name ${file_name}.bat)
  endif()

  configure_file(${LAUNCHER_NAME} ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${file_name} @ONLY)

  set(AXLSTAR_INSTALL_BIN ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
  configure_file(${LAUNCHER_NAME} ${CMAKE_BINARY_DIR}/bin_install/${file_name} @ONLY)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/bin_install/${file_name} DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

# ----------------------------------------------------------------------------

function(axlstar_configure_runner_scripts)
  axlstar_configure_runner_script(axl2ccT4 axl2ccT4)
  axlstar_configure_runner_script(axl2cc axl2cc)
  axlstar_configure_runner_script(axldoc axldoc)
  axlstar_configure_runner_script(axlcopy axlcopy)
endfunction()

# ----------------------------------------------------------------------------

axlstar_configure_runner_scripts()

# ----------------------------------------------------------------------------

include(AxlstarUtilities.cmake)

# ----------------------------------------------------------------------------

add_library(axlstar_options INTERFACE)
target_include_directories(axlstar_options INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/client_include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
arccon_install_directory(NAMES src/client_include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} PATTERN *.h)

install(TARGETS axlstar_options EXPORT AxlstarTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Installation du fichier de configuration
set(config_install_dir "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(project_config "${CMAKE_CURRENT_BINARY_DIR}/AxlstarConfig.cmake")

configure_file(CMake/AxlstarUtilities.cmake ${CMAKE_CURRENT_BINARY_DIR})
install(FILES CMake/AxlstarUtilities.cmake DESTINATION ${config_install_dir})

set(CMAKE_COMMANDS "${config_install_dir}/AxlstarUtilities.cmake")
set(EXE_AXLSTAR_AXL2CC "${CMAKE_INSTALL_BINDIR}/${AXLSTAR_AXL2CC_NAME}${AXLSTAR_CLR_EXE_EXTENSION}")
set(EXE_DIR ${CMAKE_INSTALL_BINDIR})
include(CMakePackageConfigHelpers)

set(Axlstar_project_config "${project_config}" CACHE FILEPATH "Axlstar project config" FORCE)
get_filename_component(Axlstar_config_build_dir "${project_config}" DIRECTORY CACHE)
set(Axlstar_config_install_dir "${config_install_dir}" CACHE STRING "Axlstar config install dir" FORCE)
set(Axlstar_VERSION "${Axlstar_VERSION}" CACHE STRING "Axlstar version" FORCE)

configure_package_config_file(CMake/AxlstarConfig.cmake.in
  "${project_config}"
  INSTALL_DESTINATION ${config_install_dir}
  PATH_VARS EXE_DIR EXE_AXLSTAR_AXL2CC CMAKE_COMMANDS config_install_dir
  )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/AxlstarConfigVersion.cmake
  VERSION ${Arccon_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AxlstarConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/AxlstarConfigVersion.cmake
        DESTINATION ${config_install_dir} )

install(EXPORT AxlstarTargets
        DESTINATION ${config_install_dir})

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
