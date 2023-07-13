# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Configuration des runtimes 'mono', 'dotnet' et de l'outils 'msbuild'

message(STATUS "[.Net] Checking '.Net' configuration (using 'dependencies' for nuget packages")
message(STATUS "[.Net] Wanted runtime ARCANE_DOTNET_RUNTIME=${ARCANE_DOTNET_RUNTIME}")

# Indique si on souhaite utiliser 'coreclr' (.NetCore) ou 'mono'.
# Par défaut, on prend 'coreclr'
if (NOT ARCANE_DOTNET_RUNTIME)
  if(ARCCON_DOTNET_HAS_RUNTIME_coreclr)
    set(ARCANE_DOTNET_RUNTIME "coreclr")
  elseif (ARCCON_DOTNET_HAS_RUNTIME_mono)
    set(ARCANE_DOTNET_RUNTIME "mono")
  else()
    message(FATAL_ERROR "No '.Net' runtime available. Install '.NetCore 3.0+ (coreclr)' or 'mono 5.16+'")
  endif()
endif()

if (NOT ARCCON_DOTNET_HAS_RUNTIME_${ARCANE_DOTNET_RUNTIME})
  message(FATAL_ERROR
    "'.Net' runtime '${ARCANE_DOTNET_RUNTIME}' is not available."
    "Choose another runtime ('coreclr' or 'mono')"
    )
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Vérifie qu'on a au moins la version 3.0 de 'dotnet'.
if (ARCANE_DOTNET_RUNTIME STREQUAL coreclr)
  if(CORECLR_VERSION VERSION_LESS "3.0")
    message(FATAL_ERROR "Version of 'dotnet' ${CORECLR_VERSION} is too old. Version 3.0+ is required")
  endif()
endif()
set(ARCANE_CORECLR_VERSION ${CORECLR_VERSION} CACHE STRING ".Net coreclr version" FORCE)
set(ARCANE_CORECLR_VERSION_FULL ${CORECLR_VERSION_FULL} CACHE STRING ".Net coreclr full version" FORCE)

# ----------------------------------------------------------------------------
# Positionne les noms des exécutables et des arguments pour lancer le code
# '.Net'
if (ARCANE_DOTNET_RUNTIME STREQUAL coreclr)
  set(ARCANE_DOTNET_RUNNER_EXEC ${DOTNET_EXEC})
  set(ARCANE_DOTNET_RUNNER_ARGS)
elseif(ARCANE_DOTNET_RUNTIME STREQUAL mono)
  set(ARCANE_DOTNET_RUNNER_EXEC ${MONO_EXEC})
  set(ARCANE_DOTNET_RUNNER_ARGS --debug)
else()
  message(FATAL_ERROR "Runtime '${ARCANE_DOTNET_RUNTIME}' not handled.")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Positionne des variables pour le fichier de configuration 'ArcaneConfig.cmake'.

if (DOTNET_EXEC)
  set(ARCANE_DOTNET_RUNTIME_coreclr "${DOTNET_EXEC}")
  set(ARCANE_DOTNET_HAS_RUNTIME_coreclr TRUE)
endif()
if (MONO_EXEC)
  set(ARCANE_DOTNET_RUNTIME_mono "${MONO_EXEC}")
  set(ARCANE_DOTNET_HAS_RUNTIME_mono TRUE)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 'ArcDependencies' contient les packages NuGet pour le cas où on n'a pas
# accès à Internet.

find_package(ArcDependencies REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Arguments de msbuild communs à l'utilisation avec 'coreclr' ou 'mono'.
# La variable '${ARCANE_MSBUILD_RESTORE_ARGS}' peut être définie par l'utilisateur
# pour spécifier des arguments pour la restauration, comme par exemple
# le chemin des packages NuGet si on n'a pas accès à internet.
# `BinDir` est une propriété du fichier `Directory.Build.props` de Arcane qui indique
# l'endroit òu seront compilés les projets C#. Pour une bonne gestion
# des dépendances, il est important que tous les projets utilisent le même répertoire.
if (ARCDEPENDENCIES_NUGET_SOURCES_DIR)
  set(_RESTORE_SOURCE_ARGS /p:RestoreSources="${ARCDEPENDENCIES_NUGET_SOURCES_DIR}")
endif()
set(ARCANE_MSBUILD_FULL_RESTORE_ARGS /p:RestoreNoCache=true ${_RESTORE_SOURCE_ARGS} ${ARCANE_MSBUILD_RESTORE_ARGS})
set(ARCANE_MSBUILD_ARGS /v:Quiet /p:Platform=\"Any CPU\" /p:BinDir=${CMAKE_BINARY_DIR}/dotnet_build_dir/ ${ARCANE_MSBUILD_FRAMEWORK_ARGS} ${ARCANE_MSBUILD_FULL_RESTORE_ARGS})

if (NOT ${ARCANE_DOTNET_RUNTIME} MATCHES "coreclr|mono")
  message(FATAL_ERROR "Invalid value '${ARGS_DOTNET_RUNTIME}' for ARCANE_DOTNET_RUNTIME. Valid values are 'coreclr' or 'mono'")
endif()

set(ARCANE_DOTNET_RUNTIME ${ARCANE_DOTNET_RUNTIME} CACHE STRING "Name of dotnet runtime ('coreclr' or 'mono')" FORCE)
set(ARCANE_MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS} CACHE STRING "Required arguments of .NET 'msbuild' command" FORCE)

set(ARCANE_DOTNET_PUBLISH_RELATIVE_DIR "lib/dotnet" CACHE STRING "Relative Path to publish .Net binaries" FORCE)
set(ARCANE_DOTNET_PUBLISH_BUILD_DIR "${CMAKE_BINARY_DIR}/lib/dotnet" CACHE STRING "Path in build path to publish .Net binaries" FORCE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Répertoire contenant les projets C# après configuration
set(ARCANE_CSHARP_PROJECT_PATH ${CMAKE_BINARY_DIR}/share/csproj CACHE FILEPATH "Directory containing C# projects" FORCE)
file(MAKE_DIRECTORY ${ARCANE_CSHARP_PROJECT_PATH})

# Fichier contenant les propriétés communes à tous nos projets pour 'msbuild'.
# Notamment, ce fichier permet de gérer les compilations en dehors des sources.
configure_file(Directory.Build.props ${ARCANE_CSHARP_PROJECT_PATH} COPYONLY)
configure_file(Directory.Build.targets ${ARCANE_CSHARP_PROJECT_PATH} COPYONLY)
configure_file(CommonExe.props ${ARCANE_CSHARP_PROJECT_PATH} COPYONLY)

message(STATUS "ARCANE_MSBUILD_ARGS=${ARCANE_MSBUILD_ARGS}")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Création des scripts shell pour lancer les exécutables '.Net'.
# Ne pas utiliser de fichiers mais générer directement le source correspondant
# car tous ces fichiers sont similaires.
macro(arcane_configure_script2 path name lib_or_bin)
  set(subdir_bin bin)
  set(subdir_lib lib)
  if(${lib_or_bin} STREQUAL "bin")
    set(__locpath ${subdir_bin})
  else()
    set(__locpath ${subdir_lib}/dotnet)
  endif()
  if(VERBOSE)
    message(STATUS "Configure2 name=${name} path=${path} __locpath=${__locpath} '${lib_or_bin}'")
  endif()

  # On génère deux scripts. Un pour l'espace de compilation et un pour
  # l'espace d'installation
  set(ARCANE_INSTALL_PATH ${CMAKE_BINARY_DIR})
  set(ARCANE_INSTALL_SHR ${CMAKE_BINARY_DIR}/share)
  set(ARCANE_INSTALL_LIB ${CMAKE_BINARY_DIR}/${subdir_lib})
  set(ARCANE_INSTALL_BIN ${CMAKE_BINARY_DIR}/${subdir_bin})
  set(_ARCANE_DOTNET_PUBLISH_DIR ${ARCANE_DOTNET_PUBLISH_BUILD_DIR})
  configure_file(${path}/${name} ${CMAKE_BINARY_DIR}/${__locpath}/${name} @ONLY)

  set(ARCANE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
  set(ARCANE_INSTALL_SHR ${CMAKE_INSTALL_PREFIX}/share)
  set(ARCANE_INSTALL_LIB ${CMAKE_INSTALL_PREFIX}/${subdir_lib})
  set(ARCANE_INSTALL_BIN ${CMAKE_INSTALL_PREFIX}/${subdir_bin})
  set(_ARCANE_DOTNET_PUBLISH_DIR ${CMAKE_INSTALL_PREFIX}/${ARCANE_DOTNET_PUBLISH_RELATIVE_DIR})
  configure_file(${path}/${name} ${CMAKE_BINARY_DIR}/${__locpath}_install/${name} @ONLY)

  unset(_ARCANE_DOTNET_PUBLISH_DIR)
endmacro()

macro(arcane_dotnet_configure_script path name lib_or_bin)
  set(_final_name ${name})
  if(${lib_or_bin} STREQUAL "bin")
    if (WIN32)
      # Sous windows, ajoute l'extension '.bat' pour les exécutables
      set(_final_name ${name}.bat)
      message(STATUS "WIN32! name=${_final_name}")
    endif()
  endif()

  message(STATUS "Configure name=${_final_name} path=${path} lib_or_bin=${lib_or_bin}")
  arcane_configure_script2(${path} ${_final_name} ${lib_or_bin})

  set(ARCANE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
  set(ARCANE_INSTALL_SHR ${CMAKE_INSTALL_PREFIX}/share)
  set(ARCANE_INSTALL_LIB ${CMAKE_INSTALL_PREFIX}/lib)
  set(ARCANE_INSTALL_BIN ${CMAKE_INSTALL_PREFIX}/bin)
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Macro pour configurer un script pour encapsuler un appel 'dotnet'
# 'name' est le nom du script
# 'tool_name' est le nom de la commande (premier argument du script)
macro(arcane_dotnet_configure_exe_script name tool_name)
  message(STATUS "Configure name=${name} tool_name=${tool_name}")
  set(_final_name ${name})
  if (WIN32)
    # Sous windows, ajoute l'extension '.bat' pour les exécutables
    set(_final_name ${name}.bat)
  endif()

  set(__locpath bin)
  if(VERBOSE)
    message(STATUS "Configure2 name=${name} tool_name=${tool_name} __locpath=${__locpath}")
  endif()
  set(__shell_name arcane_tool_generic_shell.in)
  if (WIN32)
    set(__shell_name arcane_tool_generic_shell.bat.in)
  endif()
  set(_ARCANE_TOOL_NAME ${tool_name})
  # Le script utilise les variables suivantes:
  # - ARCANE_DOTNET_PUBLISH_RELATIVE_DIR
  # - ARCANE_DOTNET_RUNNER_EXEC
  # - ARCANE_DOTNET_RUNNER_ARGS
  # - ARCANE_RELATIVE_LIB
  # On génère deux scripts. Un pour l'espace de compilation et un pour
  # l'espace d'installation mais les deux sont identiques.
  set(ARCANE_RELATIVE_LIB lib)
  configure_file(${__shell_name} ${CMAKE_BINARY_DIR}/${__locpath}/${_final_name} @ONLY)
  configure_file(${__shell_name} ${CMAKE_BINARY_DIR}/${__locpath}_install/${_final_name} @ONLY)
  unset(_ARCANE_TOOL_NAME)
endmacro()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Cible pour les outils C#
#
# TODO: séparer la cible 'Arcane.Utils' de 'AllTools' car elle est utilisée
# par le wrapper et il n'est pas nécessaire que tout 'AllTools' ait été
# compilé pour le wrapper swig.
#

# Conserve le nom du projet 'Arcane.Utils.csproj' dans une variable
# Cela est utilisé par le wrapper qui doit référencer ce projet
set(ARCANE_DOTNET_ARCANE_UTILS_CSPROJ_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Arcane.Utils/Arcane.Utils.csproj"
  CACHE PATH "Path of 'Arcane.Utils.csproj" FORCE)

arccon_add_csharp_target(dotnet_arcane_utils
  DOTNET_RUNTIME ${ARCANE_DOTNET_RUNTIME}
  BUILD_DIR ${ARCANE_DOTNET_PUBLISH_BUILD_DIR}
  ASSEMBLY_NAME Arcane.Utils.dll
  PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}
  PROJECT_NAME Arcane.Utils/Arcane.Utils.csproj
  MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS}
  PACK
  )

# TODO: renommer la cible 'dotnet_xbuild' en 'dotnet_tools'.
#
arccon_add_csharp_target(dotnet_xbuild
  DOTNET_RUNTIME ${ARCANE_DOTNET_RUNTIME}
  BUILD_DIR ${ARCANE_DOTNET_PUBLISH_BUILD_DIR}
  ASSEMBLY_NAME Arcane.ExecDrivers.dll
  PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}
  PROJECT_NAME AllTools.sln
  MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS}
  DOTNET_TARGET_DEPENDS dotnet_arcane_utils
  )

# Indique qu'il faut recopier à l'installation tout le répertoire où a été
# publié le code compilé.
arccon_dotnet_install_publish_directory(DIRECTORY ${ARCANE_DOTNET_PUBLISH_BUILD_DIR}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

arccon_dotnet_install_publish_directory(DIRECTORY ${CMAKE_BINARY_DIR}/nupkgs
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share)

# ----------------------------------------------------------------------------
set(ARCANE_EXECDRIVER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Arcane.ExecDrivers)
arcane_dotnet_configure_script(${ARCANE_EXECDRIVER_SOURCE_DIR} Arcane.ExecDrivers.dll.config.json lib)

arcane_dotnet_configure_exe_script(arcane_partition_mesh "mesh_utils partition")
arcane_dotnet_configure_exe_script(arcane_convert_mesh "mesh_utils convert")
arcane_dotnet_configure_exe_script(arcane_curves "curve_utils")
arcane_dotnet_configure_exe_script(arcane_templates "template")
arcane_dotnet_configure_exe_script(arcane_test_driver "test_driver")
arcane_dotnet_configure_exe_script(arcane_run "launcher")
arcane_dotnet_configure_exe_script(arcane_dotnet_compile "dotnet_compile")

set(_SCRIPT_EXTENSION "")
if (WIN32)
  set(_SCRIPT_EXTENSION ".bat")
endif()
install(PROGRAMS
  ${ARCANEBUILDROOT}/bin_install/arcane_partition_mesh${_SCRIPT_EXTENSION}
  ${ARCANEBUILDROOT}/bin_install/arcane_convert_mesh${_SCRIPT_EXTENSION}
  ${ARCANEBUILDROOT}/bin_install/arcane_curves${_SCRIPT_EXTENSION}
  ${ARCANEBUILDROOT}/bin_install/arcane_templates${_SCRIPT_EXTENSION}
  ${ARCANEBUILDROOT}/bin_install/arcane_run${_SCRIPT_EXTENSION}
  ${ARCANEBUILDROOT}/bin_install/arcane_dotnet_compile${_SCRIPT_EXTENSION}
  DESTINATION bin)

# A revoir pour Win32
message(STATUS "TEST WHERE INSTALL: ${CMAKE_BINARY_DIR}/${ARCANE_DOTNET_PUBLISH_RELATIVE_DIR}_install/Arcane.ExecDrivers.dll.config.json")
install(FILES ${CMAKE_BINARY_DIR}/${ARCANE_DOTNET_PUBLISH_RELATIVE_DIR}_install/Arcane.ExecDrivers.dll.config.json
  DESTINATION ${ARCANE_DOTNET_PUBLISH_RELATIVE_DIR} RENAME Arcane.ExecDrivers.dll.install.config.json)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion du wrapper 'Swig'.

add_subdirectory(wrapper)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Génère un fichier de configuration pour les informations pour utiliser '.Net'.
configure_file(ArcaneDotNet.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ArcaneDotNet.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ArcaneDotNet.cmake DESTINATION ${ARCANE_CMAKE_CONFIG_DIR})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
