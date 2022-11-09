# Cherche l'existence du mode embarqué de 'coreclr'

set(CoreClrEmbed_FOUND FALSE)

if (NOT DOTNET_EXEC)
  message(STATUS "[CoreClrEmbed] Disabling because DOTNET_EXEC is not found")
  return()
endif()

# ----------------------------------------------------------------------------
# Positionne une architecture par défaut.
# On essaie par la suite de détecter automatiquement cette valeur mais c'est plus prudent
# de mettre une valeur par défaut au cas où.
set(CORECLR_ARCH "linux-x64")
set(CORECLR_SUBARCH "x64")
if (WIN32)
  set(CORECLR_ARCH "win-x64")
endif()
if (UNIX)
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CORECLR_ARCH "linux-arm64")
    set(CORECLR_SUBARCH "arm64")
  endif()
endif()

# ----------------------------------------------------------------------------
# On recherche la bibliothèque 'nethost' qui se trouve dans le répertoire suivant:
#
#  ${DOTNET_ROOT}/packs/Microsoft.NETCore.App.Host.${CORECLR_ARCH}/${VERSION}/runtimes/${CORECLR_ARCH}/native
#
# Cela pose plusieurs problèmes:
# - tout d'abord, on ne connait pas forcément DOTNET_ROOT car il est possible que
#   l'exécutable 'dotnet' ne soit pas à la racine de l'installation (c'est le cas notamment avec
#   spack qui ajoute un répertoire 'bin' et fait un lien de nommé 'dotnet' dans ce répertoire)
# - ensuite, la valeurs de ${VERSION} ne correspond pas au numéro de version donné par
#   la commande 'dotnet --version'. Notamment, la troisième composante n'est pas la même
#   (par exemple, pour dotnet version 3.1.4, `dotnet --version` retourne 3.1.300.
#
# Il est possible de récupérer ces informations en analysant la sortie de la commande
# `dotnet --list-runtimes'. Cette commande va par exemple afficher les sorties suivantes:
#
# >dotnet --list-runtimes
# Microsoft.AspNetCore.App 6.0.0-preview.2.21079.10 [/ccc/dotnet/6.0.100-preview.1.21101.5/shared/Microsoft.AspNetCore.App]
# Microsoft.NETCore.App 6.0.0-preview.2.21080.7 [/ccc/dotnet/6.0.100-preview.1.21101.5/shared/Microsoft.NETCore.App]
#
# Ce qui nous intéresse est ce qui correspond à 'Microsoft.NETCore.App'. La partie suivante est le numéro
# de version qu'on cherche (6.0.0-preview.2.21080.7) et ce qui est ensuite entre les [] permet de récupérer le
# chemin de l'installation (en enlevant '/shared/Microsoft.NETCore.App').

# Il peut y avoir plusieurs SDK et runtimes installés
# Dans ce cas, la commande 'dotnet --list-runtimes' va afficher plusieurs ligne avec pour chaque ligne
# un runtime. On parcours donc ces lignes et on regarde le runtime qui nous correspond

execute_process(COMMAND ${DOTNET_EXEC} "--list-runtimes" OUTPUT_VARIABLE CORECLR_LIST_RUNTIMES_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND "/bin/cat" "${CMAKE_CURRENT_LIST_DIR}/do_win.txt" OUTPUT_VARIABLE CORECLR_LIST_RUNTIMES_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "[.Net]: CORECLR_LIST_RUNTIMES_OUTPUT = ${CORECLR_LIST_RUNTIMES_OUTPUT}")
set(_ALL_RUNTIMES ${CORECLR_LIST_RUNTIMES_OUTPUT})
string(REPLACE "\n" ";" _ALL_RUNTIMES_LIST ${_ALL_RUNTIMES})
foreach(X ${_ALL_RUNTIMES_LIST})
  string(REGEX MATCH "Microsoft\.NETCore\.App ([0-9]+)\.([0-9]+)\.([a-zA-Z0-9.-]+) [\[](.*)Microsoft\.NETCore\.App[\]]"
    CORECLR_VERSION_REGEX_MATCH ${X})
  # MATCH:
  # 1: Numéro de version majeur
  # 2: Numéro de version mineur
  # 3: Reste du numéro de version
  # 4: Chemin de l'installation
  message(STATUS "[.Net]: MATCH '${CMAKE_MATCH_1}' '${CMAKE_MATCH_2}' '${CMAKE_MATCH_3}' '${CMAKE_MATCH_4}'")
  set(_RUNTIME_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
  # On ne teste pas '2' et '3' car ils peuvent valoir '0' ce qui fait
  # échouer le test.
  if (CMAKE_MATCH_1 AND CMAKE_MATCH_4)
    # NOTE: la variable CORECLR_VERSION contient le numéro de version du runtime
    # utilisé. Il faut prendre la version du SDK associée à ce runtime.
    # C'est normalement le cas si 'CORECLR_VERSION==_RUNTIME_VERSION'
    if (${_RUNTIME_VERSION} STREQUAL ${CORECLR_VERSION})
      set(CORECLR_RUNTIME_VERSION ${_RUNTIME_VERSION})
      set(CORECLR_RUNTIME_VERSION_FULL ${CORECLR_VERSION}.${CMAKE_MATCH_3})
      set(CORECLR_RUNTIME_ROOT_PATH ${CMAKE_MATCH_4})
      message(STATUS "[.Net]: Found matching runtime version '${CORECLR_RUNTIME_VERSION_FULL}' path='${CORECLR_RUNTIME_ROOT_PATH}'")
    endif()
  endif()
endforeach()
# Enlève 'shared' CORECLR_RUNTIME_ROOT_PATH.
# Il faut le faire avec les commandes CMake car sous windows le séparateur
# de fichier n'est pas '/'
if (CORECLR_RUNTIME_ROOT_PATH)
  get_filename_component(CORECLR_RUNTIME_ROOT_PATH ${CORECLR_RUNTIME_ROOT_PATH} DIRECTORY)
endif()
set(CoreClrEmbed_ROOT_PATH "${CORECLR_RUNTIME_ROOT_PATH}")

message(STATUS "[.Net]: CORECLR_RUNTIME_VERSION = '${CORECLR_RUNTIME_VERSION}'")
message(STATUS "[.Net]: CORECLR_RUNTIME_VERSION_FULL = '${CORECLR_RUNTIME_VERSION_FULL}'")
message(STATUS "[.Net]: CORECLR_RUNTIME_ROOT_PATH = '${CORECLR_RUNTIME_ROOT_PATH}'")

message(STATUS "[.Net]: CORECLR_SUBARCH = '${CORECLR_SUBARCH}'")

# ----------------------------------------------------------------------------
# Détermine l'architecture associé au runtime
#
# En général, il s'agit d'une chaîne de caractère comme 'linux-x64' ou 'win-x64'.
# Cependant il peut y avoir d'autres valeurs. Notamment sur ubuntu lorsqu'on utiliser
# le package 'dotnet6', l'architecture est alors 'ubuntu.22.04-x64'.
# On essaie donc de détecter automatiquement l'architecture si possible.
file(GLOB _CORECLR_HOST_ARCH_PATH "${CORECLR_RUNTIME_ROOT_PATH}/packs/Microsoft.NETCore.App.Host.*-${CORECLR_SUBARCH}")
message(STATUS "[CoreClrEmbed] _CORECLR_HOST_ARCH_PATH = '${_CORECLR_HOST_ARCH_PATH}'")
if (_CORECLR_HOST_ARCH_PATH)
  get_filename_component(_CORECLR_HOST_ARCH_FILENAME ${_CORECLR_HOST_ARCH_PATH} NAME)
  message(STATUS "[CoreClrEmbed] _CORECLR_HOST_ARCH_FILENAME = '${_CORECLR_HOST_ARCH_FILENAME}'")
  string(REPLACE "Microsoft.NETCore.App.Host." "" _CORECLR_COMPUTED_ARCH "${_CORECLR_HOST_ARCH_FILENAME}")
  message(STATUS "[CoreClrEmbed] _CORECLR_COMPUTED_ARCH = '${_CORECLR_COMPUTED_ARCH}'")
  if (_CORECLR_COMPUTED_ARCH)
    set(CORECLR_ARCH "${_CORECLR_COMPUTED_ARCH}")
    message(STATUS "[CoreClrEmbed] Set CORECLR_ARCH to '${_CORECLR_COMPUTED_ARCH}'")
  endif()
endif()

set(CORECLR_HOST_BASE_PATH ${CORECLR_RUNTIME_ROOT_PATH}/packs/Microsoft.NETCore.App.Host.${CORECLR_ARCH}/${CORECLR_RUNTIME_VERSION_FULL})
message(STATUS "[CoreClrEmbed] searching runtime path '${CORECLR_HOST_BASE_PATH}'")
file(GLOB _CORECLR_FOUND_PATH ${CORECLR_HOST_BASE_PATH})
message(STATUS "[CoreClrEmbed] _CORECLR_FOUND_PATH = ${_CORECLR_FOUND_PATH}")
if (_CORECLR_FOUND_PATH)
  set(CORECLR_NETHOST_ROOT "${_CORECLR_FOUND_PATH}/runtimes/${CORECLR_ARCH}/native")
endif()

find_library(CoreClrEmbed_LIBRARY nethost PATHS
  ${CORECLR_NETHOST_ROOT}
  )
find_path(CoreClrEmbed_INCLUDE_DIR nethost.h PATHS
  ${CORECLR_NETHOST_ROOT}
  )

message(STATUS "[CoreClrEmbed] CoreClrEmbed_LIBRARY = ${CoreClrEmbed_LIBRARY}")
message(STATUS "[CoreClrEmbed] CoreClrEmbed_INCLUDE_DIR = ${CoreClrEmbed_INCLUDE_DIR}")

if (CoreClrEmbed_INCLUDE_DIR AND CoreClrEmbed_LIBRARY)
  set(CoreClrEmbed_FOUND TRUE)
  set(CoreClrEmbed_LIBRARIES ${CoreClrEmbed_LIBRARY} )
  set(CoreClrEmbed_INCLUDE_DIRS ${CoreClrEmbed_INCLUDE_DIR})
endif()

arccon_register_package_library(CoreClrEmbed CoreClrEmbed)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
