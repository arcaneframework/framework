# Cherche l'existence du mode embarqué de 'coreclr'

set(CoreClrEmbed_FOUND FALSE)

if (NOT DOTNET_EXEC)
  message(STATUS "[CoreClrEmbed] Disabling because DOTNET_EXEC is not found")
  return()
endif()

#get_filename_component(DOTNET_ROOT ${DOTNET_EXEC} DIRECTORY)
#message(STATUS "[CoreClrEmbed] DOTNET_ROOT = ${DOTNET_ROOT}")

# TODO: rechercher le pack en fonction de l'architecture (ARM64 ou x64) et de la version de dotnet
set(CORECLR_ARCH "linux-x64")
if (WIN32)
  set(CORECLR_ARCH "win-x64")
endif()

# ----------------------------------------------------------------------------
# On recherche la bibliothèque 'nethost' qui se trouve dans le répertoire suivant:
#
#  ${DOTNET_ROOT}/packs/Microsoft.NETCore.App.Host.${CORECLR_ARCH}.${VERSION}/runtimes/${CORECLR_ARCH}/native
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

execute_process(COMMAND ${DOTNET_EXEC} "--list-runtimes" OUTPUT_VARIABLE CORECLR_LIST_RUNTIMES_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "[.Net]: CORECLR_LIST_RUNTIMES_OUTPUT = ${CORECLR_LIST_RUNTIMES_OUTPUT}")
string(REGEX MATCH "Microsoft\.NETCore\.App ([0-9]+)\.([0-9]+)\.([a-zA-Z0-9.-]+) [\[](.*)/shared/Microsoft\.NETCore\.App[\]]"
  CORECLR_VERSION_REGEX_MATCH ${CORECLR_LIST_RUNTIMES_OUTPUT})
message(STATUS "[.Net]: MATCH_1 ${CMAKE_MATCH_1}") # Numéro de version majeur
message(STATUS "[.Net]: MATCH_2 ${CMAKE_MATCH_2}") # Numéro de version mineur
message(STATUS "[.Net]: MATCH_3 ${CMAKE_MATCH_3}") # Reste du numéro de version
message(STATUS "[.Net]: MATCH_4 ${CMAKE_MATCH_4}") # Chemin de l'installation

set(CORECLR_RUNTIME_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
set(CORECLR_RUNTIME_VERSION_FULL ${CORECLR_VERSION}.${CMAKE_MATCH_3})
set(CORECLR_RUNTIME_ROOT_PATH ${CMAKE_MATCH_4})
set(CoreClrEmbed_ROOT_PATH "${CORECLR_RUNTIME_ROOT_PATH}")

message(STATUS "[.Net]: CORECLR_RUNTIME_VERSION ${CORECLR_RUNTIME_VERSION}")
message(STATUS "[.Net]: CORECLR_RUNTIME_VERSION_FULL ${CORECLR_RUNTIME_VERSION_FULL}")
message(STATUS "[.Net]: CORECLR_RUNTIME_ROOT_PATH ${CORECLR_RUNTIME_ROOT_PATH}")

# ----------------------------------------------------------------------------

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
