# Cherche les chemins des SDKs de 'dotnet'.

set(CoreClrSDK_FOUND FALSE)

if (NOT DOTNET_EXEC)
  message(STATUS "[CoreClrSDK] Disabling because DOTNET_EXEC is not found")
  return()
endif()

# ----------------------------------------------------------------------------
# On recherche les SDKs' qui se trouvent dans le répertoire suivant:
#
#  ${DOTNET_ROOT}/sdk/${VERSION}/runtimes/${CORECLR_ARCH}/native
#
# Cela pose plusieurs problèmes:
# - tout d'abord, on ne connait pas forcément DOTNET_ROOT car il est possible que
#   l'exécutable 'dotnet' ne soit pas à la racine de l'installation (c'est le cas notamment avec
#   spack qui ajoute un répertoire 'bin' et fait un lien de nommé 'dotnet' dans ce répertoire)
# - ensuite, la valeurs de ${VERSION} ne correspond pas forcément au numéro de version donné par
#   la commande 'dotnet --version' car les versions des SDKs peuvent être différentes
#   de celle du 'dotnet' utilisé.
#
# Il est possible de récupérer ces informations en analysant la sortie de la commande
# `dotnet --list-sdks'. Cette commande va par exemple afficher les sorties suivantes:
#
# >dotnet --list-sdks
# 3.1.426 [/usr/share/dotnet/sdk]
# 6.0.412 [/usr/share/dotnet/sdk]
#
# Ce qui nous intéresse est ce qui correspond à 'Microsoft.NETCore.App'. La partie suivante est le numéro
# de version qu'on cherche (6.0.0-preview.2.21080.7) et ce qui est ensuite entre les [] permet de récupérer le
# chemin de l'installation (en enlevant '/shared/Microsoft.NETCore.App').

# Il peut y avoir plusieurs SDK installés
# Dans ce cas, la commande 'dotnet --list-sdks' va afficher plusieurs ligne avec pour chaque ligne
# un SDK. On parcours donc ces lignes et on prend le SDK qui nous correspond

execute_process(COMMAND ${DOTNET_EXEC} "--list-sdks" OUTPUT_VARIABLE CORECLR_LIST_SDKS_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "[.Net]: CORECLR_LIST_SDKS_OUTPUT = ${CORECLR_LIST_SDKS_OUTPUT}")
set(_ALL_SDKS ${CORECLR_LIST_SDKS_OUTPUT})
string(REPLACE "\n" ";" _ALL_SDKS_LIST ${_ALL_SDKS})
foreach(X ${_ALL_SDKS_LIST})
  string(REGEX MATCH "([0-9]+)\.([0-9]+)\.([a-zA-Z0-9.-]+) [\[](.*)[\]]"
    CORECLR_VERSION_REGEX_MATCH ${X})
  # MATCH:
  # 1: Numéro de version majeur
  # 2: Numéro de version mineur
  # 3: Reste du numéro de version
  # 4: Chemin de l'installation
  message(STATUS "[.Net]: MATCH '${CMAKE_MATCH_1}' '${CMAKE_MATCH_2}' '${CMAKE_MATCH_3}' '${CMAKE_MATCH_4}'")
  set(_SDK_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
  # On ne teste pas '2' et '3' car ils peuvent valoir '0' ce qui fait
  # échouer le test.
  if (CMAKE_MATCH_1 AND CMAKE_MATCH_4)
    # NOTE: la variable CORECLR_VERSION contient le numéro de version du runtime
    # utilisé. Il faut prendre la version du SDK associée à ce runtime.
    # C'est normalement le cas si 'CORECLR_VERSION==_SDK_VERSION'
    if (${_SDK_VERSION} STREQUAL ${CORECLR_VERSION})
      set(CORECLR_SDK_VERSION ${_SDK_VERSION})
      set(CORECLR_SDK_VERSION_FULL ${CORECLR_VERSION}.${CMAKE_MATCH_3})
      set(CORECLR_SDK_ROOT_PATH ${CMAKE_MATCH_4})
      message(STATUS "[.Net]: Found matching sdk version '${CORECLR_SDK_VERSION_FULL}' path='${CORECLR_SDK_ROOT_PATH}'")
    endif()
  endif()
endforeach()

set(CORECLR_SDK_PATH "${CORECLR_SDK_ROOT_PATH}/${CORECLR_SDK_VERSION_FULL}")

message(STATUS "[.Net]: CORECLR_SDK_VERSION = '${CORECLR_SDK_VERSION}'")
message(STATUS "[.Net]: CORECLR_SDK_VERSION_FULL = '${CORECLR_SDK_VERSION_FULL}'")
message(STATUS "[.Net]: CORECLR_SDK_ROOT_PATH = '${CORECLR_SDK_ROOT_PATH}'")
message(STATUS "[.Net]: CORECLR_SDK_PATH = '${CORECLR_SDK_PATH}'")

set(CoreClrSDK_FOUND TRUE)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
