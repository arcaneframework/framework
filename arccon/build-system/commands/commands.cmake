# commandes pour l'affichage 

set(COMMAND_PATH ${CMAKE_CURRENT_LIST_DIR})

include(${COMMAND_PATH}/internal/color.cmake)

include(${COMMAND_PATH}/user/logFatalError.cmake)
include(${COMMAND_PATH}/user/logWarning.cmake)
include(${COMMAND_PATH}/user/logStatus.cmake)

macro(message_separator)
  logStatus("----------------------------------------------------------------------------")
endmacro()

# commandes internes

# commandes avancées (pour écriture dees packages/metas/options/langages)
#include(${COMMAND_PATH}/advanced/loadLanguage.cmake)
#include(${COMMAND_PATH}/advanced/printLanguageInformations.cmake)
#include(${COMMAND_PATH}/advanced/loadMeta.cmake)

# commandes pour l'utilisateur (écriture de CMakeLists.txt)
include(${COMMAND_PATH}/user/RegisterPackageLibrary.cmake)
include(${COMMAND_PATH}/user/findLegacyPackage.cmake)
include(${COMMAND_PATH}/user/installDirectory.cmake)

