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
include(${COMMAND_PATH}/internal/copyAllDllFromTarget.cmake)
include(${COMMAND_PATH}/internal/copyOneDllFile.cmake)
include(${COMMAND_PATH}/internal/linkWholeArchiveLibraries.cmake)
include(${COMMAND_PATH}/internal/appendCompileOption.cmake)
include(${COMMAND_PATH}/internal/managePackagesActivation.cmake)
include(${COMMAND_PATH}/internal/manageMetasActivation.cmake)
include(${COMMAND_PATH}/internal/generateDynamicLoading.cmake)

# commandes avancées (pour écriture dees packages/metas/options/langages)
include(${COMMAND_PATH}/advanced/createOption.cmake)
include(${COMMAND_PATH}/advanced/printOptionInformations.cmake)
include(${COMMAND_PATH}/advanced/loadLanguage.cmake)
include(${COMMAND_PATH}/advanced/printLanguageInformations.cmake)
include(${COMMAND_PATH}/advanced/loadMeta.cmake)
include(${COMMAND_PATH}/advanced/enablePackage.cmake)
include(${COMMAND_PATH}/advanced/disablePackage.cmake)
include(${COMMAND_PATH}/advanced/importPackageXmlFile.cmake)
include(${COMMAND_PATH}/advanced/generatePackageXmlFile.cmake)
include(${COMMAND_PATH}/advanced/generateEclipseCDTXmlFile.cmake)


# commandes pour l'utilisateur (écriture de CMakeLists.txt)
include(${COMMAND_PATH}/user/RegisterPackageLibrary.cmake)
include(${COMMAND_PATH}/user/findLegacyPackage.cmake)
include(${COMMAND_PATH}/user/installDirectory.cmake)

