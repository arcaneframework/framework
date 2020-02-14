
# mettre ici tous les packages chargés par défaut
# on cherche des fichiers Find**.cmake

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/packages)

loadPackage(NAME Arccon ESSENTIAL)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH_M})

loadPackage(NAME Axlstar ESSENTIAL)

# En attendant mieux...
include(${ARCCON_MODULE_PATH_M}/../ArcconDotNet.cmake)
include(${ARCCON_MODULE_PATH_M}/../ArcconSetInstallDirs.cmake)
include(${ARCCON_MODULE_PATH_M}/../commands/user/findLegacyPackage.cmake)
include(${ARCCON_MODULE_PATH_M}/../commands/user//RegisterPackageLibrary.cmake)

if (NOT WIN32)
    loadPackage(NAME Mono ESSENTIAL)
endif ()

#loadPackage(NAME DotNet ESSENTIAL)
include(${CMAKE_SOURCE_DIR}/cmake/build-system/packages/FindDotNet.cmake)
loadPackage(NAME Glib ESSENTIAL)

add_library(glib ALIAS arcconpkg_Glib)