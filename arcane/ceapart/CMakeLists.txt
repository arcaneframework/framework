# Ce repertoire ne doit pas etre utilise directement mais inclut depuis
# le repertoire de base de Arcane
if(NOT ARCANE_CEA_SOURCE_PATH)
  set(ARCANE_CEA_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
endif()
set(ARCANECEA_SRC_PATH ${ARCANE_CEA_SOURCE_PATH}/src CACHE PATH "Arcane CEA 'src' path" FORCE)

# Cette variable est utilisée par ConfigGenerator.
# Elle ne doit pas être utilisée ailleurs.
set(ARCANEBUILDROOT ${CMAKE_CURRENT_BINARY_DIR}/src)
message(STATUS "ARCANEBUILDROOT for CEA = ${ARCANEBUILDROOT}")
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/share)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/share/axl)

# ----------------------------------------------------------------------------

include(cmake/VT100.cmake)
if(NOT WIN32)
  include(cmake/FindItac.cmake)
endif()

add_subdirectory(src/arcane/cea)
add_subdirectory(src/arcane/geometric)
add_subdirectory(src/arcane/meshutils)

# ----------------------------------------------------------------------------
# Ce fichier contient les informations de configuration de l'installation 'cea'.
configure_file(ArcaneCea.config.in ${ARCANE_DOTNET_PUBLISH_BUILD_DIR}/ArcaneCea.config @ONLY)
install(FILES ${ARCANE_DOTNET_PUBLISH_BUILD_DIR}/ArcaneCea.config DESTINATION ${ARCANE_DOTNET_PUBLISH_RELATIVE_DIR})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Exporte dans le CMakeLists.txt de Arcane les informations des packages.
# A terme, cela devrait être supprimé lorsqu'on utilisera les cibles importées
# partout.

macro(arcanecea_export_package pkg)
  set(${pkg}_FOUND ${${pkg}_FOUND} PARENT_SCOPE)
  set(${pkg}_LIBRARIES ${${pkg}_LIBRARIES} PARENT_SCOPE)
  set(${pkg}_INCLUDE_DIRS ${${pkg}_INCLUDE_DIRS} PARENT_SCOPE)
endmacro()

foreach(pkg in ${ARCANE_ADDITIONAL_EXTERNAL_PACKAGES})
  arcanecea_export_package(${pkg})
endforeach()

set(ARCANE_LIBRARIES ${ARCANE_LIBRARIES} PARENT_SCOPE)
set(ARCANE_ADDITIONAL_EXTERNAL_PACKAGES ${ARCANE_ADDITIONAL_EXTERNAL_PACKAGES} PARENT_SCOPE)

message(STATUS "(CEA) ARCANE_LIBRARIES: ${ARCANE_LIBRARIES}")
message(STATUS "(CEA) ARCANE_ADDITIONAL_EXTERNAL_PACKAGES ${ARCANE_ADDITIONAL_EXTERNAL_PACKAGES}")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
