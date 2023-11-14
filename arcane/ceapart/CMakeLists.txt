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

# A supprimer mais en attendant permet de compiler avec IntelMPI
if (ARCANE_WANT_MPI_DISTRIB MATCHES "intelmpi")
  set(DISABLE_SLOOP_AND_TRILINOS YES)
  include(cmake/FindItac.cmake)
endif ()
if(ARCANE_USE_MPC)
  set(DISABLE_SLOOP_AND_TRILINOS YES)
endif()

# ----------------------------------------------------------------------------

include(cmake/VT100.cmake)
if(NOT WIN32)
  # Lima a besoin de LM et F90 donc il faut qu'ils soient
  # recherchés avan lui
  include(cmake/FindItac.cmake)
  include(cmake/FindMathlink.cmake)

  if(NOT DISABLE_SLOOP_AND_TRILINOS)
    include(cmake/FindSuperLU.cmake)
  endif()
endif()

if(NOT WIN32 AND MATHLINK_FOUND)
  add_subdirectory(src/arcane/mathlink)
endif()

add_subdirectory(src/arcane/cea)
add_subdirectory(src/arcane/geometric)
add_subdirectory(src/arcane/meshutils)
add_subdirectory(src/arcane/hyoda)
add_subdirectory(src/arcane/aleph/sloop)

# ----------------------------------------------------------------------------
# Ajoute des tests.
# Ils doivent être après les autres cibles car ils peuvent en dépendre.
if (ARCANE_ENABLE_TESTS)
  add_subdirectory(src/arcane/tests)
  add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Ce fichier contient les informations de configuration de l'installation 'cea'.
configure_file(ArcaneCea.config.in ${ARCANE_DOTNET_PUBLISH_BUILD_DIR}/ArcaneCea.config @ONLY)
install(FILES ${ARCANE_DOTNET_PUBLISH_BUILD_DIR}/ArcaneCea.config DESTINATION ${ARCANE_DOTNET_PUBLISH_RELATIVE_DIR})

# ----------------------------------------------------------------------------
# Cibles pour generer la documentation spécifique CEA

function(CEA_GENERATE_DOC_TARGET docname)
  set(DOC_OUTPUT_DIR ${CMAKE_BINARY_DIR})
  configure_file(doc/${docname}.doxyfile ${CMAKE_BINARY_DIR}/share/${docname}.doxyfile @ONLY)
  add_custom_target(${docname} COMMAND ${DOXYGEN} ${CMAKE_BINARY_DIR}/share/${docname}.doxyfile)
endfunction()

message(STATUS "CEA_DOXYGEN = ${DOXYGEN}")
if(DOXYGEN)
  CEA_GENERATE_DOC_TARGET(ceadevdoc)
  CEA_GENERATE_DOC_TARGET(ceauserdoc)
endif()

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
