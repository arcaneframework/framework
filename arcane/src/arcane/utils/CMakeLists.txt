include(srcs.cmake)

arcane_add_library(arcane_utils
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/utils
  FILES ${ARCANE_SOURCES}
  )

arcane_add_arccon_packages(arcane_utils PRIVATE ${PKGS})

target_compile_definitions(arcane_utils PRIVATE ARCANE_COMPONENT_arcane_utils)

target_include_directories(arcane_utils PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)
target_include_directories(arcane_utils PUBLIC $<BUILD_INTERFACE:${ARCANE_SRC_PATH}> $<INSTALL_INTERFACE:include>)

# 'arcane_build_compile_flags' n'est utilisé que lors de la compilation
# de la bibliothèque mais ne doit pas être exportée.
target_link_libraries(arcane_utils PUBLIC $<BUILD_INTERFACE:arcane_build_compile_flags> arcane_export_compile_flags)

# TODO A supprimer car cette cible ne doit pas utiliser des '.h' de 'arcane_core'
target_include_directories(arcane_utils PUBLIC $<BUILD_INTERFACE:${ARCANE_COMPATIBILITY_DIRECTORY}>)

# Ajoute dépendances provenant de 'Arccore'
target_link_libraries(arcane_utils PUBLIC
  Arccore::arccore_trace
  Arccore::arccore_concurrency
  Arccore::arccore_collections
  Arccore::arccore_common
  Arccore::arccore_base
  )
arcane_add_arccon_packages(arcane_utils PUBLIC Glib)

# ----------------------------------------------------------------------------
# Indique qu'il faut le C++17 par défaut ou sinon celui spécifié par
# Arccore
if (DEFINED ARCCORE_CXX_STANDARD)
  target_compile_features(arcane_utils PUBLIC cxx_std_${ARCCORE_CXX_STANDARD})
else()
  target_compile_features(arcane_utils PUBLIC cxx_std_17)
endif()

# ----------------------------------------------------------------------------
# Regarde si les 'hooks' mémoire de 'malloc.h' de la glibc sous linux
# sont disponibles
if (${ARCANE_HAS_MALLOC_HOOKS})
  target_compile_definitions(arcane_utils PRIVATE ARCANE_CHECK_MEMORY_USE_MALLOC_HOOK)
endif()

# ----------------------------------------------------------------------------

arcane_register_library(arcane_utils)

# ----------------------------------------------------------------------------

if (GTEST_FOUND)
  add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
