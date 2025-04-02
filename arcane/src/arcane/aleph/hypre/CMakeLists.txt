set(PKGS Hypre MPI)
arcane_find_package(Hypre)
arcane_find_package(MPI)
if (NOT Hypre_FOUND OR NOT MPI_FOUND)
  return()
endif()

# Recherche le numéro de version de Hypre.
# Il est dans le fichier 'Hypre_config.h', sous la forme:
# #define HYPRE_RELEASE_VERSION "2.20.0"
# NOTE: on pourrait aussi utilier HYPRE_RELEASE_NUMBER. Il faudrait
# vérifier depuis quelle version de Hypre cette macro est disponible
#set(Hypre_INCLUDE_DIRS $ENV{HYPRE_ROOT}/include)
find_file(_HYPRE_CONFIG_PATH NAMES HYPRE_config.h PATHS ${Hypre_INCLUDE_DIRS} REQUIRED)
file(STRINGS ${_HYPRE_CONFIG_PATH} _HYPRE_VERSION_STR REGEX HYPRE_RELEASE_VERSION)
string(REGEX MATCH "#define HYPRE_RELEASE_VERSION \"([0-9]+)\.([0-9]+)\.([0-9]+)\"" HYPRE_VERSION_REGEX_MATCH ${_HYPRE_VERSION_STR})
set(ARCANE_HYPRE_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
message(STATUS "Hypre_VERSION = ${ARCANE_HYPRE_VERSION}")

include(srcs.cmake)

arcane_add_library(arcane_aleph_hypre
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/hypre
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_aleph_hypre PRIVATE ${PKGS})
target_compile_definitions(arcane_aleph_hypre PRIVATE ARCANE_COMPONENT_arcane_aleph_hypre)

target_link_libraries(arcane_aleph_hypre PUBLIC ${ARCANE_BASE_LIBRARIES})
target_link_libraries(arcane_aleph_hypre PRIVATE arcane_aleph)
arcane_register_library(arcane_aleph_hypre OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
