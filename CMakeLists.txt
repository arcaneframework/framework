# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "Including arccore/CMakeLists.txt")

# ----------------------------------------------------------------------------

if (NOT ARCCORE_EXPORT_TARGET)
  set(ARCCORE_EXPORT_TARGET ArccoreTargets)
endif()

# ----------------------------------------------------------------------------

if (NOT ARCCON_ROOT)
  message(FATAL_ERROR "Variable 'ARCCON_ROOT' is not set")
endif()
if (NOT ARCCON_EXPORT_TARGET)
  set(ARCCON_EXPORT_TARGET ${ARCCORE_EXPORT_TARGET})
endif()
include(${ARCCON_ROOT}/Arccon.cmake)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(ARCCORE_DEBUG YES)
endif()

# ----------------------------------------------------------------------------
# Définit une cible fictive 'arccore_build_compile_flags' et qui contient les
# options de compilation internes aux cibles de Arccore. Ce sont par exemple
# les avertissements de compilation ou les options sur la visibilité des symboles.

add_library(arccore_build_compile_flags INTERFACE)
install(TARGETS arccore_build_compile_flags EXPORT ${ARCCORE_EXPORT_TARGET})

# ----------------------------------------------------------------------------
# Les symboles des bibliothèques ne sont pas visibles sauf si demandé
# explicitement

target_compile_options(arccore_build_compile_flags INTERFACE -fvisibility-inlines-hidden -fvisibility=hidden)

# ----------------------------------------------------------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CheckCompileFlags.cmake)

# ----------------------------------------------------------------------------
# Fonctions utilitaires diverses
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/Functions.cmake)

# ----------------------------------------------------------------------------
# Regarde la taille des différents types (short,int,long,long long) du C++
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CheckArccoreTypeSize.cmake)

# ----------------------------------------------------------------------------

configure_file(${CMAKE_CURRENT_LIST_DIR}/arccore_config.h.in arccore_config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/arccore_config.h DESTINATION include/arccore)

# ----------------------------------------------------------------------------

set(Arccore_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(Arccore_BINARY_DIR ${CMAKE_BINARY_DIR}/arccore)

message(STATUS "Arccore_SOURCE_DIR=${Arccore_SOURCE_DIR}")
message(STATUS "Arccore_BINARY_DIR=${Arccore_BINARY_DIR}")

add_subdirectory(src/base/arccore/base)
add_subdirectory(src/collections/arccore/collections)
add_subdirectory(src/concurrency/arccore/concurrency)
add_subdirectory(src/trace/arccore/trace)
add_subdirectory(src/messagepassing/arccore/messagepassing)
add_subdirectory(src/messagepassingmpi/arccore/messagepassingmpi)
