# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "Including arccore/CMakeLists.txt")

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

if (NOT ARCCORE_EXPORT_TARGET)
  set(ARCCORE_EXPORT_TARGET ArccorTargets)
endif()

add_subdirectory(src/base/arccore/base)
