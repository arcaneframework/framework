# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "Including arccore/CMakeLists.txt")

# ----------------------------------------------------------------------------
# Regarde la taille des différents types (short,int,long,long long) du C++

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CheckArccoreTypeSize.cmake)


configure_file(${CMAKE_CURRENT_LIST_DIR}/arccore_config.h.in arccore_config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/arccore_config.h DESTINATION include/arccore)
