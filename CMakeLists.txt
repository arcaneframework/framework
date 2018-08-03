# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "Including arccore/CMakeLists.txt")

# ----------------------------------------------------------------------------
# Regarde la taille des différents types (short,int,long,long long) du C++

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CheckArccoreTypeSize.cmake)
