#------------------------------------------------------------------------------#
#-- Macro de traitement d'un répertoire
#------------------------------------------------------------------------------#

macro(__create_library lib_name)

  message(STATUS "library = ${lib_name}")

  include_directories(${ARCANE_INCLUDE_DIRS})
  include_directories(${APPLI_INCLUDE_DIRS})
  include_directories(${HYPRE_INCLUDE_DIRS})
  include_directories(${MPI_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  
  file(GLOB_RECURSE AXL_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.axl")
  file(GLOB_RECURSE HEADER_FILES "*.h")
  file(GLOB_RECURSE CPP_FILES "*.cc")
  
  set_source_files_properties(${CPP_FILES} PROPERTIES COMPILE_FLAGS ${HYPRE_FLAGS})

  foreach(axl ${AXL_FILES})
    get_filename_component(axl_base ${axl} NAME_WE)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${axl_base}_axl.h
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${axl} ${AXL2CC_COMPILER}
      COMMAND ${AXL2CC_COMPILER}
      ARGS ${AXL_ARGS} -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${axl}
      )
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${axl_base}_axl.h PROPERTIES GENERATED true)
    list(APPEND HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/${axl_base}_axl.h)
  endforeach(axl)

  add_library(${lib_name} ${HEADER_FILES} ${CPP_FILES})

endmacro(__create_library)

#------------------------------------------------------------------------------#
#-- Début du cmake
#------------------------------------------------------------------------------#

cmake_minimum_required (VERSION 2.6)

project (NumericalModelTest)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(FATAL_ERROR "--------------------------------
You cannot do in-source compilation. You have to build in a directory distinct from the source directory. 
Now you have to clean CMakeCache.txt and CMakeFiles from project root directory (created by your errornous command).
--------------------------------")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)

if("$ENV{PACKAGE_FILE}" STREQUAL "")
  message(FATAL_ERROR "--------------------------------
You must define and set PACKAGE_FILE environment variable to your current package file.
--------------------------------")
endif("$ENV{PACKAGE_FILE}" STREQUAL "")

if(EXISTS $ENV{PACKAGE_FILE})
  include($ENV{PACKAGE_FILE})
else(EXISTS $ENV{PACKAGE_FILE})
  message(FATAL_ERROR "User package file '${PACKAGE_FILE}' not found")
endif(EXISTS $ENV{PACKAGE_FILE})

set(CMAKE_FIND_DIR ${CMAKE_SOURCE_DIR}/CMake)

enable_language(CXX)

message(STATUS "Build root directory is ${CMAKE_BINARY_DIR}")
message(STATUS "Src root directory is ${CMAKE_SOURCE_DIR}")
message(STATUS "CMake find directory is ${CMAKE_FIND_DIR}")
message(STATUS "Package file is $ENV{PACKAGE_FILE}")

#------------------------------------------------------------------------------#
#-- RECHERCHE DES PAQUETAGES
#------------------------------------------------------------------------------#
MESSAGE(STATUS "---------------------------------------------------")

include(FindPkgConfig)

include(${CMAKE_FIND_DIR}/FindArcane.cmake)
include(${CMAKE_FIND_DIR}/FindVTK.cmake)
include(${CMAKE_FIND_DIR}/FindHypre.cmake)
include(${CMAKE_FIND_DIR}/FindMpi.cmake)
include(${CMAKE_FIND_DIR}/FindMKL.cmake)

#------------------------------------------------------------------------------#
#-- RAPPORT DE RECHERCHE DES PAQUETAGES
#------------------------------------------------------------------------------#
MESSAGE(STATUS "---------------------------------------------------")

message(STATUS "")
if(NOT ARCANE_FOUND)
  message(FATAL_ERROR "PACKAGE: Arcane                   : not found")
else(NOT ARCANE_FOUND)  
  message(STATUS      "PACKAGE: Arcane                   : OK")
  message(STATUS      "         Arcane PATH              : ${ARCANE_PATH}")
  message(STATUS      "         Arcane VERSION           : ${ARCANE_VERSION}")
  message(STATUS      "         Arcane AXL2CC            : ${AXL2CC_COMPILER}")
  message(STATUS      "         Arcane INCLUDE           : ${ARCANE_INCLUDE_DIRS}")
  message(STATUS      "         Arcane LIBRARIES         : ${ARCANE_LIBRARIES}")
endif(NOT ARCANE_FOUND)

#------------------------------------------------------------------------------#

message(STATUS "")
if(NOT VTK_FOUND)
  message(FATAL_ERROR "PACKAGE: VTK                       : not found")
else(NOT VTK_FOUND)
  message(STATUS      "PACKAGE: VTK                       : OK")
  message(STATUS      "         VTK INCLUDES              : ${VTK_INCLUDE_DIRS}")
  message(STATUS      "         VTK LIBRARIES             : ${VTK_LIBRARIES}")
endif(NOT VTK_FOUND)

#------------------------------------------------------------------------------#

message(STATUS "")
if(NOT HYPRE_FOUND)
  message(FATAL_ERROR "PACKAGE: Hypre                     : not found")
else(NOT HYPRE_FOUND)
  message(STATUS      "PACKAGE: Hypre                     : OK")
  message(STATUS      "         Hypre INCLUDES            : ${HYPRE_INCLUDE_DIRS}")
  message(STATUS      "         Hypre LIBRARIES           : ${HYPRE_LIBRARIES}")
endif(NOT HYPRE_FOUND)

#------------------------------------------------------------------------------#

message(STATUS "")
if(NOT MPI_FOUND)
  message(FATAL_ERROR "PACKAGE: Mpi                      : not found")
else(NOT MPI_FOUND)
  message(STATUS        "PACKAGE: Mpi                      : OK")
  message(STATUS        "         Mpi LAUNCHER             : ${MPI_EXECUTABLE}")
  message(STATUS        "         Mpi INCLUDES             : ${MPI_INCLUDE_DIRS}")
  message(STATUS        "         Mpi LIBRARIES            : ${MPI_LIBRARIES}")
endif(NOT MPI_FOUND)

#------------------------------------------------------------------------------#

message(STATUS "")
if(NOT MKL_FOUND)
  message(FATAL_ERROR "PACKAGE: MKL                       : not found")
else(NOT MKL_FOUND)
  message(STATUS      "PACKAGE: MKL                       : OK")
  message(STATUS      "         MKL INCLUDES              : ${MKL_INCLUDE_DIRS}")
  message(STATUS      "         MKL LIBRARIES             : ${MKL_LIBRARIES}")
endif(NOT MKL_FOUND)

#------------------------------------------------------------------------------#
#-- EXECUTABLE
#------------------------------------------------------------------------------#

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin) 

set(APPLICATION ${PROJECT_NAME}.exe)

set(APPLI_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src)

add_executable(${APPLICATION} ${CMAKE_SOURCE_DIR}/src/Main/main.cc)

include_directories(${ARCANE_INCLUDE_DIRS})

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${APPLICATION}
  ${ARCANE_LIBRARIES}
  Application 
  ${HYPRE_LIBRARIES}
  ${MKL_LIBRARIES}
  ${MPI_LIBRARIES}
  ${VTK_LIBRARIES})
