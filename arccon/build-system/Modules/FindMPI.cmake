#
# Find the Mpi includes and library
#
# This module defines
# MPI_INCLUDE_DIRS, where to find headers,
# MPI_LIBRARIES, the libraries to link against to use Mpi.
# MPI_FOUND, If false, do not try to use Mpi.

# Avant d'include ce fichier, il est possible de positionner les variables suivantes:
# - WANTED_MPI_EXEC_NAME qui est le nom du lanceur MPI.
# - WANTED_MPI_LIBRARIES qui est le nom des bibliothèques MPI recherchées pour le cas
#   ou il en faut plusieurs (par exemple via ITAC). Si cette variable est positionnée
#   alors on n'utilise pas MPI_C_LIBRARIES retourné par le find_package().

# NOTE: il faut toujours faire le find_package(MPI) même si la
# cible existe déjà car le module MPI positionne certaines variables
# comme MPIEXEC_EXECUTABLE qui sont nécessaires.

# Essaie d'utilier le 'findMPI' fourni par CMake.
# Avec les version récentes de CMake (3.9+), il devrait y avoir une cible
# importee MPI::MPI_C définie. Il faudrait la tester.
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(MPI)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

include (${CMAKE_CURRENT_LIST_DIR}/../commands/commands.cmake)

# Essaie de trouver le nom du fournisseur de MPI.
# Cela peut-être utile pour positionner certaines options lors de la compilation
# ou du lancement des jobs. Par exemple, avec OpenMPI 3+, il faut ajouter
# l'option '--oversubscribe' à 'mpiexec' si on veut utiliser plus de PE que
# de coeurs disponibles sur la machine.
if (MPI_FOUND)
  if (NOT MPI_VENDOR_NAME)
    set(MPI_VENDOR_NAME "Unknown")
    # Pour rechercher la nom du vendeur, on regarde dans mpi.h
    # Pour openmpi, on recherche le define OPEN_MPI.
    # pour mpich, on recherche le define MPICH_VERSION
    if (MPI_CXX_HEADER_DIR)
      set(_MPI_HEADER_FILE ${MPI_CXX_HEADER_DIR}/mpi.h)
      file(READ ${_MPI_HEADER_FILE} _MPIH_CONTENT)
      if (_MPIH_CONTENT)
        string(FIND "${_MPIH_CONTENT}" "#define OPEN_MPI 1" _IS_OPENMPI)
        if (${_IS_OPENMPI} GREATER_EQUAL "0")
          set(MPI_VENDOR_NAME "openmpi")
        else()
          string(FIND "${_MPIH_CONTENT}" "#define MPICH_VERSION" _IS_MPICH)
          if (${_IS_MPICH} GREATER_EQUAL "0")
            set(MPI_VENDOR_NAME "mpich")
          else()
            string(FIND "${_MPIH_CONTENT}" "#define MSMPI_VER" _IS_MSMPI)
            if (${_IS_MSMPI} GREATER_EQUAL "0")
              set(MPI_VENDOR_NAME "msmpi")
            endif()
          endif()
        endif()
      else()
        message(STATUS "WARNING: Can not read content of ${_MPI_HEADER_FILE}")
      endif()
    endif()
    message(STATUS "MPI Vendor name is '${MPI_VENDOR_NAME}'")
    set(MPI_VENDOR_NAME ${MPI_VENDOR_NAME} CACHE STRING "MPI's vendor name" FORCE)
  endif()
endif()

arccon_return_if_package_found(MPI)

# Cette variable est definie si une plateforme specifie sa
# propre bibliotheque MPI
if (MPIEXEC_EXECUTABLE)
  set(MPI_EXEC_NAME ${MPIEXEC_EXECUTABLE})
else()
  if(NOT WANTED_MPI_EXEC_NAME)
    set(WANTED_MPI_EXEC_NAME mpiexec)
  endif()
  find_program(MPI_EXEC_NAME ${WANTED_MPI_EXEC_NAME})
endif()

if(MPI_EXEC_NAME)
  get_filename_component(MPI_BIN_PATH ${MPI_EXEC_NAME} PATH)
  get_filename_component(MPI_ROOT_PATH ${MPI_BIN_PATH} PATH)
  set(MPI_EXEC_PATH ${MPI_BIN_PATH})
endif()

message(STATUS "MPIEXEC_EXECUTABLE   = ${MPIEXEC_EXECUTABLE}")
message(STATUS "MPI_EXEC_NAME        = ${MPI_EXEC_NAME}")
message(STATUS "MPI_ROOT_PATH        = ${MPI_ROOT_PATH}")
message(STATUS "MPI_CXX_INCLUDE_PATH = ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI_CXX_HEADER_DIR   = ${MPI_CXX_HEADER_DIR}")
message(STATUS "MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI_C_INCLUDE_DIRS = ${MPI_C_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_ADDITIONAL_INCLUDE_DIRS = ${MPI_CXX_ADDITIONAL_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LIBRARIES    = ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI_VENDOR_NAME      = ${MPI_VENDOR_NAME}")

# Si find_package() a trouvé MPI, alors MPI_CXX_INCLUDE_PATH est positionné.
# Cependant il n'est pas forcément valide si l'installation de MPI n'est pas standard.
#find_path(MPI_INCLUDE_DIRS NAMES mpi.h PATHS ${MPI_CXX_INCLUDE_PATH} ${MPI_ROOT_PATH}/include)
#message(STATUS "MPI_INCLUDE_DIRS = ${MPI_INCLUDE_DIRS}")
set(MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})

# Sous windows, le FindMPI de CMake semble mettre dans MPI_CXX_INCLUDE_DIRS des
# répertoires issus des sources (à étudier...). Comme on n'a besoin que
# de 'mpi.h', on prend juste le r�pertoire le contenant
if (WIN32)
  if (MPI_CXX_HEADER_DIR)  
    set(MPI_INCLUDE_DIRS ${MPI_CXX_HEADER_DIR})
  endif()
endif()

# -- -- Support pour spécifier directement les bibliothèques MPI
if(NOT WANTED_MPI_LIBRARIES)
  if(MPI_PREFIX_LIBRARIES OR MPI_ADDITIONAL_LIBRARIES)
    set(WANTED_MPI_LIBRARIES ${MPI_PREFIX_LIBRARIES} ${WANTED_MPI_LIBRARY} ${MPI_ADDITIONAL_LIBRARIES})
  endif()
endif()

# Analyse les bibliothèques spécifiques éventuelles.
# Toutes celles de ${WANTED_MPI_LIBRARIES} doivent exister
set(_HAS_ALL_WANTED_MPI_LIBRARIES TRUE)
set(ALL_MPI_LIBRARIES)
foreach(wanted_lib ${WANTED_MPI_LIBRARIES})
  find_library(_MPI_LIBRARY_${wanted_lib} ${wanted_lib}
    PATHS
    ${MPI_ROOT_PATH}/lib
    )
  message(STATUS "Additional lib name=${wanted_lib} found=${_MPI_LIBRARY_${wanted_lib}}")
  if (NOT _MPI_LIBRARY_${wanted_lib})
    message(STATUS "Error: Can not find required MPI library name=${wanted_lib}")
    set(_HAS_ALL_WANTED_MPI_LIBRARIES FALSE)
  else()
    list(APPEND ALL_MPI_LIBRARIES ${_MPI_LIBRARY_${wanted_lib}})
  endif()
endforeach()
message(STATUS "ALL_MPI_LIBRARIES=${ALL_MPI_LIBRARIES} all?=${_HAS_ALL_WANTED_MPI_LIBRARIES}")
# -- -- -- -- -- -- -- -- -- -- -- -- --

set(MPI_FOUND NO)
if (MPI_INCLUDE_DIRS AND MPI_EXEC_NAME AND _HAS_ALL_WANTED_MPI_LIBRARIES)
  # Si on a spécifie les bibliothèques, on prend celles là.
  # Sinon, on prend par défaut les bibliothèques C++ car même si on les
  # utilisent pas directement, elles sont des fois automatiquement prises
  # en compte dès qu'on inclus 'mpi.h' dans un source C++.
  unset(MPI_LIBRARIES)
  if (ALL_MPI_LIBRARIES)
    set(MPI_LIBRARIES ${ALL_MPI_LIBRARIES})
  elseif (MPI_CXX_LIBRARIES)
    message(STATUS "Using MPI C++ libraries")
    set(MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
  elseif (MPI_C_LIBRARIES)
    message(STATUS "Using MPI C libraries")
    set(MPI_LIBRARIES ${MPI_C_LIBRARIES})
  endif()
  if (MPI_LIBRARIES)
    set(MPI_FOUND YES)
    message(STATUS "MPI_LIBRARIES ${MPI_LIBRARIES}")
  endif()
endif()

if (NOT MPI_FOUND)
  message(STATUS "Disabling MPI because of missing config. The following variables have to be valid:")
  message(STATUS "-- MPI_INCLUDE_DIRS              = ${MPI_INCLUDE_DIRS}")
  message(STATUS "-- MPI_EXEC_NAME                 = ${MPI_EXEC_NAME}")
  message(STATUS "-- _HAS_ALL_WANTED_MPI_LIBRARIES = ${_HAS_ALL_WANTED_MPI_LIBRARIES}")
endif()

arccon_register_package_library(MPI MPI)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
