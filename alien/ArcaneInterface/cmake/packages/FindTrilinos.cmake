#
# Find the TRILINOS includes and library
#
# This module uses
# TRILINOS_ROOT
#
# This module defines
# TRILINOS_FOUND
# TRILINOS_INCLUDE_DIRS
# TRILINOS_LIBRARIES
#
# Target mcgsolver

if (NOT TRILINOS_ROOT)
    set(TRILINOS_ROOT $ENV{TRILINOS_ROOT})
endif ()

if (TRILINOS_ROOT)
    set(_TRILINOS_SEARCH_OPTS NO_DEFAULT_PATH)
else ()
    set(_TRILINOS_SEARCH_OPTS)
endif ()

SET(Trilinos_PREFIX ${TRILINOS_ROOT})

SET(Trilinos_DIR ${TRILINOS_ROOT})

SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})

# Get Trilinos as one entity
#FIND_PACKAGE(Trilinos) 
FIND_PACKAGE(Trilinos COMPONENTS
        Ifpack2 Anasazi Amesos2
        # ShyLU_Node ShyLU_NodeTacho
        MueLu NOX
        Belos ML Ifpack Zoltan2 Pamgen Amesos Galeri AztecOO Xpetra
        Teuchos
        TeuchosKokkosComm TeuchosKokkosCompat
        TeuchosRemainder TeuchosNumerics TeuchosComm TeuchosParameterList TeuchosParser TeuchosCore
        Kokkos KokkosAlgorithms KokkosContainers
        KokkosCore
        PATHS ${TRILINOS_ROOT}
        HINTS lib
        )

# Echo trilinos build info just for fun
MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")


message("TARGET TRILINOS :${Trilinos_FOUND}")
if (Trilinos_FOUND AND NOT TARGET trilinos)
    #if(Trilinos_FOUND)

    message(" TRILINOS INCS : ${TRILINOS_INCLUDE_DIRS}")
    set(TRILINOS_INCLUDE_DIRS ${Trilinos_INCLUDE_DIR})
    message(" TRILINOS INCS : ${TRILINOS_INCLUDE_DIRS} ; ${Trilinos_INCLUDE_DIRS}")

    foreach (comp ${Trilinos_LIBRARIES})

        if (${comp} STREQUAL "gtest")
            #message("GTEST ")
            find_library(Trilinos_${comp}_LIBRARY
                    NAMES ${comp}
                    HINTS $ENV{GTEST_ROOT}
                    PATH_SUFFIXES lib lib64
                    ${_TRILINOS_SEARCH_OPTS})

            #message(status "GTEST :${Trilinos_${comp}_LIBRARY}")
        else ()
            #message("PACK ${comp}")
            find_library(Trilinos_${comp}_LIBRARY
                    NAMES ${comp}
                    HINTS ${TRILINOS_ROOT}
                    PATH_SUFFIXES lib lib64
                    ${_TRILINOS_SEARCH_OPTS})
        endif ()

        list(APPEND TRILINOS_LIBRARIES ${Trilinos_${comp}_LIBRARY})

        #message("Trilinos pack ${comp} lib : ${Trilinos_${comp}_LIBRARY}")

        add_library(trilinos_${comp} UNKNOWN IMPORTED)

        set_target_properties(trilinos_${comp} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${Trilinos_INCLUDE_DIRS}")

        set_target_properties(trilinos_${comp} PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                IMPORTED_LOCATION "${Trilinos_${comp}_LIBRARY}")

    endforeach ()

    # TRILINOS
    add_library(trilinos INTERFACE IMPORTED)

    foreach (comp ${Trilinos_LIBRARIES})
        set_property(TARGET trilinos APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES "trilinos_${comp}")
    endforeach ()

endif ()
