#
# Find the MPC includes and library
#
# This module defines
# MPC_INCLUDE_DIR, where to find headers,
# MPC_LIBRARIES, the libraries to link against to use MPC.
# MPC_FOUND, If false, do not try to use MPC.
 
FIND_LIBRARY(MPC_LIBRARY mpc)

FIND_LIBRARY(MPC_ALLOC_LIBRARY mpc_iso_alloc)

MESSAGE(STATUS "MPC_LIBRARY = ${MPC_LIBRARY}")
IF(MPC_LIBRARY)
  get_filename_component(MPC_LIB_PATH ${MPC_LIBRARY} PATH)
  get_filename_component(MPC_ROOT_PATH ${MPC_LIB_PATH} PATH)
  MESSAGE(STATUS "MPC ROOT PATH = ${MPC_ROOT_PATH}")
ENDIF(MPC_LIBRARY)

FIND_PATH(MPC_INCLUDE_DIR mpc.h
  PATHS
  ${MPC_ROOT_PATH}/include
)

MESSAGE(STATUS "MPC_INCLUDE_DIR = ${MPC_INCLUDE_DIR}")
 
SET( MPC_FOUND "NO" )
IF(MPC_INCLUDE_DIR)
  IF(MPC_LIBRARY)
    SET( MPC_FOUND "YES" )
    SET( MPC_LIBRARIES ${MPC_LIBRARY} ${MPC_ALLOC_LIBRARY})
    SET( MPC_INCLUDE_DIRS ${MPC_INCLUDE_DIR})
    # Liste des arguments a ajouter lors de la compilation
    execute_process(COMMAND ${MPC_ROOT_PATH}/bin/mpc_cflags OUTPUT_VARIABLE MPC_CFLAGS)
    message(STATUS "MPC_CFLAGS=${MPC_CFLAGS}")
    # Liste des arguments a ajouter lors de l'edition de lien
    execute_process(COMMAND ${MPC_ROOT_PATH}/bin/mpc_ldflags OUTPUT_VARIABLE _MPC_LDFLAGS)
    string(STRIP ${_MPC_LDFLAGS} MPC_LDFLAGS)
    message(STATUS "MPC_LDFLAGS=${MPC_LDFLAGS}")
 ENDIF(MPC_LIBRARY)
ENDIF(MPC_INCLUDE_DIR)
