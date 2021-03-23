#
# Find the Hypre includes and library
#
# This module defines
# HYPRE_INCLUDE_DIRS, where to find headers,
# HYPRE_LIBRARIES, the libraries to link against to use Hdf5.
# HYPRE_FOUND, If false, do not try to use Hdf5.
 
find_path(HYPRE_INCLUDE_DIR HYPRE.h
  PATHS ${HYPRE_INCLUDE_PATH} NO_DEFAULT_PATH)

foreach(HYPRE_LIB HYPRE)# HYPRE_DistributedMatrixPilutSolver HYPRE_Euclid 
    #HYPRE_sstruct_ls  HYPRE_sstruct_mv HYPRE_struct_ls HYPRE_struct_mv HYPRE_parcsr_ls 
    #HYPRE_parcsr_block_mv HYPRE_MatrixMatrix  HYPRE_DistributedMatrix HYPRE_IJ_mv 
    #HYPRE_parcsr_mv HYPRE_seq_mv HYPRE_krylov HYPRE_utilities HYPRE_ParaSails 
    #HYPRE_LSI HYPRE_FEI HYPRE_mli HYPRE_multivector HYPRE_superlu
    #)    
  find_library(HYPRE_SUB_${HYPRE_LIB} ${HYPRE_LIB}
    PATHS ${HYPRE_LIBRARY_PATH} NO_DEFAULT_PATH)
  if(HYPRE_SUB_${HYPRE_LIB})
    set(HYPRE_LIBRARY ${HYPRE_LIBRARY} ${HYPRE_SUB_${HYPRE_LIB}})
  else(HYPRE_SUB_${HYPRE_LIB})
    set(HYPRE_LIBRARY_FAILED "YES")
  endif(HYPRE_SUB_${HYPRE_LIB})
endforeach(HYPRE_LIB)

set(HYPRE_FOUND "NO")
if(HYPRE_INCLUDE_DIR)
  if(NOT HYPRE_LIBRARY_FAILED)
    set(HYPRE_FOUND "YES")
    set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})
    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})
    set(HYPRE_FLAGS "-DUSE_HYPRE")
  endif(NOT HYPRE_LIBRARY_FAILED)
endif(HYPRE_INCLUDE_DIR)
