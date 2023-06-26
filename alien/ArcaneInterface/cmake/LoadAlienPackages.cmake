# packages
# --------

# Chargement des packages de arcane

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

loadPackage(NAME Alien ESSENTIAL)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

loadPackage(NAME Mpi   ESSENTIAL)
loadPackage(NAME Boost ESSENTIAL)
loadPackage(NAME GTest ESSENTIAL)

set(MPI_ROOT ${MPI_ROOT_PATH})

## En fait pour cette dependance, en reecrivant a minima, on veut juste les blas
loadPackage(NAME MKL)
#if (NOT MKL_FOUND)
#loadPackage(NAME BLAS ESSENTIAL)
#endif (NOT MKL_FOUND)
loadPackage(NAME OpenBLAS)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# tools

loadPackage(NAME LibXml2)
loadPackage(NAME Metis)
loadPackage(NAME HDF5)
loadPackage(NAME HWLoc)
loadPackage(NAME Numa)
loadPackage(NAME TBB)
loadPackage(NAME HARTS)
loadPackage(NAME Cuda)
loadPackage(NAME NvAMG)
loadPackage(NAME FFTW3)
loadPackage(NAME MPIFort)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# solveurs

#loadPackage(NAME Umfpack)
loadPackage(NAME PETSc)
loadPackage(NAME SLEPc)
loadPackage(NAME Hypre)
loadPackage(NAME MTL4)
loadPackage(NAME SuperLU)
loadPackage(NAME SuperLU_DIST)
loadPackage(NAME MUMPS)
loadPackage(NAME IFPSolver)
loadPackage(NAME MCGSolver)
loadPackage(NAME Eigen3)
loadPackage(NAME Spectra)
loadPackage(NAME HTSSolver)
loadPackage(NAME Trilinos)
loadPackage(NAME Arpack)
loadPackage(NAME HPDDM)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# arccon fix
if (TARGET arcconpkg_Hypre)
  add_library(hypre ALIAS arcconpkg_Hypre)
endif()

if (TARGET arcconpkg_MPI)
  add_library(mpi ALIAS arcconpkg_MPI)
endif()

if (TARGET arcconpkg_PETSc)
  add_library(petsc ALIAS arcconpkg_PETSc)
endif()

if (TARGET arcconpkg_SuperLU)
  add_library(superlu ALIAS arcconpkg_SuperLU)
endif()

if (TARGET arcconpkg_MTL4)
  add_library(mtl4 ALIAS arcconpkg_MTL4)
endif()

if (TARGET arcconpkg_TBB)
  add_library(tbb ALIAS arcconpkg_TBB)
endif()

# load package can't deal with this...
find_package(Boost COMPONENTS program_options system REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# arcane

# NB: en dernier car arcane charge éventuellement d'autres packages
#     si le package existe déjà, on ne fait rien
loadPackage(NAME Arcane)

if(NOT TARGET arcane_core)
    logStatus("arcane is not found")
else()
    logStatus("arcane is found")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
