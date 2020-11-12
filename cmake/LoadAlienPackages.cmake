# packages
# --------

# Chargement des packages de arcane

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

isp_loadPackage(NAME Alien ESSENTIAL)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

isp_loadPackage(NAME MPI   ESSENTIAL)
isp_loadPackage(NAME Boost ESSENTIAL)
isp_loadPackage(NAME GTest ESSENTIAL)

set(MPI_ROOT ${MPI_ROOT_PATH})

## En fait pour cette dependance, en reecrivant a minima, on veut juste les blas
#loadPackage(NAME MKL)
#if (NOT MKL_FOUND)
#  loadPackage(NAME BLAS ESSENTIAL)
#endif (NOT MKL_FOUND)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# tools

isp_loadPackage(NAME LibXml2)
isp_loadPackage(NAME Metis)
isp_loadPackage(NAME HDF5)
isp_loadPackage(NAME HWLoc)
isp_loadPackage(NAME Numa)
isp_loadPackage(NAME TBB)
isp_loadPackage(NAME HARTS)
isp_loadPackage(NAME Cuda)
isp_loadPackage(NAME NvAMG)
isp_loadPackage(NAME FFTW3)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# solveurs

#loadPackage(NAME Umfpack)
isp_loadPackage(NAME PETSc)
isp_loadPackage(NAME SLEPc)
isp_loadPackage(NAME Hypre)
isp_loadPackage(NAME MTL4)
isp_loadPackage(NAME SuperLU)
isp_loadPackage(NAME SuperLU_DIST)
isp_loadPackage(NAME MUMPS)
isp_loadPackage(NAME IFPSolver)
isp_loadPackage(NAME MCGSolver)
isp_loadPackage(NAME Eigen3)
isp_loadPackage(NAME Spectra)
isp_loadPackage(NAME HTSSolver)
isp_loadPackage(NAME Trilinos)
isp_loadPackage(NAME Arpack)
isp_loadPackage(NAME HPDDM)

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
  add_library(mtl ALIAS arcconpkg_MTL4)
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
isp_loadPackage(NAME Arcane)

if(NOT TARGET arcane_core)
    logStatus("arcane is not found")
else()
    logStatus("arcane is found")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
