# packages
# --------

# Chargement des packages de arcane

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if (NOT ALIEN_FOUND)
    loadPackage(NAME Alien ESSENTIAL)
endif ()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

loadPackage(NAME MPI   ESSENTIAL)
loadPackage(NAME Boost ESSENTIAL)
loadPackage(NAME GTest)

set(MPI_ROOT ${MPI_ROOT_PATH})

loadPackage(NAME MPIFort)

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

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# solveurs

# Needed to use the find_package provided by Arccon
set (Hypre_USE_CMAKE_CONFIG TRUE)

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

# Packages already loaded for Arcane : added here for dll copy on Windows
loadPackage(NAME Zoltan)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# arccon fix
if (TARGET arcconpkg_Hypre)
  add_library(hypre ALIAS arcconpkg_Hypre)
elseif (TARGET HYPRE::HYPRE)
  # Target 'HYPRE::HYPRE' is defined when Hypre is compiled with CMake
  # and provide a config file
  add_library(hypre ALIAS HYPRE::HYPRE)
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

if (TARGET arcconpkg_HDF5)
  add_library(hdf5 ALIAS arcconpkg_HDF5)
endif()

# load package can't deal with this...
find_package(Boost COMPONENTS program_options REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# arcane

# NB: en dernier car arcane charge éventuellement d'autres packages
#     si le package existe déjà, on ne fait rien
if (NOT Arcane_FOUND)
    message(STATUS "Load Arcane, since not found")
    loadPackage(NAME Arcane)
endif()
if (Arcane_FOUND)
    set(ALIEN_USE_ARCANE YES)
endif()

if(NOT TARGET arcane_core)
    logStatus("arcane is not found")
else()
    logStatus("arcane is found")
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
