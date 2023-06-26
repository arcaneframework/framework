
#set(INTEL_ROOT 
#  $ENV{EBROOTIFORT}/compilers_and_libraries/linux
#  ${IRSRVSOFT}/Intel_12.1.3/composer_xe_2011_sp1.9.293/compiler 
#  CACHE INTERNAL "Intel root path")


set(MPI_ROOT 
  $ENV{EBROOTIMPI}/intel64
  CACHE INTERNAL "Mpi root path")
set(MPI_SKIP_FINDPACKAGE ON 
  CACHE INTERNAL "Mpi skip find package")

set(BOOST_ROOT 
  $ENV{EBROOTBOOST}
  CACHE INTERNAL "Boost root path")

set(MKL_ROOT
  $ENV{EBROOTIMKL}/mkl
  CACHE INTERNAL "MKL root path")

set(FFTW3_ROOT
  $ENV{EBROOTIMKL}/mkl/include/fftw
#  $ENV{EBROOTFFTW}
  CACHE INTERNAL "FFTW3 root path")

#set(OpenBLAS_ROOT
#  $ENV{EBROOTOPENBLAS}
#  CACHE INTERNAL "OpenBLAS root path")

set(GTEST_ROOT
  $ENV{EBROOTGTEST}
  CACHE INTERNAL "Gtest root path")

set(PETSC_ROOT
  $ENV{EBROOTPETSC}
  CACHE INTERNAL "PETSc root path")

set(HYPRE_ROOT 
  $ENV{EBROOTHYPRE}
  CACHE INTERNAL "Hypre root path")

#set(IFPSOLVER_ROOT 
#  /home/irsrvshare1/R11/X_HPC/GPUSolver/anciauxa/IFPSolver-2018/Linux/RHEL6/x86_64/ref-mpi-IntelMPI2017-intel17.0.4
#  CACHE INTERNAL "IFPSolver root path")

set(MCGSOLVER_ROOT 
  /home/irsrvshare1/R11/infradev/softs/MCGSolver/2018-1/src/build_release_RHEL6_CUDA10.0_gcc7.3
#  /home/irsrvshare1/R11/X_HPC/GPUSolver/anciauxa/MCGProject-initial/build_debug_RHEL6_CUDA10.0_gcc7.3 
  CACHE INTERNAL "MCGSolver root path")

set(HWLOC_ROOT 
  $ENV{EBROOTHWLOC}
  CACHE INTERNAL "HWLoc root path")

set(NVAMG_ROOT 
  /home/irsrvshare1/R11/X_HPC/tools/nvamg/build-RHEL6-gcc-7-3-intel2018b-cuda10
  CACHE INTERNAL "NvAMG root path")

set(METIS_ROOT 
  $ENV{EBROOTPARMETIS}
  CACHE INTERNAL "Metis root path")

#set(HARTS_ROOT 
#  ${INFRADEV}/HARTS/V1
#  CACHE INTERNAL "HARTS root path")

#set(MTL_ROOT 
#  $ENV{EBROOTMTL4}
#  CACHE INTERNAL "MTL root path")

set(LIBXML2_ROOT 
  $ENV{EBROOTLIBXML2}
  CACHE INTERNAL "LibXML2 root path")

set(HDF5_ROOT 
  $ENV{EBROOTHDF5}
  CACHE INTERNAL "Hdf5 root path")

set(TBB_ROOT 
  $ENV{EBROOTTBB}
  CACHE INTERNAL "TBB root path")

set(SUPERLU_ROOT 
  $ENV{EBROOTSUPERLU}
  CACHE INTERNAL "Superlu root path")

set(SUPERLUDIST_ROOT
  $ENV{EBROOTSUPERLU_DIST}
  CACHE INTERNAL "Superlu root path")

set(UMFPACK_ROOT 
  $ENV{EBROOTSUITESPARSE}
  CACHE INTERNAL "Umfpack root path")

