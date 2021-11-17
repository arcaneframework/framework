set(IRSRVSOFT   /soft/irsrvsoft1/expl)
set(INFRADEV    /home/irsrvshare1/R11/infradev/softs)
set(ARCUSER     /home/irsrvshare1/R11/arcuser)
set(ARCUSERSOFT /home/irsrvshare1/R11/arcuser/softs)

if(DEBUG) 
  set(ARCANE_ROOT 
    ${ARCUSER}/Arcane/2.2.1/Linux/RHEL6/x86_64/dbg-gcc47-intelMPI4
    CACHE INTERNAL "Arcane root path")
else()
  set(ARCANE_ROOT 
    ${ARCUSER}/Arcane/2.2.1/Linux/RHEL6/x86_64/ref-gcc47-intelMPI4
    CACHE INTERNAL "Arcane root path")
endif()

set(INTEL_ROOT 
  ${IRSRVSOFT}/Intel_12.1.3/composer_xe_2011_sp1.9.293/compiler 
  CACHE INTERNAL "Intel root path")

set(MPI_ROOT 
  ${IRSRVSOFT}/IntelMPI_4.0.2/impi_4.0.2.003/intel64
  CACHE INTERNAL "Mpi root path")
set(MPI_SKIP_FINDPACKAGE ON 
   CACHE INTERNAL "Mpi skip find package")

set(BOOST_ROOT 
  ${ARCUSERSOFT}/boost/1_62_0/Linux/RHEL6/x86_64/ref-gcc472 
  CACHE INTERNAL "Boost root path")

set(MKL_ROOT
  ${IRSRVSOFT}/Intel_11.0/mkl 
  CACHE INTERNAL "MKL root path")

set(GTEST_ROOT
  ${ARCUSERSOFT}/googletest/7_02_2017/Linux/RHEL6/x86_64/ref 
  CACHE INTERNAL "Gtest root path")

set(PETSC_ROOT
  ${ARCUSERSOFT}/petsc/3.1-p3/Linux/RHEL6/x86_64/ref-intel12-IntelMPI_4.0-mkl-without-metis-with-hypre2.10.1
  CACHE INTERNAL "PETSc root path")

set(HYPRE_ROOT 
  ${INFRADEV}/hypre/hypre-2.10.1/Linux/RHEL6/x86_64/ref-intel12-intelmpi4.0.2-mkl-4arcgeosim 
  CACHE INTERNAL "Hypre root path")

set(IFPSOLVER_ROOT 
  ${INFRADEV}/IFPSolver/2018-b/Linux/RHEL6/x86_64/ref-mpi-IntelMPI4.0-intel12.1
  CACHE INTERNAL "IFPSolver root path")

set(HWLOC_ROOT 
  ${INFRADEV}/hwloc/Linux/RHEL6/x86_64
  CACHE INTERNAL "HWLoc root path")

set(NVAMG_ROOT 
  /home/irsrvshare1/R11/X_HPC/tools/nvamg/18.06.2015-intelmpi
  CACHE INTERNAL "NvAMG root path")

set(MTL_ROOT 
  ${INFRADEV}/mtl4
  CACHE INTERNAL "MTL root path")

set(LIBXML2_ROOT 
  ${ARCUSERSOFT}/libxml2/2.9.2/Linux/RHEL6/x86_64/ref-gcc472
  CACHE INTERNAL "LibXML2 root path")

set(HDF5_ROOT 
  ${ARCUSERSOFT}/hdf5/1.8.8/Linux/RHEL5/x86_64/ref-gcc462
  CACHE INTERNAL "Hdf5 root path")

set(SUPERLU_ROOT 
  ${INFRADEV}/SuperLU_3.1/Linux/RHEL5/x86_64
  CACHE INTERNAL "Superlu root path")

set(UMFPACK_ROOT 
  ${ARCUSERSOFT}/suitesparse/5.5/Linux/RHEL5/x86_64/ref-gcc412-without-metis
  CACHE INTERNAL "Umfpack root path")

unset(IRSRVSOFT)
unset(INFRADEV)
unset(ARCUSER)
