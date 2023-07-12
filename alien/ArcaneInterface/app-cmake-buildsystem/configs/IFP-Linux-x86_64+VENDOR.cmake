############################# Packages for both debug/release modes ################################

set(IRSRVSOFT   /soft/irsrvsoft1/expl)
set(INFRADEV    /home/irsrvshare1/R11/infradev/softs)
set(ARCUSER     /home/irsrvshare1/R11/arcuser)
set(ARCUSERSOFT /home/irsrvshare1/R11/arcuser/softs)

set(ARCANE_PATH_PREFIX 
  ${ARCUSER}/Arcane/${ARCANE_VERSION}/Linux/${RHEL_TAG}/x86_64
  CACHE INTERNAL "Arcane path prefix")

if(NOT TARGET old_alien)
  set(MPI_ROOT 
      ${IRSRVSOFT}/IntelMPI_4.0.2/impi_4.0.2.003/intel64
      CACHE INTERNAL "Mpi root path")

  set(BOOST_ROOT 
      ${ARCUSERSOFT}/boost/1_62_0/Linux/RHEL6/x86_64/ref-gcc472 
      CACHE INTERNAL "Boost root path")

  set(MKL_ROOT
      ${IRSRVSOFT}/Intel_11.0/mkl 
      CACHE INTERNAL "MKL root path")

  set(BINUTILS_ROOT 
      ${ARCUSERSOFT}/binutils/2.17/Linux/RHEL5/x86_64
      CACHE INTERNAL "BinUtils root path")

  set(EIGEN3_ROOT 
      ${INFRADEV}/eigen/3.3
      CACHE INTERNAL "Eigen3 root path")

  set(EIGEN2_ROOT 
      ${INFRADEV}/eigen
      CACHE INTERNAL "Eigen2 root path")
  
  set(UMFPACK_ROOT 
      ${ARCUSERSOFT}/suitesparse/5.5/Linux/RHEL5/x86_64/ref-gcc412-without-metis
      CACHE INTERNAL "Umfpack root path")
  
  set(GEOMETRYKERNEL_VERSION
      201507-3d2d-simple_corefinement_algorithm
      CACHE INTERNAL "Geometry version")
  
  set(CUDA_ROOT 
      /home/irsrvshare1/R11/X_HPC/CUDA_7.0_RHEL6/cuda
      CACHE INTERNAL "Cuda root path")
  
  ############################# Packages specialized for debug/release modes ################################
  
  # RELEASE
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
  
    set(ARCANE_ROOT
        ${ARCANE_PATH_PREFIX}/ref-gcc47-intelMPI4
        CACHE INTERNAL "Arcane root path")
  
    set(ALIEN_ROOT
    	${ARCUSER}/Alien/1.0/Linux/RHEL6/x86_64/ref-wo-metis-gcc47-intelMPI4
        CACHE INTERNAL "ALIEN root path")
    
    set(ALIEN_VERSION 1.0
    	CACHE INTERNAL "ALIEN VERSION")
  
    set(GEOMETRYKERNEL_ROOT
        ${ARCUSERSOFT}/GeometryKernel/${GEOMETRYKERNEL_VERSION}/Linux/RHEL5/x86_64/ref
        CACHE INTERNAL "GeometryKernel root path")
  
  endif()
  
  # DEBUG
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  
    set(ARCANE_ROOT
        ${ARCANE_PATH_PREFIX}/dbg-gcc47-intelMPI4
        CACHE INTERNAL "Arcane root path")
  
    set(ALIEN_ROOT
        ${ARCUSER}/Alien/1.0/Linux/RHEL6/x86_64/dbg-wo-metis-gcc47-intelMPI4
        CACHE INTERNAL "ALIEN root path")
    
    set(ALIEN_VERSION 1.0
        CACHE INTERNAL "ALIEN VERSION")
  
    set(GEOMETRYKERNEL_ROOT
        ${ARCUSERSOFT}/GeometryKernel/${GEOMETRYKERNEL_VERSION}/Linux/RHEL5/x86_64/debug
        CACHE INTERNAL "GeometryKernel root path")
  
  endif()
  
  ##########################################################################################################
  
  unset(IRSRVSOFT)
  unset(INFRADEV)
  unset(ARCUSER)

else()

  set(INTEL_ROOT 
      ${IRSRVSOFT}/Intel_12.1.3/composer_xe_2011_sp1.9.293/compiler 
      CACHE INTERNAL "Intel root path")
  
  set(MPI_ROOT 
      ${IRSRVSOFT}/IntelMPI_4.0.2/impi_4.0.2.003/intel64
      CACHE INTERNAL "Mpi root path")
    
  set(BOOST_ROOT 
      ${ARCUSERSOFT}/boost/1_62_0/Linux/RHEL6/x86_64/ref-gcc472 
      CACHE INTERNAL "Boost root path")
  
  set(MKL_ROOT
      ${IRSRVSOFT}/Intel_11.0/mkl 
      CACHE INTERNAL "MKL root path")
  
  set(PETSC_ROOT
      ${ARCUSERSOFT}/petsc/3.1-p3/Linux/RHEL6/x86_64/ref-intel12-IntelMPI_4.0-mkl-without-metis-with-hypre2.10.1
      CACHE INTERNAL "PETSc root path")
  
  set(HYPRE_ROOT 
      ${INFRADEV}/hypre/hypre-2.10.1/Linux/RHEL6/x86_64/ref-intel12-intelmpi4.0.2-mkl-4arcgeosim 
      CACHE INTERNAL "Hypre root path")
  
  set(IFPSOLVER_ROOT
      ${INFRADEV}/IFPSolver/2017-a/Linux/RHEL6/x86_64/ref-mpi-IntelMPI4.0-intel12.1
      CACHE INTERNAL "IFPSolver root path")
  
  set(HWLOC_ROOT 
      ${INFRADEV}/hwloc/Linux/RHEL6/x86_64
      CACHE INTERNAL "HWLoc root path")
  
  set(NUMA_ROOT 
      /usr
      CACHE INTERNAL "Numa root path")
  
  set(NVAMG_ROOT 
      /home/irsrvshare1/R11/X_HPC/tools/nvamg/24.12.2014
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
  
  set(BINUTILS_ROOT 
      ${ARCUSERSOFT}/binutils/2.17/Linux/RHEL5/x86_64
      CACHE INTERNAL "BinUtils root path")
  
  set(EIGEN3_ROOT 
      ${INFRADEV}/eigen/3.3
      CACHE INTERNAL "Eigen3 root path")
  
  set(EIGEN2_ROOT 
      ${INFRADEV}/eigen
      CACHE INTERNAL "Eigen2 root path")
  
  set(UMFPACK_ROOT 
      ${ARCUSERSOFT}/suitesparse/5.5/Linux/RHEL5/x86_64/ref-gcc412-without-metis
      CACHE INTERNAL "Umfpack root path")
  
  set(SUPERLU_ROOT 
      ${ARCUSERSOFT}/superlu/4.0/Linux/RHEL6/x86_64/ref-intel12
      CACHE INTERNAL "Superlu root path")
    
  set(GEOMETRYKERNEL_VERSION
      201507-3d2d-simple_corefinement_algorithm
      CACHE INTERNAL "Geometry version")
  
  ############################# Packages specialized for debug/release modes ################################
  
  # RELEASE
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
  
    set(ARCANE_ROOT
        ${ARCANE_PATH_PREFIX}/ref-gcc47-intelMPI4
        CACHE INTERNAL "Arcane root path")
  
    set(GEOMETRYKERNEL_ROOT
        ${ARCUSERSOFT}/GeometryKernel/${GEOMETRYKERNEL_VERSION}/Linux/RHEL5/x86_64/ref
        CACHE INTERNAL "GeometryKernel root path")
  
  endif()
  
  # DEBUG
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  
    set(ARCANE_ROOT
        ${ARCANE_PATH_PREFIX}/dbg-gcc47-intelMPI4
        CACHE INTERNAL "Arcane root path")
  
    set(GEOMETRYKERNEL_ROOT
        ${ARCUSERSOFT}/GeometryKernel/${GEOMETRYKERNEL_VERSION}/Linux/RHEL5/x86_64/debug
        CACHE INTERNAL "GeometryKernel root path")
  
  endif()
  
  ##########################################################################################################
  
  unset(IRSRVSOFT)
  unset(INFRADEV)
  unset(ARCUSER)
  
endif()