set(ARCGEOSIM_BASE_PATH //irfich3/R11-Group/Rt40_50_prj/D/D1653_Arcane/Public/Windows64)
set(SOFTS_BASE_PATH ${ARCGEOSIM_BASE_PATH}/Softs)
set(INTEL_COMPOSER_PATH "${SOFTS_BASE_PATH}/Intel/Composer XE 2011 SP1")

set(ARCANE_BASE_PATH //irfich3/R11-Group/Rt40_50_prj/D/D1653_Arcane/Public/Windows64/Arcane)

set(GLIB_ROOT 
	${SOFTS_BASE_PATH}/glib-2.54.3-vs15
	CACHE INTERNAL "GLib root path")

if(DEBUG) 
  set(ARCANE_ROOT 
    C:/Users/tuncx/Desktop/Arcane.gitlab/install-debug
    CACHE INTERNAL "Arcane root path")
else()
  set(ARCANE_ROOT 
    C:/Users/tuncx/Desktop/Arcane.gitlab/install-release
    CACHE INTERNAL "Arcane root path")
endif()

if(DEBUG) 
set(GTEST_ROOT 
    ${SOFTS_BASE_PATH}/googletest/1.8.0/VS15/Debug
	CACHE INTERNAL "GTest root path")
else()
set(GTEST_ROOT 
    ${SOFTS_BASE_PATH}/googletest/1.8.0/VS15/Release
	CACHE INTERNAL "GTest root path")
endif()

set(MPI_ROOT 
    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mpi/intel64"
	CACHE INTERNAL "Mpi root path")
set(MPI_SKIP_FINDPACKAGE ON 
    CACHE INTERNAL "Mpi skip find package")
set(MPI_BIN_FROM_ENV OFF 
    CACHE INTERNAL "Mpi bin from env")
   
set(BOOST_ROOT 
    ${SOFTS_BASE_PATH}/boost/1_67_0-vs15
	CACHE INTERNAL "Boost root path")
   
set(MKL_ROOT 
    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl"
	CACHE INTERNAL "MKL root path")

set(HDF5_ROOT 
    ${SOFTS_BASE_PATH}/hdf5/1.8.20-vs15
	CACHE INTERNAL "Hdf5 root path")
set(H5BuiltAsDynamicLib ON 
    CACHE INTERNAL "Hdf5 as dll")

if(DEBUG)
set(HYPRE_ROOT 
    ${SOFTS_BASE_PATH}/hypre/2.14.0-vs15/Debug
	CACHE INTERNAL "Hypre root path")
else()
set(HYPRE_ROOT 
    ${SOFTS_BASE_PATH}/hypre/2.14.0-vs15/Release
	CACHE INTERNAL "Hypre root path")
endif()

if(DEBUG)
set(PETSC_ROOT 
    ${SOFTS_BASE_PATH}/petsc/3.10.2-vs15/Debug
	CACHE INTERNAL "PETSc root path")
else()
set(PETSC_ROOT 
    ${SOFTS_BASE_PATH}/petsc/3.10.2-vs15/Release
	CACHE INTERNAL "PETSc root path")
endif()

set(MTL_ROOT 
    ${SOFTS_BASE_PATH}/mtl4/MTL4-4.0.9555
	CACHE INTERNAL "MTL root path")

set(IFPSOLVER_ROOT 
    ${SOFTS_BASE_PATH}/IFPSolver/2018-b/ref-mpi-IntelMPI2019-intel2019-hypre2.14
	CACHE INTERNAL "IFPSolver root path")
	
#set(EIGEN3_ROOT
#    ${SOFTS_BASE_PATH}/eigen3/eigen-3.3.4
#	CACHE INTERNAL "Eigen3 root path")

# problème d'installation, on doit recopier cette lib
# TODO: fixer l'installation de xerces
set(EXTRA_DLLS_TO_COPY 
    ${SOFTS_BASE_PATH}/xerces-c-3.2.2-vs15/bin/xerces-c_3_2.dll
    ${SOFTS_BASE_PATH}/superlu/5.2.1-vs15/${CMAKE_BUILD_TYPE}/lib/libsuperlu_5.2.dll
    CACHE INTERNAL "Extra dlls")

unset(SOFTS_BASE_PATH)
unset(INTEL_COMPOSER_PATH)
