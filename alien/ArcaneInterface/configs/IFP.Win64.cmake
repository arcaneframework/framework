set(ARCGEOSIM_BASE_PATH //irfich3/R11-Group/Rt40_50_prj/D/D1653_Arcane/Public/Windows64)
set(SOFTS_BASE_PATH ${ARCGEOSIM_BASE_PATH}/Softs)
set(INTEL_COMPOSER_PATH "${SOFTS_BASE_PATH}/Intel/Composer XE 2011 SP1")

set(ARCANE_BASE_PATH //irfich3/R11-Group/Rt40_50_prj/D/D1653_Arcane/Public/Windows64/Arcane)

set(GLIB_ROOT 
	${SOFTS_BASE_PATH}/GLib
	CACHE INTERNAL "GLib root path")

set(GTEST_ROOT 
    ${SOFTS_BASE_PATH}/googletest/VS13 
	CACHE INTERNAL "GTest root path")

set(MPI_ROOT 
    ${SOFTS_BASE_PATH}/Intel/MPI/4.0.1.004-4.0.3.010/em64t 
	CACHE INTERNAL "Mpi root path")
set(MPI_SKIP_FINDPACKAGE ON 
    CACHE INTERNAL "Mpi skip find package")
set(MPI_BIN_FROM_ENV ON 
    CACHE INTERNAL "Mpi bin from env")
   
set(BOOST_ROOT 
    ${SOFTS_BASE_PATH}/boost/1_62_0 
	CACHE INTERNAL "Boost root path")

set(MKL_ROOT 
    ${INTEL_COMPOSER_PATH}/mkl 
	CACHE INTERNAL "MKL root path")

set(HDF5_ROOT 
    ${SOFTS_BASE_PATH}/hdf5/1.8.11 
	CACHE INTERNAL "Hdf5 root path")

set(HYPRE_ROOT 
    ${SOFTS_BASE_PATH}/hypre/2.9.0b/ref-mpi-IntelMPI4-Intel12
	CACHE INTERNAL "Hypre root path")

set(PETSC_ROOT 
    ${SOFTS_BASE_PATH}/petsc/3.1-p3/ref-mpi-IntelMPI4-Intel12-Hypre2.9
	CACHE INTERNAL "PETSc root path")

set(MTL_ROOT 
    ${SOFTS_BASE_PATH}/mtl4/install
	CACHE INTERNAL "MTL root path")

set(IFPSOLVER_ROOT 
    ${SOFTS_BASE_PATH}/IFPSolver/2017-a/ref-mpi-IntelMPI4-Intel12.1-hypre_2.10.1
	CACHE INTERNAL "IFPSolver root path")

set(EIGEN2_ROOT
    ${SOFTS_BASE_PATH}/eigen2/install
	CACHE INTERNAL "Eigen2 root path")
	
set(EIGEN3_ROOT
    ${SOFTS_BASE_PATH}/eigen3/install
	CACHE INTERNAL "Eigen3 root path")

set(GEOMETRYKERNEL_VERSION 201507-3d2d-simple_corefinement_algorithm)
	
set(GEOMETRYKERNEL_ROOT
    ${SOFTS_BASE_PATH}/GeometryKernel/${GEOMETRYKERNEL_VERSION}/Windows/Seven/x86_64/$ENV{VSTUDIO_SHORT_NAME}
	CACHE INTERNAL "GeometryKernel root path")

# problème d'installation, on doit recopier cette lib
# TODO: fixer l'installation de xerces
set(EXTRA_DLLS_TO_COPY 
    ${SOFTS_BASE_PATH}/xerces-c-3.1.1-x86_64-windows-vc-10.0/bin/xerces-c_3_1.dll
    CACHE INTERNAL "Extra dlls")

unset(SOFTS_BASE_PATH)
unset(INTEL_COMPOSER_PATH)
