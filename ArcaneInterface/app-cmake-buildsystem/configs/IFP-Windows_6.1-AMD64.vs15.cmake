set(ARCGEOSIM_BASE_PATH //irfich3/R11-Group/Rt40_50_prj/D/D1653_Arcane/Public/Windows64)
set(SOFTS_BASE_PATH ${ARCGEOSIM_BASE_PATH}/Softs)
set(INTEL_COMPOSER_PATH "${SOFTS_BASE_PATH}/Intel/Composer XE 2011 SP1")
set(INTEL_SHARED_LIBS_PATH "C:/Program Files (x86)/Common Files/Intel/Shared Libraries/redist/intel64/compiler")

############################# Arcane libraries ################################

set(ARCANE_BASE_PATH ${ARCGEOSIM_BASE_PATH}/Arcane)

# Sous windows le suivi des liens de type "shortcut" n'est pas natif on utilise donc WindowsPathResolver
exec_program(${WINDOWS_PATH_RESOLVER_TOOL}
               ARGS --cmake --directory ${ARCANE_BASE_PATH}/${ARCANE_VERSION}
               OUTPUT_VARIABLE RESOLVED_ARCANE_PATH
               RETURN_VALUE RESOLVED_ARCANE_RETURN)
if(NOT RESOLVED_ARCANE_RETURN STREQUAL "0")
  message(FATAL_ERROR "An error occurs while resolving path '${ARCANE_BASE_PATH}/${ARCANE_VERSION}'")
else()
  message(STATUS "Resolved Arcane Path is '${RESOLVED_ARCANE_PATH}'")
endif()

if(NOT TARGET old_alien)
set(ALIEN_BASE_PATH ${ARCGEOSIM_BASE_PATH}/Alien)
set(ALIEN_VERSION 1.1
    CACHE INTERNAL "ALIEN VERSION")

# Sous windows le suivi des liens de type "shortcut" n'est pas natif on utilise donc WindowsPathResolver
exec_program(${WINDOWS_PATH_RESOLVER_TOOL}
               ARGS --cmake --directory ${ALIEN_BASE_PATH}/${ALIEN_VERSION}
               OUTPUT_VARIABLE RESOLVED_ALIEN_PATH
               RETURN_VALUE RESOLVED_ALIEN_RETURN)
if(NOT RESOLVED_ALIEN_RETURN STREQUAL "0")
  message(FATAL_ERROR "An error occurs while resolving path '${ALIEN_BASE_PATH}/${ALIEN_VERSION}'")
else()
  message(STATUS "Resolved Alien Path is '${RESOLVED_ALIEN_PATH}'")
endif()


# RELEASE
if(CMAKE_BUILD_TYPE STREQUAL "Release")

    set(ARCANE_ROOT
        ${RESOLVED_ARCANE_PATH}/Windows/Seven/x86-64/ref-$ENV{VSTUDIO_SHORT_NAME}-Intel2019.3
        CACHE INTERNAL "Arcane root path")
	
	set(ALIEN_ROOT
		${RESOLVED_ALIEN_PATH}/Windows/Seven/x86-64/ref-arcane-2.2.1-vs17-IntelMpi4
		CACHE INTERNAL "ALIEN root path")
endif()

# DEBUG
if(CMAKE_BUILD_TYPE STREQUAL "Debug")

    set(ARCANE_ROOT
        ${RESOLVED_ARCANE_PATH}/Windows/Seven/x86-64/dbg-$ENV{VSTUDIO_SHORT_NAME}-Intel2019.3
        CACHE INTERNAL "Arcane root path")

	set(ALIEN_ROOT
		${RESOLVED_ALIEN_PATH}/Windows/Seven/x86-64/dbg-arcane-2.2.1-vs17-IntelMpi4
		CACHE INTERNAL "ALIEN root path")
		
endif()
			   
############################# Other libraries ################################

set(GTEST_ROOT 
    ${SOFTS_BASE_PATH}/googletest/1.8.0/VS15/Debug
	CACHE INTERNAL "GTest root path")

set(MPI_ROOT 
    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mpi/intel64"
	CACHE INTERNAL "Mpi root path")
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

set(LIBXML2_ROOT 
    ${SOFTS_BASE_PATH}/libxml2/libxml2-2.9.8-vs15
	CACHE INTERNAL "libxml2 root path")

set(ICONV_ROOT 
    ${SOFTS_BASE_PATH}/libiconv-1.15-vs15
	CACHE INTERNAL "libiconv root path")
	
#set(EIGEN2_ROOT
#    ${SOFTS_BASE_PATH}/eigen2/install
#	CACHE INTERNAL "Eigen2 root path")
	
set(EIGEN3_ROOT
    ${SOFTS_BASE_PATH}/eigen3/eigen-3.3.4
	CACHE INTERNAL "Eigen3 root path")

set(GLIB_ROOT 
	${SOFTS_BASE_PATH}/glib-2.54.3-vs15
	CACHE INTERNAL "GLib root path")

set(GEOMETRYKERNEL_ROOT
    ${SOFTS_BASE_PATH}/GeometryKernel/BasicGeometryKernel-release2013/GeometryKernel-vs15
	CACHE INTERNAL "GeometryKernel root path")


set(CARNOT_ROOT 
    ${SOFTS_BASE_PATH}/Carnot/V3.1.7.3/VC12
    CACHE INTERNAL "Carnot root path")

# probleme d'installation, on doit recopier ces libs
set(EXTRA_DLLS_TO_COPY 
    ${INTEL_SHARED_LIBS_PATH}/libifportmd.dll
    ${INTEL_SHARED_LIBS_PATH}/libifcoremd.dll
    ${INTEL_SHARED_LIBS_PATH}/libmmd.dll
	CACHE INTERNAL "Extra dlls")
	
unset(SOFTS_BASE_PATH)
unset(INTEL_COMPOSER_PATH)
unset(FORTRAN_LIBRARY_PATH)
else()
# RELEASE
if(CMAKE_BUILD_TYPE STREQUAL "Release")

    set(ARCANE_ROOT
        ${RESOLVED_ARCANE_PATH}/Windows/Seven/x86-64/ref-$ENV{VSTUDIO_SHORT_NAME}-IntelMpi4
        CACHE INTERNAL "Arcane root path")

endif()

# DEBUG
if(CMAKE_BUILD_TYPE STREQUAL "Debug")

    set(ARCANE_ROOT
        ${RESOLVED_ARCANE_PATH}/Windows/Seven/x86-64/dbg-$ENV{VSTUDIO_SHORT_NAME}-IntelMpi4
        CACHE INTERNAL "Arcane root path")

endif()

############################# Other libraries ################################

set(GTEST_ROOT 
    ${SOFTS_BASE_PATH}/googletest/VS13 
	CACHE INTERNAL "GTest root path")

set(MPI_ROOT 
    ${SOFTS_BASE_PATH}/Intel/MPI/4.0.1.004-4.0.3.010/em64t 
	CACHE INTERNAL "Mpi root path")
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

set(GLIB_ROOT
    ${SOFTS_BASE_PATH}/GLib
	CACHE INTERNAL "GLib root path")

set(GEOMETRYKERNEL_VERSION 201507-3d2d-simple_corefinement_algorithm)
	
set(GEOMETRYKERNEL_ROOT
    ${SOFTS_BASE_PATH}/GeometryKernel/${GEOMETRYKERNEL_VERSION}/Windows/Seven/x86_64/$ENV{VSTUDIO_SHORT_NAME}
	CACHE INTERNAL "GeometryKernel root path")

# probleme d'installation, on doit recopier ces libs
set(EXTRA_DLLS_TO_COPY 
    ${ARCGEOSIM_BASE_PATH}/Utils/Magic/libintl-8.dll
    ${ARCGEOSIM_BASE_PATH}/Utils/Magic/intel12_extralibs/libifportmd.dll
    ${ARCGEOSIM_BASE_PATH}/Utils/Magic/intel12_extralibs/libifcoremd.dll
    ${ARCGEOSIM_BASE_PATH}/Utils/Magic/intel12_extralibs/libmmd.dll
    ${ARCGEOSIM_BASE_PATH}/Utils/Magic/intel12_extralibs/svml_dispmd.dll
	CACHE INTERNAL "Extra dlls")

unset(SOFTS_BASE_PATH)
unset(INTEL_COMPOSER_PATH)
unset(FORTRAN_LIBRARY_PATH)
endif()