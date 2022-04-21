addSources(alien_ifpen_solvers
    algebra/HTSInternalLinearAlgebra.cc
    linear_solver/HTSInternalLinearSolver.cc
    converters/SimpleCSR/SimpleCSR_to_HTS_MatrixConverter.cc
    converters/SimpleCSR/SimpleCSR_to_HTS_VectorConverter.cc
    converters/SimpleCSR/HTS_to_SimpleCSR_VectorConverter.cc
    data_structure/HTSMatrix.cc
    data_structure/HTSVector.cc
    )
addSources(alien_ifpen_solvers
    eigen_solver/HTSInternalEigenSolver.cc
    )
generateAxl(alien_ifpen_solvers
    linear_solver/arcane/HTSSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
generateAxl(alien_ifpen_solvers
    eigen_solver/arcane/HTSEigenSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
addSources(alien_ifpen_solvers
    linear_solver/arcane/HTSLinearSolver.cc
    )

addSources(alien_ifpen_solvers
    eigen_solver/arcane/HTSEigenSolver.cc
    )
linkLibraries(alien_ifpen_solvers htssolver harts)

if(TARGET spectra)
  linkLibraries(alien_ifpen_solvers spectra)
endif()

if(TARGET eigen3)
  linkLibraries(alien_ifpen_solvers eigen3)
endif()

if(TARGET hypre)
  linkLibraries(alien_ifpen_solvers hypre)
endif()

if(TARGET arpack)
  linkLibraries(alien_ifpen_solvers arpack)
endif()

if(TARGET metis)
  linkLibraries(alien_ifpen_solvers metis)
endif()

if(TARGET hdf5)
  linkLibraries(alien_ifpen_solvers hdf5)
endif()

if(TARGET mkl)
  linkLibraries(alien_ifpen_solvers mkl)
endif()

if(TARGET tbb)
  linkLibraries(alien_ifpen_solvers tbb)
endif()

if(TARGET cuda)
  linkLibraries(alien_ifpen_solvers cuda)
endif()

if(TARGET numa)
  linkLibraries(alien_ifpen_solvers numa)
endif()
