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
linkLibraries(alien_ifpen_solvers htssolver harts tbb spectra)

if(TARGET arpack)
  linkLibraries(alien_ifpen_solvers arpack)
endif()

if(TARGET cuda)
  linkLibraries(alien_ifpen_solvers cuda)
endif()
