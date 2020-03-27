addSources(alien_ifpen_solvers
    Algebra/HTSInternalLinearAlgebra.cc
    LinearSolver/HTSInternalLinearSolver.cc
    Converters/SimpleCSR/SimpleCSR_to_HTS_MatrixConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_HTS_VectorConverter.cc
    Converters/SimpleCSR/HTS_to_SimpleCSR_VectorConverter.cc
    DataStructure/HTSMatrix.cc
    DataStructure/HTSVector.cc
    )
addSources(alien_ifpen_solvers
    EigenSolver/HTSInternalEigenSolver.cc
    )
generateAxl(alien_ifpen_solvers
    LinearSolver/Arcane/HTSSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
generateAxl(alien_ifpen_solvers
    EigenSolver/Arcane/HTSEigenSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
addSources(alien_ifpen_solvers
    LinearSolver/Arcane/HTSLinearSolver.cc
    )

addSources(alien_ifpen_solvers
    EigenSolver/Arcane/HTSEigenSolver.cc
    )
linkLibraries(alien_ifpen_solvers htssolver harts tbb spectra mtl)

if(TARGET arpack)
  linkLibraries(alien_ifpen_solvers arpack)
endif()

if(TARGET cuda)
  linkLibraries(alien_ifpen_solvers cuda)
endif()
