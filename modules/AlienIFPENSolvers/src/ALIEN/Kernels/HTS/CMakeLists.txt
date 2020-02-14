addSources(ALIEN-IFPENSolvers
    Algebra/HTSInternalLinearAlgebra.cc
    LinearSolver/HTSInternalLinearSolver.cc
    Converters/SimpleCSR/SimpleCSR_to_HTS_MatrixConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_HTS_VectorConverter.cc
    Converters/SimpleCSR/HTS_to_SimpleCSR_VectorConverter.cc
    DataStructure/HTSMatrix.cc
    DataStructure/HTSVector.cc
    )
addSources(ALIEN-IFPENSolvers
    EigenSolver/HTSInternalEigenSolver.cc
    )
generateAxl(ALIEN-IFPENSolvers
    LinearSolver/Arcane/HTSSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
generateAxl(ALIEN-IFPENSolvers
    EigenSolver/Arcane/HTSEigenSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
addSources(ALIEN-IFPENSolvers
    LinearSolver/Arcane/HTSLinearSolver.cc
    )

addSources(ALIEN-IFPENSolvers
    EigenSolver/Arcane/HTSEigenSolver.cc
    )
linkLibraries(ALIEN-IFPENSolvers htssolver harts tbb spectra mtl)
if(TARGET cuda)
linkLibraries(ALIEN-IFPENSolvers cuda)
endif()
