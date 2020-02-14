set(CMAKE_CXX_COMPILER "$ENV{TRILINOS_ROOT}/bin/nvcc_wrapper")
addSources(ALIEN-ExternalPackages
    Algebra/TrilinosInternalLinearAlgebra.cc
    LinearSolver/TrilinosInternalLinearSolver.cc
    Converters/SimpleCSR/SimpleCSR_to_Trilinos_MatrixConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_Trilinos_VectorConverter.cc
    Converters/SimpleCSR/Trilinos_to_SimpleCSR_VectorConverter.cc
    DataStructure/TrilinosInternal.cc
    DataStructure/TrilinosMatrix.cc
    DataStructure/TrilinosVector.cc
    )
addSources(ALIEN-ExternalPackages
    EigenSolver/TrilinosInternalEigenSolver.cc
    )
generateAxl(ALIEN-ExternalPackages
    LinearSolver/Arcane/TrilinosSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
generateAxl(ALIEN-ExternalPackages
    EigenSolver/Arcane/TrilinosEigenSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
addSources(ALIEN-ExternalPackages
    LinearSolver/Arcane/TrilinosLinearSolver.cc
    )

addSources(ALIEN-ExternalPackages
    EigenSolver/Arcane/TrilinosEigenSolver.cc
    )
linkLibraries(ALIEN-ExternalPackages trilinos)
