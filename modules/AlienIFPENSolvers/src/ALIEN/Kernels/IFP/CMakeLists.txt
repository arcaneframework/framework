addSources(ALIEN-IFPENSolvers
    LinearSolver/IFPInternalLinearSolver.cc
    Converters/SimpleCSR/IFP_to_SimpleCSR_VectorConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_IFP_VectorConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_IFP_MatrixConverter.cc
    DataStructure/IFPMatrix.cc
    DataStructure/IFPVector.cc
    DataStructure/IFPSolverInternal.cc
    )
 
linkLibraries(ALIEN-IFPENSolvers ifpsolver)

generateAxl(ALIEN-IFPENSolvers
    LinearSolver/Arcane/IFPLinearSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )

addSources(ALIEN-IFPENSolvers
    LinearSolver/Arcane/IFPLinearSolverService.cc
    )
