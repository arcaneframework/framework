addSources(alien_ifpen_solvers
    LinearSolver/IFPInternalLinearSolver.cc
    Converters/SimpleCSR/IFP_to_SimpleCSR_VectorConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_IFP_VectorConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_IFP_MatrixConverter.cc
    DataStructure/IFPMatrix.cc
    DataStructure/IFPVector.cc
    DataStructure/IFPSolverInternal.cc
    )
 
linkLibraries(alien_ifpen_solvers ifpsolver)

generateAxl(alien_ifpen_solvers
    LinearSolver/Arcane/IFPLinearSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )

addSources(alien_ifpen_solvers
    LinearSolver/Arcane/IFPLinearSolverService.cc
    )
