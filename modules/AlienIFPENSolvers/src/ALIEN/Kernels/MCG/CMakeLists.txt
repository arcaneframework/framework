addSources(ALIEN-IFPENSolvers
    Algebra/MCGInternalLinearAlgebra.cc
    LinearSolver/MCGInternalLinearSolver.cc
    Converters/SimpleCSR/MCG_to_SimpleCSR_VectorConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_MCG_VectorConverter.cc
    Converters/SimpleCSR/SimpleCSR_to_MCG_MatrixConverter.cc
    Converters/Composite/Composite_to_MCG_VectorConverter.cc
    Converters/Composite/Composite_to_MCG_MatrixConverter.cc
    DataStructure/MCGVector.cc
    DataStructure/MCGMatrix.cc
    )

generateAxl(ALIEN-IFPENSolvers
    LinearSolver/Arcane/GPUSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )

addSources(ALIEN-IFPENSolvers
    LinearSolver/Arcane/GPULinearSolver.cc
    )

linkLibraries(ALIEN-IFPENSolvers mcgsolver nvamg hwloc metis hdf5)

if(TARGET harts)
  linkLibraries(ALIEN-IFPENSolvers harts)
endif()

if(TARGET tbb)
  linkLibraries(ALIEN-IFPENSolvers tbb)
endif()
