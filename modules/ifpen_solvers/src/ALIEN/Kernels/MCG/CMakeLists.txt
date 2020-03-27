addSources(alien_ifpen_solvers
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

generateAxl(alien_ifpen_solvers
    LinearSolver/Arcane/GPUSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )

addSources(alien_ifpen_solvers
    LinearSolver/Arcane/GPULinearSolver.cc
    )

linkLibraries(alien_ifpen_solvers mcgsolver nvamg hwloc metis hdf5)

if(TARGET harts)
  linkLibraries(alien_ifpen_solvers harts)
endif()

if(TARGET tbb)
  linkLibraries(alien_ifpen_solvers tbb)
endif()
