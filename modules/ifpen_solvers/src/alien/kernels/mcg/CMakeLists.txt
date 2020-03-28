addSources(alien_ifpen_solvers
    algebra/MCGInternalLinearAlgebra.cc
    linear_solver/MCGInternalLinearSolver.cc
    converters/SimpleCSR/MCG_to_SimpleCSR_VectorConverter.cc
    converters/SimpleCSR/SimpleCSR_to_MCG_VectorConverter.cc
    converters/SimpleCSR/SimpleCSR_to_MCG_MatrixConverter.cc
    converters/Composite/Composite_to_MCG_VectorConverter.cc
    converters/Composite/Composite_to_MCG_MatrixConverter.cc
    data_structure/MCGVector.cc
    data_structure/MCGMatrix.cc
    )

generateAxl(alien_ifpen_solvers
    linear_solver/arcane/GPUSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )

addSources(alien_ifpen_solvers
    linear_solver/arcane/GPULinearSolver.cc
    )

linkLibraries(alien_ifpen_solvers mcgsolver nvamg hwloc metis hdf5)

if(TARGET harts)
  linkLibraries(alien_ifpen_solvers harts)
endif()

if(TARGET tbb)
  linkLibraries(alien_ifpen_solvers tbb)
endif()
