addSources(alien_ifpen_solvers
    linear_solver/IFPInternalLinearSolver.cc
    converters/SimpleCSR/IFP_to_SimpleCSR_VectorConverter.cc
    converters/SimpleCSR/SimpleCSR_to_IFP_VectorConverter.cc
    converters/SimpleCSR/SimpleCSR_to_IFP_MatrixConverter.cc
    data_structure/IFPMatrix.cc
    data_structure/IFPVector.cc
    data_structure/IFPSolverInternal.cc
    )
 
linkLibraries(alien_ifpen_solvers ifpsolver)

generateAxl(alien_ifpen_solvers
    linear_solver/arcane/IFPLinearSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )

addSources(alien_ifpen_solvers
    linear_solver/arcane/IFPLinearSolverService.cc
    )

if(WIN32)
    target_compile_definitions(alien_ifpen_solvers PUBLIC WIN32)
else()
    target_compile_definitions(alien_ifpen_solvers PUBLIC arch_Linux)
endif()