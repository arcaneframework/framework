addSources(alien_composyx
    algebra/ComposyxInternalLinearAlgebra.cc
    data_structure/ComposyxMatrix.cc
    data_structure/ComposyxVector.cc
    linear_solver/ComposyxInternalSolver.cc
    converters/SimpleCSR/Composyx_to_SimpleCSR_VectorConverter.cc
    converters/SimpleCSR/SimpleCSR_to_Composyx_VectorConverter.cc
    converters/SimpleCSR/SimpleCSR_to_Composyx_MatrixConverter.cc
    )
    
generateAxl(alien_composyx
    linear_solver/arcane/ComposyxSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    USER_INSTALL_PREFIX ALIEN
    )
addSources(alien_composyx
    linear_solver/arcane/ComposyxSolverService.cc
    )

if(TARGET mumps)
    linkLibraries(alien_composyx mumps)
endif()

if(TARGET arpack)
    linkLibraries(alien_composyx arpack)
endif()

if(TARGET COMPOSYX::composyx)
    linkLibraries(alien_composyx COMPOSYX::composyx)
endif()
