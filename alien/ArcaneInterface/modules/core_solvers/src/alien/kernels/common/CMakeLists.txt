generateAxl(alien_core_solvers
    linear_solver/arcane/AlienCoreSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    USER_INSTALL_PREFIX ALIEN
    )
    
addSources(alien_core_solvers
    linear_solver/arcane/AlienLinearSolver.cc
    )
    
if(ALIEN_USE_SYCL)
addSources(alien_core_solvers
    linear_solver/arcane/AlienSYCLLinearSolver.cc
    )
endif()