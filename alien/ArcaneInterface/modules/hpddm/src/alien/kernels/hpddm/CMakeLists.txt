addSources(alien_hpddm
    algebra/HPDDMInternalLinearAlgebra.cc
    data_structure/HPDDMMatrix.cc
    linear_solver/HPDDMInternalSolver.cc
    )
    
generateAxl(alien_hpddm
    linear_solver/arcane/HPDDMSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    USER_INSTALL_PREFIX ALIEN
    )
addSources(alien_hpddm
    linear_solver/arcane/HPDDMSolverService.cc
    )

if(TARGET mumps)
    linkLibraries(alien_hpddm mumps)
endif()

if(TARGET arpack)
    linkLibraries(alien_hpddm arpack)
endif()

if(TARGET hpddm)
    linkLibraries(alien_hpddm hpddm)
endif()
