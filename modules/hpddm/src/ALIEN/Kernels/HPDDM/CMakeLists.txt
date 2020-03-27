addSources(alien_hpddm
    Algebra/HPDDMInternalLinearAlgebra.cc
    DataStructure/HPDDMMatrix.cc
    LinearSolver/HPDDMInternalSolver.cc
    )
    
generateAxl(alien_hpddm
    LinearSolver/Arcane/HPDDMSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
addSources(alien_hpddm
    LinearSolver/Arcane/HPDDMSolverService.cc
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
