addSources(ALIEN-HPDDM
    Algebra/HPDDMInternalLinearAlgebra.cc
    DataStructure/HPDDMMatrix.cc
    LinearSolver/HPDDMInternalSolver.cc
    )
    
generateAxl(ALIEN-HPDDM
    LinearSolver/Arcane/HPDDMSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    )
addSources(ALIEN-HPDDM
    LinearSolver/Arcane/HPDDMSolverService.cc
    )

if(TARGET mumps)
    linkLibraries(ALIEN-HPDDM mumps)
endif()

if(TARGET arpack)
    linkLibraries(ALIEN-HPDDM arpack)
endif()

if(TARGET hpddm)
    linkLibraries(ALIEN-HPDDM hpddm)
endif()
