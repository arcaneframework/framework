  addSources(ALIEN-ExternalPackages
          Algebra/PETScInternalLinearAlgebra.cc
          LinearSolver/PETScInitType.cc
          LinearSolver/PETScInternalLinearSolver.cc
          Converters/SimpleCSR/SimpleCSR_to_PETSc_MatrixConverter.cc
          Converters/SimpleCSR/PETSc_to_SimpleCSR_VectorConverter.cc
          Converters/SimpleCSR/SimpleCSR_to_PETSc_VectorConverter.cc
          DataStructure/PETScInit.cc
          DataStructure/PETScMatrix.cc
          DataStructure/PETScInternal.cc
          DataStructure/PETScVector.cc
          IO/AsciiDumper.cc
          )

  linkLibraries(ALIEN-ExternalPackages petsc)

  generateAxl(ALIEN-ExternalPackages
          LinearSolver/SPAI/PETScPrecConfigSPAI.axl
          LinearSolver/Arcane/PETScSolverConfigBiCGStab.axl
          LinearSolver/Arcane/PETScPrecConfigAdditiveSchwarz.axl
          LinearSolver/Arcane/PETScSolverConfigRichardson.axl
          LinearSolver/Arcane/PETScSolverConfigGMRES.axl
          LinearSolver/Arcane/PETScLinearSolver.axl
          LinearSolver/Arcane/PETScPrecConfigNoPreconditioner.axl
          LinearSolver/Arcane/PETScSolverConfigCustom.axl
          LinearSolver/Arcane/PETScPrecConfigJacobi.axl
          LinearSolver/Arcane/PETScPrecConfigDiagonal.axl
          LinearSolver/Arcane/PETScSolverConfigCG.axl
          LinearSolver/Arcane/PETScSolverConfigLU.axl
          LinearSolver/FieldSplit/SchurFieldSplitType.axl
          LinearSolver/FieldSplit/PETScPrecConfigFieldSplit.axl
          LinearSolver/FieldSplit/MultiplicativeFieldSplitType.axl
          LinearSolver/FieldSplit/AdditiveFieldSplitType.axl
          LinearSolver/FieldSplit/SymmetricMultiplicativeFieldSplitType.axl
          AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
          NAMESPACE ${NAMESPACE_TYPE}
          ${IS_WITH_ARCANE}
          ${IS_WITH_MESH}
          INSTALL_GENERATED_FILES
          )

  addSources(ALIEN-ExternalPackages
          LinearSolver/SPAI/PETScPrecConfigSPAIService.cc
          LinearSolver/Arcane/PETScSolverConfigBiCGStabService.cc
          LinearSolver/Arcane/PETScPrecConfigAdditiveSchwarzService.cc
          LinearSolver/Arcane/PETScSolverConfigRichardsonService.cc
          LinearSolver/Arcane/PETScSolverConfigGMRESService.cc
          LinearSolver/Arcane/PETScLinearSolverService.cc
          LinearSolver/Arcane/PETScPrecConfigNoPreconditionerService.cc
          LinearSolver/Arcane/PETScSolverConfigCustomService.cc
          LinearSolver/Arcane/PETScPrecConfigJacobiService.cc
          LinearSolver/Arcane/PETScPrecConfigDiagonalService.cc
          LinearSolver/Arcane/PETScSolverConfigCGService.cc
          LinearSolver/Arcane/PETScSolverConfigLUService.cc
          LinearSolver/FieldSplit/SchurFieldSplitTypeService.cc
          LinearSolver/FieldSplit/PETScPrecConfigFieldSplitService.cc
          LinearSolver/FieldSplit/MultiplicativeFieldSplitTypeService.cc
          LinearSolver/FieldSplit/AdditiveFieldSplitTypeService.cc
          LinearSolver/FieldSplit/SymmetricMultiplicativeFieldSplitTypeService.cc
          )

  if (TARGET slepc)
      generateAxl(ALIEN-ExternalPackages
              EigenSolver/Arcane/SLEPcEigenSolver.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              )
      addSources(ALIEN-ExternalPackages
              EigenSolver/SLEPcInternalEigenSolver.cc
              EigenSolver/Arcane/SLEPcEigenSolver.cc)
      linkLibraries(ALIEN-ExternalPackages slepc)

  endif (TARGET slepc)

  if (TARGET hypre)
      generateAxl(ALIEN-ExternalPackages
              LinearSolver/Hypre/PETScPrecConfigHypreEuclid.axl
              LinearSolver/Hypre/PETScPrecConfigHypre.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              )
      addSources(ALIEN-ExternalPackages
              LinearSolver/Hypre/PETScPrecConfigHypreEuclidService.cc
              LinearSolver/Hypre/PETScPrecConfigHypreService.cc
              )
  endif (TARGET hypre)

  if (TARGET superlu)
      generateAxl(ALIEN-ExternalPackages
              LinearSolver/SuperLU/PETScSolverConfigSuperLU.axl
              LinearSolver/SuperLU/PETScPrecConfigSuperLU.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              )
      addSources(ALIEN-ExternalPackages
              LinearSolver/SuperLU/PETScSolverConfigSuperLUService.cc
              LinearSolver/SuperLU/PETScPrecConfigSuperLUService.cc
              )
  endif (TARGET superlu)

  if (TARGET superlu)
      linkLibraries(ALIEN-ExternalPackages superlu)
  endif ()

  if (TARGET superludist)
      linkLibraries(ALIEN-ExternalPackages superludist)
  endif ()

  if (TARGET metis)
      linkLibraries(ALIEN-ExternalPackages metis)
  endif ()

  if (TARGET fftw3)
      linkLibraries(ALIEN-ExternalPackages fftw3)
  endif ()
