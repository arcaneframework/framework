  addSources(alien_external_packages
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

  linkLibraries(alien_external_packages petsc)

  generateAxl(alien_external_packages
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

  addSources(alien_external_packages
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
      generateAxl(alien_external_packages
              EigenSolver/Arcane/SLEPcEigenSolver.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              )
      addSources(alien_external_packages
              EigenSolver/SLEPcInternalEigenSolver.cc
              EigenSolver/Arcane/SLEPcEigenSolver.cc)
      linkLibraries(alien_external_packages slepc)

  endif (TARGET slepc)

  if (TARGET hypre)
      generateAxl(alien_external_packages
              LinearSolver/Hypre/PETScPrecConfigHypreEuclid.axl
              LinearSolver/Hypre/PETScPrecConfigHypre.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              )
      addSources(alien_external_packages
              LinearSolver/Hypre/PETScPrecConfigHypreEuclidService.cc
              LinearSolver/Hypre/PETScPrecConfigHypreService.cc
              )
  endif (TARGET hypre)

  if (TARGET superlu)
      generateAxl(alien_external_packages
              LinearSolver/SuperLU/PETScSolverConfigSuperLU.axl
              LinearSolver/SuperLU/PETScPrecConfigSuperLU.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              )
      addSources(alien_external_packages
              LinearSolver/SuperLU/PETScSolverConfigSuperLUService.cc
              LinearSolver/SuperLU/PETScPrecConfigSuperLUService.cc
              )
  endif (TARGET superlu)

  if (TARGET superlu)
      linkLibraries(alien_external_packages superlu)
  endif ()

  if (TARGET superludist)
      linkLibraries(alien_external_packages superludist)
  endif ()

  if (TARGET metis)
      linkLibraries(alien_external_packages metis)
  endif ()

  if (TARGET fftw3)
      linkLibraries(alien_external_packages fftw3)
  endif ()
