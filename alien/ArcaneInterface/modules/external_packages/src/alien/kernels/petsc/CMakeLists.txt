addSources(alien_external_packages
          algebra/PETScInternalLinearAlgebra.cc
          linear_solver/PETScInitType.cc
          linear_solver/PETScInternalLinearSolver.cc
          converters/SimpleCSR/SimpleCSR_to_PETSc_MatrixConverter.cc
          converters/SimpleCSR/PETSc_to_SimpleCSR_VectorConverter.cc
          converters/SimpleCSR/SimpleCSR_to_PETSc_VectorConverter.cc
          data_structure/PETScInit.cc
          data_structure/PETScMatrix.cc
          data_structure/PETScInternal.cc
          data_structure/PETScVector.cc
          io/AsciiDumper.cc
          )

if(ALIEN_USE_SYCL)
  addSources(alien_external_packages
             converters/HCSR/HCSR_to_PETSc_VectorConverter.cc
             converters/HCSR/SYCL_to_PETSc_VectorConverter.cc
             converters/HCSR/HCSR_to_PETSc_MatrixConverter.cc
            )
endif()

linkLibraries(alien_external_packages petsc)
  
if(PETSc_USE_PKGCONFIG)
    if(PKG_PETSC_FOUND) 
      target_link_options(alien_external_packages PRIVATE ${PKG_PETSC_STATIC_LDFLAGS})
     endif()
endif()  


if(USE_AXLSTAR)
  generateAxl(alien_external_packages
          linear_solver/spai/PETScPrecConfigSPAI.axl
          linear_solver/arcane/PETScSolverConfigBiCGStab.axl
          linear_solver/arcane/PETScPrecConfigAdditiveSchwarz.axl
          linear_solver/arcane/PETScSolverConfigRichardson.axl
          linear_solver/arcane/PETScSolverConfigGMRES.axl
          linear_solver/arcane/PETScLinearSolver.axl
          linear_solver/arcane/PETScPrecConfigNoPreconditioner.axl
          linear_solver/arcane/PETScSolverConfigCustom.axl
          linear_solver/arcane/PETScPrecConfigJacobi.axl
          linear_solver/arcane/PETScPrecConfigDiagonal.axl
          linear_solver/arcane/PETScPrecConfigGAMG.axl
          linear_solver/arcane/PETScSolverConfigCG.axl
          linear_solver/arcane/PETScSolverConfigLU.axl
          linear_solver/field_split/SchurFieldSplitType.axl
          linear_solver/field_split/PETScPrecConfigFieldSplit.axl
          linear_solver/field_split/MultiplicativeFieldSplitType.axl
          linear_solver/field_split/AdditiveFieldSplitType.axl
          linear_solver/field_split/SymmetricMultiplicativeFieldSplitType.axl
          AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
          NAMESPACE ${NAMESPACE_TYPE}
          ${IS_WITH_ARCANE}
          ${IS_WITH_MESH}
          INSTALL_GENERATED_FILES
          USER_INSTALL_PREFIX ALIEN
          )
endif()

addSources(alien_external_packages
          linear_solver/spai/PETScPrecConfigSPAIService.cc
          linear_solver/arcane/PETScSolverConfigBiCGStabService.cc
          linear_solver/arcane/PETScPrecConfigAdditiveSchwarzService.cc
          linear_solver/arcane/PETScSolverConfigRichardsonService.cc
          linear_solver/arcane/PETScSolverConfigGMRESService.cc
          linear_solver/arcane/PETScLinearSolverService.cc
          linear_solver/arcane/PETScPrecConfigNoPreconditionerService.cc
          linear_solver/arcane/PETScSolverConfigCustomService.cc
          linear_solver/arcane/PETScPrecConfigJacobiService.cc
          linear_solver/arcane/PETScPrecConfigDiagonalService.cc
          linear_solver/arcane/PETScPrecConfigGAMGService.cc
          linear_solver/arcane/PETScSolverConfigCGService.cc
          linear_solver/arcane/PETScSolverConfigLUService.cc
          linear_solver/field_split/SchurFieldSplitTypeService.cc
          linear_solver/field_split/PETScPrecConfigFieldSplitService.cc
          linear_solver/field_split/MultiplicativeFieldSplitTypeService.cc
          linear_solver/field_split/AdditiveFieldSplitTypeService.cc
          linear_solver/field_split/SymmetricMultiplicativeFieldSplitTypeService.cc
          )

if (TARGET slepc)
  if (USE_AXLSTAR)
      generateAxl(alien_external_packages
              eigen_solver/arcane/SLEPcEigenSolver.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              USER_INSTALL_PREFIX ALIEN
              )
  endif()
  addSources(alien_external_packages
              eigen_solver/SLEPcInternalEigenSolver.cc
              eigen_solver/arcane/SLEPcEigenSolver.cc)
      linkLibraries(alien_external_packages slepc)

endif (TARGET slepc)

if (TARGET hypre)
  if (USE_AXLSTAR)
      generateAxl(alien_external_packages
              linear_solver/hypre/PETScPrecConfigHypreEuclid.axl
              linear_solver/hypre/PETScPrecConfigHypre.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              USER_INSTALL_PREFIX ALIEN
              )
  endif()
  addSources(alien_external_packages
             linear_solver/hypre/PETScPrecConfigHypreEuclidService.cc
             linear_solver/hypre/PETScPrecConfigHypreService.cc
            )
              
endif (TARGET hypre)

  if (TARGET superlu)
      if(USE_AXLSTAR)
      generateAxl(alien_external_packages
              linear_solver/super_lu/PETScSolverConfigSuperLU.axl
              linear_solver/super_lu/PETScPrecConfigSuperLU.axl
              AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
              NAMESPACE ${NAMESPACE_TYPE}
              ${IS_WITH_ARCANE}
              ${IS_WITH_MESH}
              INSTALL_GENERATED_FILES
              USER_INSTALL_PREFIX ALIEN
              )
      endif()
      addSources(alien_external_packages
              linear_solver/super_lu/PETScSolverConfigSuperLUService.cc
              linear_solver/super_lu/PETScPrecConfigSuperLUService.cc
              )
  endif (TARGET superlu)

if (TARGET petsc::mumps)
  if(USE_AXLSTAR)
          generateAxl(alien_external_packages
                  linear_solver/mumps/PETScSolverConfigMUMPS.axl
                  linear_solver/mumps/PETScPrecConfigMUMPS.axl
                  AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
                  NAMESPACE ${NAMESPACE_TYPE}
                  ${IS_WITH_ARCANE}
                  ${IS_WITH_MESH}
                  INSTALL_GENERATED_FILES
                  USER_INSTALL_PREFIX ALIEN
          )
  endif()
  addSources(alien_external_packages
              linear_solver/mumps/PETScSolverConfigMUMPSService.cc
              linear_solver/mumps/PETScPrecConfigMUMPSService.cc
            )
endif (TARGET petsc::mumps)

if (TARGET hypre)
      linkLibraries(alien_external_packages hypre)
endif ()

if (TARGET superlu)
      linkLibraries(alien_external_packages superlu)
endif ()

if (TARGET superludist)
      linkLibraries(alien_external_packages superludist)
endif ()

if (TARGET mumps)
      linkLibraries(alien_external_packages mumps)
endif ()

if (TARGET metis)
      linkLibraries(alien_external_packages metis)
endif ()

if (TARGET fftw3)
      linkLibraries(alien_external_packages fftw3)
endif ()
