  addSources(alien_external_packages
          algebra/MTLInternalLinearAlgebra.cc
          linear_solver/MTLInternalLinearSolver.cc
          converters/SimpleCSR/SimpleCSR_to_MTL_MatrixConverter.cc
          converters/SimpleCSR/MTL_to_SimpleCSR_VectorConverter.cc
          converters/SimpleCSR/SimpleCSR_to_MTL_VectorConverter.cc
          data_structure/MTLMatrix.cc
          data_structure/MTLVector.cc
          )


  if(USE_AXLSTAR)
  generateAxl(alien_external_packages
        linear_solver/arcane/MTLLinearSolver.axl
        AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
        NAMESPACE ${NAMESPACE_TYPE}
        ${IS_WITH_ARCANE}
        ${IS_WITH_MESH}
        INSTALL_GENERATED_FILES
        )
  endif()
  
  addSources(alien_external_packages
          linear_solver/arcane/MTLLinearSolverService.cc
          )

  linkLibraries(alien_external_packages mtl4 Boost::boost)
