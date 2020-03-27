  addSources(alien_external_packages
          Algebra/MTLInternalLinearAlgebra.cc
          LinearSolver/MTLInternalLinearSolver.cc
          Converters/SimpleCSR/SimpleCSR_to_MTL_MatrixConverter.cc
          Converters/SimpleCSR/MTL_to_SimpleCSR_VectorConverter.cc
          Converters/SimpleCSR/SimpleCSR_to_MTL_VectorConverter.cc
          DataStructure/MTLMatrix.cc
          DataStructure/MTLVector.cc
          )

  generateAxl(alien_external_packages
          LinearSolver/Arcane/MTLLinearSolver.axl
          AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
          NAMESPACE ${NAMESPACE_TYPE}
          ${IS_WITH_ARCANE}
          ${IS_WITH_MESH}
          INSTALL_GENERATED_FILES
          )

  addSources(alien_external_packages
          LinearSolver/Arcane/MTLLinearSolverService.cc
          )

  linkLibraries(alien_external_packages mtl Boost::boost)