addSources(alien_external_packages
        Algebra/HypreInternalLinearAlgebra.cc
        LinearSolver/HypreInternalLinearSolver.cc
        Converters/SimpleCSR/Hypre_to_SimpleCSR_VectorConverter.cc
        Converters/SimpleCSR/SimpleCSR_to_Hypre_VectorConverter.cc
        Converters/SimpleCSR/SimpleCSR_to_Hypre_MatrixConverter.cc
        DataStructure/HypreVector.cc
        DataStructure/HypreInternal.cc
        DataStructure/HypreMatrix.cc
        )

include(${AXLSTAR_CMAKE_COMMANDS})

generateAxl(alien_external_packages
        LinearSolver/Arcane/HypreSolver.axl
        AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
        NAMESPACE ${NAMESPACE_TYPE}
        ${IS_WITH_ARCANE}
        ${IS_WITH_MESH}
        INSTALL_GENERATED_FILES
        )

addSources(alien_external_packages
        LinearSolver/Arcane/HypreLinearSolver.cc
        )
#boost is necessary for timings :(
linkLibraries(alien_external_packages hypre Boost::boost)
