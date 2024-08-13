set(CMAKE_CXX_COMPILER "$ENV{TRILINOS_ROOT}/bin/nvcc_wrapper")



if (TARGET  trilinos)
    createLibrary(alien_belos)
    addSources(alien_belos belos_solver_fabric.cpp)
    linkLibraries(alien_belos trilinos)
    commit(alien_belos)
    target_include_directories(alien_belos PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif ()




addSources(alien_trilinos
    algebra/TrilinosInternalLinearAlgebra.cc
    linear_solver/TrilinosInternalLinearSolver.cc
    converters/SimpleCSR/SimpleCSR_to_Trilinos_MatrixConverter.cc
    converters/SimpleCSR/SimpleCSR_to_Trilinos_VectorConverter.cc
    converters/SimpleCSR/Trilinos_to_SimpleCSR_VectorConverter.cc
    data_structure/TrilinosInternal.cc
    data_structure/TrilinosMatrix.cc
    data_structure/TrilinosVector.cc
    )
addSources(alien_trilinos
    eigen_solver/TrilinosInternalEigenSolver.cc
    )
generateAxl(alien_trilinos
    linear_solver/arcane/TrilinosSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    USER_INSTALL_PREFIX ALIEN
    )
generateAxl(alien_trilinos
    eigen_solver/arcane/TrilinosEigenSolver.axl
    AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
    NAMESPACE ${NAMESPACE_TYPE}
    ${IS_WITH_ARCANE}
    ${IS_WITH_MESH}
    INSTALL_GENERATED_FILES
    USER_INSTALL_PREFIX ALIEN
    )
addSources(alien_trilinos
    linear_solver/arcane/TrilinosLinearSolver.cc
    )

addSources(alien_trilinos
    eigen_solver/arcane/TrilinosEigenSolver.cc
    )


if (TARGET alien_belos)
    linkLibraries(alien_trilinos alien_belos)
endif()

if (TARGET trilinos)
    linkLibraries(alien_trilinos trilinos)
endif()

#if(TARGET nvamg)
#linkLibraries(alien_trilinos nvamg)
#endif()
