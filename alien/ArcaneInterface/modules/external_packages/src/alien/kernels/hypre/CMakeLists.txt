

addSources(alien_external_packages
        algebra/HypreInternalLinearAlgebra.cc
        linear_solver/HypreInternalLinearSolver.cc
        converters/SimpleCSR/Hypre_to_SimpleCSR_VectorConverter.cc
        converters/SimpleCSR/SimpleCSR_to_Hypre_VectorConverter.cc
        converters/SimpleCSR/SimpleCSR_to_Hypre_MatrixConverter.cc
        data_structure/HypreVector.cc
        data_structure/HypreInternal.cc
        data_structure/HypreMatrix.cc
        )

if(USE_AXLSTAR)
generateAxl(alien_external_packages
        linear_solver/arcane/HypreSolver.axl
        AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
        NAMESPACE ${NAMESPACE_TYPE}
        ${IS_WITH_ARCANE}
        ${IS_WITH_MESH}
        INSTALL_GENERATED_FILES
        USER_INSTALL_PREFIX ALIEN
        )
endif()

addSources(alien_external_packages
        linear_solver/arcane/HypreLinearSolver.cc
        )


#boost is necessary for timings :(
linkLibraries(alien_external_packages hypre Boost::boost)

IF(ALIEN_USE_SYCL)
    addSources(alien_external_packages
               converters/HCSR/HCSR_to_Hypre_VectorConverter.cc
               converters/HCSR/SYCL_to_Hypre_VectorConverter.cc
               converters/HCSR/Hypre_to_SYCL_VectorConverter.cc
               converters/HCSR/HCSR_to_Hypre_MatrixConverter.cc
              )
ENDIF()
