if (NOT TARGET arcane_core OR NOT TARGET alien_hypre)
    return()
endif ()

createLibrary(alien_plugins_solvers
        EXPORT alien/AlienOpensourceArcaneSolversExport.h
        )

set(AXL_OPTION_GENERATION_TYPE "ALL")
set(NAMESPACE_TYPE "Arccore")
set(IS_WITH_ARCANE "")
set(IS_WITH_MESH "NO_MESH")

generateAxl(alien_plugins_solvers
        hypre/src/OSHypreSolver.axl
        AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
        NAMESPACE ${NAMESPACE_TYPE}
        ${IS_WITH_ARCANE}
        ${IS_WITH_MESH}
        INSTALL_GENERATED_FILES
        USER_INSTALL_PREFIX ALIEN
        )

addSources(alien_plugins_solvers
        hypre/src/OSHypreLinearSolverService.cc
        )

linkLibraries(alien_plugins_solvers
        alien_hypre
        arcane_core
        arcane_std
        )

target_include_directories(alien_plugins_solvers PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)

install(TARGETS alien_hypre
        DESTINATION lib
        EXPORT ALIENTargets
        )

commit(alien_plugins_solvers)

add_subdirectory(hypre/test)
