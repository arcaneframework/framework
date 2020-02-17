if (NOT TARGET arcane_core OR NOT TARGET alien_hypre)
    return()
endif ()

createLibrary(ALIEN-OpensourceArcaneSolvers
        EXPORT ALIEN/ALIEN-OpensourceArcaneSolversExport.h
        )

set(AXL_OPTION_GENERATION_TYPE "ALL")
set(NAMESPACE_TYPE "Arccore")
set(IS_WITH_ARCANE "")
set(IS_WITH_MESH "NO_MESH")

generateAxl(ALIEN-OpensourceArcaneSolvers
        hypre/src/OSHypreSolver.axl
        AXL_OPTION_GENERATION_MODE ${AXL_OPTION_GENERATION_TYPE}
        NAMESPACE ${NAMESPACE_TYPE}
        ${IS_WITH_ARCANE}
        ${IS_WITH_MESH}
        INSTALL_GENERATED_FILES
        )

addSources(ALIEN-OpensourceArcaneSolvers
        hypre/src/OSHypreLinearSolverService.cc
        )

linkLibraries(ALIEN-OpensourceArcaneSolvers
        alien_hypre
        arcane_core
        arcane_std
        )

target_include_directories(ALIEN-OpensourceArcaneSolvers PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)

install(TARGETS alien_hypre
        DESTINATION lib
        EXPORT ALIENTargets
        )

commit(ALIEN-OpensourceArcaneSolvers)

add_subdirectory(hypre/test)
