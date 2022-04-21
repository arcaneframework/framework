if (NOT TARGET trilinos)
    return()
endif ()

createLibrary(alien_trilinos
        EXPORT alien/AlienTrilinosExport.h
        )

set_target_properties(alien_trilinos
        PROPERTIES LINKER_LANGUAGE "CXX")

if(ENABLE_CUDA)
target_compile_options(alien_trilinos PRIVATE "-arch=sm_80 --extended-lambda ")
endif()

if (TARGET arcane_core)
    set(AXL_OPTION_GENERATION_TYPE "ALL")
    set(NAMESPACE_TYPE "Arcane")
    set(IS_WITH_ARCANE "")
    set(IS_WITH_MESH "NO_MESH")
else ()
    set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "NO_ARCANE")
    set(IS_WITH_MESH "NO_MESH")
endif ()


if (TARGET trilinos)
    addDirectory(alien/kernels/trilinos)
endif ()

if (TARGET arcane_core)
    linkLibraries(alien_trilinos arcane_core arcane_std)
else ()
    linkLibraries(alien_trilinos alien_strong_options)
endif ()

if(TARGET mkl)
  linkLibraries(alien_trilinos mkl)
endif()

if (TARGET umfpack)
    linkLibraries(alien_trilinos umfpack)
endif ()

if(TARGET nvamg)
  linkLibraries(alien_trilinos nvamg)
endif()

linkLibraries(alien_trilinos Alien::alien_core Alien::alien_semantic_ref)

target_include_directories(alien_trilinos PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)

commit(alien_trilinos)

install(DIRECTORY alien
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        )
