createLibrary(alien_ifpen_solvers
  EXPORT alien/AlienIFPENSolversExport.h
  )

set_target_properties(alien_ifpen_solvers
		      PROPERTIES LINKER_LANGUAGE "CXX")

#Needed if no external packages are found !
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cc)
addSources(alien_ifpen_solvers ${CMAKE_CURRENT_BINARY_DIR}/empty.cc)

if (TARGET arcane_core)
    set(AXL_OPTION_GENERATION_TYPE "ALL")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "")
    set(IS_WITH_MESH "NO_MESH")
else ()
    set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "NO_ARCANE")
    set(IS_WITH_MESH "NO_MESH")
endif ()


if(USE_AXLSTAR)
    include(${AXLSTAR_CMAKE_COMMANDS})
endif()
if(TARGET mcgsolver)
    addDirectory(alien/kernels/mcg)
endif()

if(TARGET ifpsolver)
    addDirectory(alien/kernels/ifp)
endif()

if(TARGET htssolver)
    # This target need boost::system
    find_package(Boost REQUIRED COMPONENTS program_options system)
    addDirectory(alien/kernels/hts)
endif()

if(TARGET arcane_core)
    linkLibraries(alien_ifpen_solvers arcane_core arcane_std)
else ()
    linkLibraries(alien_ifpen_solvers alien_strong_options)
endif()

if(TARGET umfpack)
    linkLibraries(alien_ifpen_solvers umfpack)
endif()

if (TARGET Alien::alien_core)
    linkLibraries(alien_ifpen_solvers Alien::alien_core)
endif()

if (TARGET Alien::alien_semantic_ref)
    linkLibraries(alien_ifpen_solvers Alien::alien_semantic_ref)
endif()

linkLibraries(alien_ifpen_solvers Boost::program_options)
if(TARGET htssolver)
    target_link_libraries(alien_ifpen_solvers PRIVATE Boost::system)
endif()

target_include_directories(alien_ifpen_solvers PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

commit(alien_ifpen_solvers)

install(DIRECTORY alien
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
  )
