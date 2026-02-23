if (NOT TARGET COMPOSYX::composyx)
    return()
endif ()

createLibrary(alien_composyx
  EXPORT alien/AlienComposyxExport.h
  )

set_target_properties(alien_composyx
		      PROPERTIES LINKER_LANGUAGE "CXX")

if(TARGET arcane_core)
	set(AXL_OPTION_GENERATION_TYPE "ALL")
	set(NAMESPACE_TYPE "Arcane")
	set(IS_WITH_ARCANE "")
	set(IS_WITH_MESH "NO_MESH")
else()
	set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
	set(NAMESPACE_TYPE "Arccore")
	set(IS_WITH_ARCANE "NO_ARCANE")
	set(IS_WITH_MESH "NO_MESH")
endif()

addDirectory(alien/kernels/composyx)

if(TARGET arcane_core)
  linkLibraries(alien_composyx arcane_core arcane_std)
else ()
  linkLibraries(alien_composyx alien_strong_options)
endif()

linkLibraries(alien_composyx Alien::alien_core Alien::alien_semantic_ref)

target_include_directories(alien_composyx PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

commit(alien_composyx)

add_library(Alien::composyx ALIAS alien_composyx)

install(DIRECTORY alien
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
  )
