if (NOT TARGET arcane_core)
    return()
endif ()

createLibrary(alien_arcane_tools
  EXPORT alien/AlienArcaneToolsExport.h
  )

addSources(alien_arcane_tools
  alien/arcane_tools/data/Space.cc
  alien/arcane_tools/accessors/ItemVectorAccessor.cc
  alien/arcane_tools/indexManager/SimpleAbstractFamily.cc
  alien/arcane_tools/indexManager/BasicIndexManager.cc
  alien/arcane_tools/indexManager/AnyItemAbstractFamily.cc
  alien/arcane_tools/block/BlockBuilder.cc
  alien/arcane_tools/block/BlockSizes.cc
  )

if(TARGET petsc)
  addSources(alien_arcane_tools 
    alien/arcane_tools/io/MatrixRowPrinter.cc
    )
  linkLibraries(alien_arcane_tools petsc)
  linkLibraries(alien_arcane_tools alien_external_packages)
endif()

linkLibraries(alien_arcane_tools Alien::alien_core arcane_core)

commit(alien_arcane_tools)

target_include_directories(alien_arcane_tools PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY alien 
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
  )
