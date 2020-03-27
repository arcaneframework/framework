if (NOT TARGET arcane_core)
    return()
endif ()

createLibrary(alien_arcane_tools
  EXPORT alien/AlienArcaneToolsExport.h
  )

addSources(alien_arcane_tools
  ALIEN/ArcaneTools/Accessors/ItemVectorAccessor.cc
  #ALIEN/ArcaneTools/IndexManager/ArcGeoSim/ContactAbstractFamily.cc
  ALIEN/ArcaneTools/IndexManager/SimpleAbstractFamily.cc
  ALIEN/ArcaneTools/IndexManager/BasicIndexManager.cc
  ALIEN/ArcaneTools/IndexManager/AnyItemAbstractFamily.cc
  ALIEN/ArcaneTools/Block/IndexManagerBlockBuilder.cc
  )

if(TARGET petsc)
  addSources(alien_arcane_tools 
    ALIEN/ArcaneTools/IO/MatrixRowPrinter.cc
    )
  linkLibraries(alien_arcane_tools petsc)
  linkLibraries(alien_arcane_tools ALIEN-ExternalPackages)
endif()

linkLibraries(alien_arcane_tools alien)

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
