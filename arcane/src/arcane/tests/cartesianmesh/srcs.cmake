list(APPEND ARCANE_SOURCES
  cartesianmesh/AMRCartesianMeshTesterModule.cc
  cartesianmesh/CartesianMeshTesterModule.cc
  cartesianmesh/CartesianMeshTestUtils.cc
  cartesianmesh/CartesianMeshTestUtils.h
  cartesianmesh/CartesianMeshV2TestUtils.cc
  cartesianmesh/CartesianMeshV2TestUtils.h
  cartesianmesh/UnitTestCartesianMeshPatch.cc
)
list(APPEND AXL_FILES
  cartesianmesh/AMRCartesianMeshTester
  cartesianmesh/CartesianMeshTester
  cartesianmesh/UnitTestCartesianMeshPatch
)

if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND ARCANE_SOURCES
    cartesianmesh/AdiProjectionModule.cc
  )
  list(APPEND AXL_FILES
    cartesianmesh/AdiProjection
  )
  list(APPEND ARCANE_ACCELERATOR_SOURCES
    cartesianmesh/AdiProjectionModule.cc
    cartesianmesh/CartesianMeshTestUtils.cc
  )
endif()
