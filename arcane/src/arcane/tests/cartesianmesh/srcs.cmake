list(APPEND ARCANE_SOURCES
  cartesianmesh/AdiProjectionModule.cc
  cartesianmesh/AMRCartesianMeshTesterModule.cc
  cartesianmesh/AMRPatchTesterModule.cc
  cartesianmesh/CartesianMeshTesterModule.cc
  cartesianmesh/CartesianMeshTestUtils.cc
  cartesianmesh/CartesianMeshTestUtils.h
  cartesianmesh/CartesianMeshV2TestUtils.cc
  cartesianmesh/CartesianMeshV2TestUtils.h
  cartesianmesh/DynamicCircleAMRModule.cc
  cartesianmesh/UnitTestCartesianMeshPatch.cc
)
list(APPEND ARCANE_ACCELERATOR_SOURCES
  cartesianmesh/AdiProjectionModule.cc
  cartesianmesh/CartesianMeshTestUtils.cc
)
list(APPEND AXL_FILES
  cartesianmesh/AdiProjection
  cartesianmesh/AMRCartesianMeshTester
  cartesianmesh/AMRPatchTester
  cartesianmesh/CartesianMeshTester
  cartesianmesh/DynamicCircleAMR
  cartesianmesh/UnitTestCartesianMeshPatch
)
