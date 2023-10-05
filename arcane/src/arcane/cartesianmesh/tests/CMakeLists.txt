arcane_add_component_test_executable(cartesianmesh
  FILES
  TestCartesianMeshV2.cc
  )

target_link_libraries(arcane_cartesianmesh.tests PUBLIC arcane_cartesianmesh GTest::GTest GTest::Main)

gtest_discover_tests(arcane_cartesianmesh.tests DISCOVERY_TIMEOUT 30)
