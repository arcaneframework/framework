set(SOURCE_FILES
  TestDataTypes.cc
  TestHashUniqueId.cc
  TestNodeReorderer.cc
)

arcane_add_component_test_executable(core
  FILES ${SOURCE_FILES}
  )

target_link_libraries(arcane_core.tests PUBLIC arcane_core GTest::GTest GTest::Main)

gtest_discover_tests(arcane_core.tests DISCOVERY_TIMEOUT 30)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
