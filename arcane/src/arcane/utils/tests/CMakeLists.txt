set(SOURCE_FILES
  TestAutoRef.cc
  TestDependencyInjection.cc
  TestEvent.cc
  TestRealN.cc
  TestNumVector.cc
  TestValueConvert.cc
  TestCollections.cc
  TestLinearOffsetMap.cc
  TestHash.cc
  TestHashTable.cc
  TestMemory.cc
  TestMemoryPool.cc
  TestPlatform.cc
  TestVector2.cc
  TestVector3.cc
)

if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND SOURCE_FILES
    TestCxx20.cc
    TestNumArray.cc
  )
endif()

arcane_add_component_test_executable(utils
  FILES ${SOURCE_FILES}
  )

target_link_libraries(arcane_utils.tests PUBLIC arcane_utils GTest::GTest GTest::Main)

gtest_discover_tests(arcane_utils.tests DISCOVERY_TIMEOUT 30)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
