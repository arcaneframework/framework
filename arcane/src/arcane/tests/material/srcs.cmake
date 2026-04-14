list(APPEND ARCANE_SOURCES
  material/MaterialHeatTestModule.cc
  material/MeshMaterialSimdUnitTest.cc
  material/MeshMaterialSyncUnitTest.cc
)
list(APPEND ARCANE_ACCELERATOR_SOURCES
  material/MaterialHeatTestModule.cc
  material/MeshMaterialSyncUnitTest.cc
)
list(APPEND AXL_FILES
  material/MaterialHeatTest
  material/MeshMaterialSyncUnitTest
)
