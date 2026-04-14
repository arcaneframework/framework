list(APPEND ARCANE_SOURCES
  material/IMaterialEquationOfState.h
  material/MaterialHeatTestModule.cc
  material/MeshMaterialSimdUnitTest.cc
  material/MeshMaterialSyncUnitTest.cc
  material/MeshMaterialTesterModule.cc
  material/MeshMaterialTesterModule_Init.cc
  material/MeshMaterialTesterModule_Samples.cc
  material/MeshMaterialTesterModule.h
)
list(APPEND ARCANE_ACCELERATOR_SOURCES
  material/MaterialHeatTestModule.cc
  material/MeshMaterialSyncUnitTest.cc
)
list(APPEND AXL_FILES
  material/MaterialHeatTest
  material/MeshMaterialSyncUnitTest
  material/MeshMaterialTester
)
