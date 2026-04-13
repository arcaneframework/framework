set(ARCANE_SOURCES
  GeometricUnitTest.cc
  IMaterialEquationOfState.h
)

set(ARCANE_MATERIAL_SOURCES
  HyodaMixedCellsUnitTest.cc
  MeshMaterialTesterModule.cc
  MeshMaterialTesterModule_Init.cc
  MeshMaterialTesterModule_Samples.cc
  MeshMaterialTesterModule.h
  MeshMaterialSyncUnitTest.cc
  MeshMaterialSimdUnitTest.cc
  MaterialHeatTestModule.cc
)

set(AXL_FILES 
  MaterialHeatTest
  MeshMaterialTester
  HyodaMixedCellsUnitTest
  GeometricUnitTest
  MeshMaterialSyncUnitTest
)
