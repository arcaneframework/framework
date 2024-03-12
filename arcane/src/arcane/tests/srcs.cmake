set(ARCANE_SOURCES
  CheckpointTesterService.cc
  JSONUnitTest.cc
  XmlUnitTest.cc
  SingletonService.cc
  ParticleUnitTest.cc
  TestUnitTest.cc
  CaseFunctionUnitTest.cc
  CaseFunctionTesterModule.cc
  CaseOptionsTesterModule.cc
  ParallelTesterModule.cc
  SubMeshTestModule.cc
  HydroAdditionalTestModule.cc
  TestTplParameter.cc
  ModuleSimpleHydro.cc
  ModuleSimpleHydroGeneric.cc
  ModuleSimpleHydroSimd.cc
  ModuleSimpleHydroDepend.cc
  MeshMergeBoundariesUnitTest.cc
  MeshMergeNodesUnitTest.cc
  MeshUnitTest.cc
  MultipleMeshUnitTest.cc
  ThreadUnitTest.cc
  TaskUnitTestCS.cc
  IosUnitTest.cc
  VariableUnitTest.cc
  VariableSimdUnitTest.cc
  AMRTestModule.cc
  AMR/ErrorEstimate.cc
  AMR/ErrorEstimate.h
  MatVecUnitTest.cc
  MiniWeatherTypes.h
  MiniWeatherModule.cc
  StdArrayMeshVariables.cc
  StdArrayMeshVariables.h
  StdMeshVariables.cc
  StdMeshVariables.h
  StdScalarVariables.cc
  StdScalarVariables.h
  StdScalarMeshVariables.cc
  StdScalarMeshVariables.h
  PartialVariableTester.cc
  RandomUnitTest.cc
  ArrayUnitTest.cc
  PropertiesUnitTest.cc
  ItemVectorUnitTest.cc
  ConfigurationUnitTest.cc
  VoronoiTest.cc
  UtilsUnitTest.cc
  SimdUnitTest.cc
  ParallelMngTest.cc
  ParallelMngDataTypeTest.cc
  SingletonServiceTestModule.cc
  TimeHistoryTestModule.cc
  MeshModificationTester.cc
  DirectedGraphUnitTest.cc
  ExchangeItemsUnitTest.cc
  anyitem/AnyItemTester.cc
  dof/DoFTester.cc
  dof/DoFNodeTestService.cc
  graph/GraphUnitTest.cc
  inout/InOutTester.cc
  geometry/GeometryUnitTest.cc
  StdArrayMeshVariables.h
  StdMeshVariables.h
  StdScalarVariables.h
  StdScalarMeshVariables.h
  IServiceInterface.h
  ArcaneTestInit.cc
  ArcaneTestDirectExecution.cc
  ArcaneTestStandaloneSubDomain.cc
  CustomMeshTestModule.cc
  TaskUnitTest.cc
  PDESRandomNumberGeneratorUnitTest.cc
  RandomNumberGeneratorUnitTest.cc
  SimpleTableOutputUnitTest.cc
  SimpleTableComparatorUnitTest.cc
  StringVariableReplaceTest.cc
  TimeHistoryAdderTestModule.cc
)

set(AXL_FILES
  CheckpointTester
  SimpleHydro
  ScriptTester
  CaseFunctionTester
  CaseOptionsTester
  ParallelTester
  SubMeshTest
  SingletonService
  ParticleUnitTest
  TestUnitTest
  MeshMergeNodesUnitTest
  MeshUnitTest
  ThreadUnitTest
  TaskUnitTest
  IosUnitTest
  MDVariableUnitTest
  VariableUnitTest
  AMRTest
  MatVecUnitTest
  MiniWeather
  PartialVariableTester
  VoronoiTest
  SingletonServiceTest
  TimeHistoryTest
  MeshModificationTester
  DirectedGraphUnitTest
  ExchangeItemsUnitTest
  anyitem/AnyItemTester
  dof/DoFTester
  graph/GraphUnitTest
  inout/InOutTester
  geometry/GeometryUnitTest
  CustomMeshTest
  HydroAdditionalTest
  accelerator/SimpleHydroAccelerator
  accelerator/AcceleratorReduceUnitTest
  accelerator/AcceleratorScanUnitTest
  accelerator/AcceleratorFilterUnitTest
  PDESRandomNumberGeneratorUnitTest
  RandomNumberGeneratorUnitTest
  ServiceInterface1ImplTest
  ServiceInterface5ImplTest
  SimpleTableOutputUnitTest
  SimpleTableComparatorUnitTest
  SimpleTableOutputUnitTest
  StringVariableReplaceTest
  TimeHistoryAdderTest
)

