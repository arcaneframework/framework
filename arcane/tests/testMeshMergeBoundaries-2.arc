<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <titre>Test Mesh Merge Boundaries (using legacy mesh handling)</titre>
  <timeloop>UnitTest</timeloop>
 </arcane>

  <mesh>
    <file internal-partition='true'>plancher.msh</file>
  </mesh>
  <mesh>
    <file internal-partition='true'>square_v41.msh</file>
  </mesh>

 <unit-test-module>
   <test name="MeshMergeBoundariesUnitTest" />
 </unit-test-module>

</case>
