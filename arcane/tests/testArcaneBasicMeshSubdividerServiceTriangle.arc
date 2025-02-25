<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0" xml:lang="en">
  <arcane>
    <title>Test subdivision d'un maillage 2D triangulaire</title>
    <description>Subdivision d'un maillage 2D uniquement triangulaire avec le partitionneur par défaut</description>
    <timeloop>UnitTest</timeloop>
  </arcane>
  <meshes>
    <mesh>
      <filename>plancher.msh</filename>
      <partitioner>MeshPartitionerTester</partitioner>
      <subdivider>
          <nb-subdivision>1</nb-subdivision>
      </subdivider>
    </mesh>
  </meshes>
  <unit-test-module>
    <test name="MeshUnitTest">
    <test-adjency>1</test-adjency>
    <write-mesh>false</write-mesh>
    </test>
  </unit-test-module>
</case>
