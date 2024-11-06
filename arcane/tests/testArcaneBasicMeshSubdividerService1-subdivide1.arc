<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0" xml:lang="en">
  <arcane>
    <title>Test subdivision d'un maillage 3D hexahédrique</title>
    <description>Subdivision d'un maillage 3D uniquement hexahédrique avec le partitionneur par défaut</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>sod.vtk</filename>
      <partitioner>MeshPartitionerTester</partitioner>
      <subdivider>
          <nb-subdivision>1</nb-subdivision>
      </subdivider>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest">
      <test-adjency>0</test-adjency>
    </test>
  </unit-test-module>

</case>
