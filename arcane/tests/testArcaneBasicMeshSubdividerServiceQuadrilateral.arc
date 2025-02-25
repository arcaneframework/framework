<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0" xml:lang="en">
  <arcane>
    <title>Test subdivision d'un maillage 2D quadrangulaire</title>
    <description>Subdivision d'un maillage 2D uniquement quadrangulaire avec le partitionneur par défaut</description>
    <timeloop>UnitTest</timeloop>
  </arcane>
  <meshes>
    <mesh>
      <filename>planar_unstructured_quad1.msh</filename>
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
