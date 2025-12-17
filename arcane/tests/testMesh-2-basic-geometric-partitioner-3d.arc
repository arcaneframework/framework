<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <titre>Test basic geometric mesh partitioner</titre>
    <description>Test basic geometric mesh partitioner</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>sphere_na.vtk</filename>
      <partitioner>ArcaneGeometricMeshPartitioner</partitioner>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest">
      <write-mesh-service-name>VtkLegacyMeshWriter</write-mesh-service-name>
    </test>
  </unit-test-module>

</case>
