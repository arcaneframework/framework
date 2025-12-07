<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <titre>Test basic geometric mesh partitioner</titre>
    <description>Test basic geometric mesh partitioner</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>planar_unstructured_quad1.msh</filename>
      <partitioner>ArcaneGeometricMeshPartitioner</partitioner>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest">
      <write-mesh-service-name>VtkLegacyMeshWriter</write-mesh-service-name>
      <check-local-ids-from-connectivity>true</check-local-ids-from-connectivity>
      <test-adjency>false</test-adjency>
    </test>
  </unit-test-module>

</case>
