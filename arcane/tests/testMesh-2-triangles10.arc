<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test mesh with Triangles10</title>
    <description>Test mesh with Triangles10</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>triangles10_v41.msh</filename>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest">
      <write-mesh-service-name>MshMeshWriter</write-mesh-service-name>
      <check-local-ids-from-connectivity>true</check-local-ids-from-connectivity>
      <test-adjency>false</test-adjency>
    </test>
  </unit-test-module>

</case>
