<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test non manifold mesh</title>
    <description>Test non manifold mesh</description>
    <timeloop>UnitTest</timeloop>
  </arcane>
  <meshes>
    <mesh>
      <filename>mesh_with_loose_items.msh</filename>
     <cell-dimension-kind>non-manifold</cell-dimension-kind>
      <face-numbering-version>0</face-numbering-version>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest">
      <write-mesh-service-name>MshMeshWriter</write-mesh-service-name>
      <test-adjency>false</test-adjency>
    </test>
  </unit-test-module>
</case>
