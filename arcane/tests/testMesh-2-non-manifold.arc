<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Test mesh with loose items</title>
  <description>Test mesh with loose items</description>
  <timeloop>UnitTest</timeloop>
 </arcane>

 <meshes>
   <mesh>
     <filename>mesh_with_loose_items.msh</filename>
     <cell-dimension-kind>non-manifold</cell-dimension-kind>
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
