<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Test mesh with misc types of Order2 with non-manifold support</title>
  <description>Test mesh with misc types of Order2 with non-manifold support</description>
  <timeloop>UnitTest</timeloop>
 </arcane>

 <meshes>
   <mesh>
     <filename>order2_misc_types.msh</filename>
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
