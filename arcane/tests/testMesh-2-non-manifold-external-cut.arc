<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Test mesh with loose items and external cut</title>
  <description>Test mesh with loose items and external cut</description>
  <timeloop>UnitTest</timeloop>
 </arcane>

 <meshes>
   <mesh>
     <filename>mesh_with_loose_items.msh</filename>
     <cell-dimension-kind>non-manifold</cell-dimension-kind>
     <partitioner>External</partitioner>
   </mesh>
 </meshes>

 <unit-test-module>
  <test name="MeshUnitTest">
    <write-mesh-service-name>VtkLegacyMeshWriter</write-mesh-service-name>
    <test-adjency>false</test-adjency>
  </test>
 </unit-test-module>

</case>
