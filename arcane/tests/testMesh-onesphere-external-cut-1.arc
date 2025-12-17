<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
   <title>Test reading GMSH mesh with periodic information</title>
   <description>Test reading GMSH mesh with periodic information</description>
   <timeloop>UnitTest</timeloop>
 </arcane>

  <meshes>
    <mesh>
      <filename>onesphere.msh</filename>
      <partitioner>External</partitioner>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest">
      <test-adjency>false</test-adjency>
      <test-variable-writer>false</test-variable-writer>
      <write-mesh>false</write-mesh>
    </test>
  </unit-test-module>
</case>
