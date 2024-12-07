<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
   <title>Test reading GMSH mesh with periodic information</title>
   <description>Test Maillage VTK 4.2 Binary sphere</description>
   <timeloop>UnitTest</timeloop>
 </arcane>

  <meshes>
    <mesh>
      <filename>onesphere.msh</filename>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest" />
  </unit-test-module>
</case>
