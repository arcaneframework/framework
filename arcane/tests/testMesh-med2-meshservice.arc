<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0" xml:lang="en">
  <arcane>
    <title>Test Maillage MED2</title>
    <description>Test Maillage MED2 (avec elements quadratiques)</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>mesh2.med</filename>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshUnitTest">
      <test-adjency>0</test-adjency>
    </test>
  </unit-test-module>

</case>
