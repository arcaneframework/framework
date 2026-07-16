<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test MeshCut 1</title>
    <description>Test MeshCut 1</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>sod3d-misc.msh</filename>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshSectionTest">
      <plane>
        <p0>0.4 0 0</p0>
        <normal>1 0 0</normal>
      </plane>
      <plane>
        <p0>0.6 0 0</p0>
        <normal>-1 0 0</normal>
      </plane>
    </test>

  </unit-test-module>
</case>
