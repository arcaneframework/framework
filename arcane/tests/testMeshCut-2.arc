<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test MeshCut 1</title>
    <description>Test MeshCut 1</description>
    <timeloop>MeshCutTestLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>sod3d-misc.msh</filename>
    </mesh>
  </meshes>

  <mesh-cut-test>
    <enable-post-processing>false</enable-post-processing>
    <plane>
      <p0>0 0.05 0</p0>
      <normal>0 1 0</normal>
    </plane>
    <plane>
      <p0>0.98 0 0</p0>
      <p0-velocity>0 -0.1 0</p0-velocity>
      <normal>1 0.5 0.1</normal>
    </plane>
  </mesh-cut-test>
</case>
