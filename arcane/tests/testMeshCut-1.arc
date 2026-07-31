<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test MeshCut 1</title>
    <description>Test MeshCut 1</description>
    <timeloop>MeshCutTestLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian3D">
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <nb-part-z>1</nb-part-z>
        <origin>0.0 0.0 0.0</origin>
        <x><n>2</n><length>2.0</length></x>
        <y><n>2</n><length>2.0</length></y>
        <z><n>2</n><length>2.0</length></z>
      </generator>
    </mesh>
  </meshes>

  <mesh-cut-test>
    <enable-post-processing>false</enable-post-processing>
    <plane>
      <p0>1 1 1</p0>
      <normal>1 0 0</normal>
    </plane>
    <plane>
      <p0>1 1 1</p0>
      <normal>1 0.5 0</normal>
    </plane>
    <plane>
      <p0>1.6 0 0</p0>
      <normal>1 0.5 0.1</normal>
    </plane>
  </mesh-cut-test>
</case>
