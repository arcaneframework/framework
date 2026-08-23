<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Vtk Mesh Polyhedral 4</title>
    <description>Test polyhedral mesh cross 2x1x1 vtk xml</description>
    <timeloop>MeshPolyhedralTestLoop</timeloop>
    <modules>
      <module name="ArcanePostProcessing" active="true"/>
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <filename>faultx4_3x3x2.vtk</filename>
      <specific-reader name="VtkPolyhedralCaseMeshReader">
        <print-mesh-infos>true</print-mesh-infos>
        <print-debug-infos>false</print-debug-infos>
      </specific-reader>
    </mesh>
  </meshes>

  <mesh-polyhedral-test>
    <mesh-coordinates>
      <do-check>false</do-check>
    </mesh-coordinates>
  </mesh-polyhedral-test>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>CellVariable</variable>
      <!--      <variable>FaceVariable</variable>-->
      <variable>NodeVariable</variable>
      <group>AllCells</group>
      <!--      <group>AllFaces</group>-->
    </output>
    <save-init>true</save-init>
    <format name="Ensight7PostProcessor">
      <binary-file>false</binary-file>
    </format>
  </arcane-post-processing>

</case>