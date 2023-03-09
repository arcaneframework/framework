<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Maillage Custom 1</title>
    <description>Test Maillage 1</description>
    <timeloop>CustomMeshTestLoop</timeloop>
    <modules>
      <module name="ArcanePostProcessing" active="true"/>
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <!--     <fichier>example_polyhedral_cell.xmf</fichier>-->
      <filename>faultx1_2x1x1.vtk</filename>
      <specific-reader name="VtkPolyhedralCaseMeshReader"/>
    </mesh>
  </meshes>

  <custom-mesh-test>
    <mesh-size>
      <nb-cells>2</nb-cells>
      <nb-faces>13</nb-faces>
      <nb-edges>26</nb-edges>
      <nb-nodes>16</nb-nodes>
    </mesh-size>
    <mesh-coordinates>
      <do-check>true</do-check>
      <coords>
        <value>0 0 25</value>
        <value>0 0 -75</value>
        <value>100 0 25</value>
        <value>100 0 -25</value>
        <value>100 0 -75</value>
        <value>100 0 -125</value>
        <value>0 100 25</value>
        <value>0 100 -75</value>
        <value>100 100 25</value>
        <value>100 100 -25</value>
        <value>100 100 -75</value>
        <value>100 100 -125</value>
        <value>200 0 -25</value>
        <value>200 0 -125</value>
        <value>200 100 -25</value>
        <value>200 100 -125</value>
      </coords>
    </mesh-coordinates>
  </custom-mesh-test>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>CellVariable</variable>
      <variable>FaceVariable</variable>
      <variable>NodeVariable</variable>
      <group>AllCells</group>
      <group>AllFaces</group>
    </output>
    <save-init>true</save-init>
    <format name="Ensight7PostProcessor">
      <binary-file>false</binary-file>
    </format>
  </arcane-post-processing>

</case>