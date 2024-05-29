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
      <filename>cross_a_2x1x1.vtk</filename>
      <specific-reader name="VtkPolyhedralCaseMeshReader">
        <print-mesh-infos>true</print-mesh-infos>
        <print-debug-infos>false</print-debug-infos>
      </specific-reader>
    </mesh>
  </meshes>

  <custom-mesh-test>
    <mesh-size>
      <nb-cells>2</nb-cells>
      <nb-faces>15</nb-faces>
      <nb-edges>29</nb-edges>
      <nb-nodes>16</nb-nodes>
    </mesh-size>
    <nb-mesh-group>5</nb-mesh-group>
    <check-group>
      <name>HALF_CELL</name>
      <size>1</size>
    </check-group>
    <check-group>
      <name>FIRST_CELL_NODES</name>
      <size>10</size>
    </check-group>
    <check-group>
      <name>HALF_FACE</name>
      <size>7</size>
    </check-group>
    <check-group>
      <name>BOUNDARY_FACES</name>
      <size>14</size>
    </check-group>
    <check-boundary-face-group>BOUNDARY_FACES</check-boundary-face-group>
    <check-internal-face-group>INTERNAL_FACES</check-internal-face-group>
    <check-cell-variable-integer>CellFlags</check-cell-variable-integer>
    <check-cell-variable-real>CellReal</check-cell-variable-real>
    <check-cell-variable-array-integer>CellArrayFlags</check-cell-variable-array-integer>
    <check-cell-variable-array-real>CellArrayReal</check-cell-variable-array-real>
    <check-node-variable-integer>NodeFlags</check-node-variable-integer>
    <check-node-variable-real>NodeReal</check-node-variable-real>
    <check-node-variable-array-integer>NodeArrayFlags</check-node-variable-array-integer>
    <check-node-variable-array-real>NodeArrayReal</check-node-variable-array-real>
    <check-face-variable-integer>FaceFlags</check-face-variable-integer>
    <check-face-variable-real>FaceReal</check-face-variable-real>
    <check-face-variable-array-integer>FaceArrayFlags</check-face-variable-array-integer>
    <check-face-variable-array-real>FaceArrayReal</check-face-variable-array-real>
    <mesh-coordinates>
      <do-check>true</do-check>
      <coords>
        <value>0 0 50</value>
        <value>0 0 150</value>
        <value>100 0 -50</value>
        <value>100 0 50</value>
        <value>100 0 150</value>
        <value>0 100 -50</value>
        <value>0 100 50</value>
        <value>100 100 -50</value>
        <value>100 100 50</value>
        <value>100 100 150</value>
        <value>100 50 0</value>
        <value>100 50 100</value>
        <value>200 0 -50</value>
        <value>200 0 50</value>
        <value>200 100 50</value>
        <value>200 100 150</value>
      </coords>
    </mesh-coordinates>
  </custom-mesh-test>

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