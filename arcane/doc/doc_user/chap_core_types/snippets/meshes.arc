<!--
  Documentation examples for the specification of meshes in the data set
  (.arc file). These examples are included in the documentation using the
  Doxygen \snippet command (see the 'ArcaneCaseMeshMasterService' and
  'ArcaneCaseMeshService' services and the chapter about the data set).
  This file is only used for the documentation: it is not a valid data set.
-->

<!-- //![snippet_meshes_multi_mesh] -->
<meshes>
  <mesh>
    <filename>sod.vtk</filename>
  </mesh>
  <mesh>
    <generator name="Cartesian3D"/>
  </mesh>
</meshes>
<!-- //![snippet_meshes_multi_mesh] -->

<!-- //![snippet_meshes_generator_initialization] -->
<meshes>
  <mesh>
    <generator name="Cartesian3D">
      <x><n>100</n><length>2.0</length></x>
      <y><n>15</n><length>2.0</length></y>
      <z><n>10</n><length>4.0</length></z>
    </generator>
    <initialization>
      <variable>
        <name>Density</name>
        <value>1.0</value>
        <group>ZG</group>
      </variable>
    </initialization>
  </mesh>
</meshes>
<!-- //![snippet_meshes_generator_initialization] -->

<!-- //![snippet_meshes_cartesian2d_generator] -->
<generator name="Cartesian2D">
  <origin>0.0 0.0</origin>
  <x><n>20</n><length>1.0</length></x>
  <y><n>10</n><length>1.0</length></y>
</generator>
<!-- //![snippet_meshes_cartesian2d_generator] -->

<!-- //![snippet_meshes_specific_reader] -->
<meshes>
  <mesh>
    <filename>mesh.vtk</filename>
    <specific-reader name="VtkPolyhedralCaseMeshReader">
      <print-mesh-infos>true</print-mesh-infos>
    </specific-reader>
  </mesh>
</meshes>
<!-- //![snippet_meshes_specific_reader] -->

<!-- //![snippet_meshes_initialization_variables] -->
<initialization>
  <variable>
    <name>Density</name>
    <value>1.0</value>
    <group>ZG</group>
  </variable>
  <variable>
    <name>Density</name>
    <value>0.125</value>
    <group>ZD</group>
  </variable>
</initialization>
<!-- //![snippet_meshes_initialization_variables] -->
