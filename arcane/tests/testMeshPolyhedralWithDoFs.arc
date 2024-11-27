<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Vtk Mesh Polyhedral 1</title>
    <description>Test polyhedral mesh fault 2x1x1</description>
    <timeloop>UnitTest</timeloop>
    <modules>
      <module name="ArcanePostProcessing" active="true"/>
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <filename>faultx1_2x1x1.vtk</filename>
      <specific-reader name="VtkPolyhedralCaseMeshReader">
        <print-mesh-infos>true</print-mesh-infos>
        <print-debug-infos>false</print-debug-infos>
      </specific-reader>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="DoFTester">
        <do-check-compacting>false</do-check-compacting>
    </test>
  </unit-test-module>

</case>