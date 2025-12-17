<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test Maillage 1</title>
    <description>Test Maillage 1</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>onesphere.msh</filename>
      <face-numbering-version>0</face-numbering-version>
      <edge-numbering-version>0</edge-numbering-version>
    </mesh>
  </meshes>

  <unit-test-module>
    <test name="MeshReaderUnitTest">
      <create-edges>true</create-edges>
      <generate-uid-from-nodes-uid>true</generate-uid-from-nodes-uid>
    </test>
  </unit-test-module>

</case>
