<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <titre>Test IO Reader/Writer for SimpleMeshGenerator</titre>
    <description>Test IO Reader/Writer for SimpleMeshGenerator</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

  <mesh>
    <meshgenerator>
      <simple>
        <mode>4</mode>
      </simple>
    </meshgenerator>
  </mesh>
 
  <unit-test-module>
    <test name="IosUnitTest">
      <write-vtu>false</write-vtu>
      <write-xmf>false</write-xmf>
      <write-msh>false</write-msh>
      <write-vtk-legacy>true</write-vtk-legacy>
    </test>
  </unit-test-module>
</case>
