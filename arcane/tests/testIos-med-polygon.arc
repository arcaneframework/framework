<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <titre>Test IO Reader/Writer for MED and VTK with polygons</titre>
    <description>Test IO Reader/Writer for MED and VTK with polygons</description>
    <timeloop>UnitTest</timeloop>
  </arcane>

 <mesh>
   <file internal-partition="true">circle_cut-poly.med</file>
   
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
