<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <titre>Test Mesh HoneyComb 2D</titre>
  <description>Test Mesh HoneyComb 2D</description>
  <timeloop>UnitTest</timeloop>
 </arcane>

 <meshes>
   <mesh>
     <generator name="HoneyComb2D">
       <origin>0.0 0.0</origin>
       <pitch-size>2.0</pitch-size>
       <nb-layer>10</nb-layer>
     </generator>
   </mesh>
 </meshes>

 <unit-test-module>
   <test name="MeshUnitTest">
     <test-adjency>0</test-adjency>
   </test>
 </unit-test-module>

</case>
