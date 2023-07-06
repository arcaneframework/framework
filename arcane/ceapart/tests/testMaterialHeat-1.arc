<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
   <title>Test MaterialHeat</title>
   <description>Test des ateriaux</description>
   <timeloop>MaterialHeatTestLoop</timeloop>
 </arcane>

 <arcane-post-processing>
   <output-period>1</output-period>
   <output>
    <variable>Temperature</variable>
   </output>
 </arcane-post-processing>

 <meshes>
   <mesh>
     <generator name="Cartesian2D">
       <nb-part-x>2</nb-part-x>
       <nb-part-y>2</nb-part-y>
       <origin>0.0 0.0</origin>
       <x><n>10</n><length>1.0</length><progression>1.0</progression></x>
       <y><n>20</n><length>2.0</length><progression>1.0</progression></y>
     </generator>
   </mesh>
 </meshes>

 <material-heat-test>
  <material>
   <name>MAT1</name>
  </material>
  <material>
   <name>MAT2</name>
  </material>

  <environment>
   <name>ENV1</name>
   <material>MAT1</material>
   <material>MAT2</material>
  </environment>

 </material-heat-test>

</case>
