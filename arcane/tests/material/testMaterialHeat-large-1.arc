<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
   <title>Test MaterialHeat</title>
   <description>Test des ateriaux</description>
   <timeloop>MaterialHeatTestLoop</timeloop>
 </arcane>

 <arcane-post-processing>
   <output-period>0</output-period>
   <output>
    <variable>Temperature</variable>
    <variable>AllTemperatures</variable>
   </output>
 </arcane-post-processing>

 <meshes>
   <mesh>
     <generator name="Cartesian2D">
       <nb-part-x>2</nb-part-x>
       <nb-part-y>2</nb-part-y>
       <origin>0.0 0.0</origin>
       <x><n>400</n><length>1.2</length><progression>1.0</progression></x>
       <y><n>600</n><length>1.5</length><progression>1.0</progression></y>
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

   <heat-object>
     <center>0.3 0.4 0.0</center>
     <velocity>0.02 0.04 0.0</velocity>
     <radius>0.18</radius>
     <material>ENV1_MAT1</material>
   </heat-object>
   <heat-object>
     <center>0.8 0.4 0.0</center>
     <velocity>-0.02 0.04 0.0</velocity>
     <radius>0.25</radius>
     <material>ENV1_MAT2</material>
   </heat-object>

 </material-heat-test>

</case>
