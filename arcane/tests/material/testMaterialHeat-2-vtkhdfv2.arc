<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
   <title>Test MaterialHeat</title>
   <description>Test des Materiaux</description>
   <timeloop>MaterialHeatTestLoop</timeloop>
 </arcane>

 <arcane-post-processing>
   <output-period>5</output-period>
   <format name="VtkHdfV2PostProcessor" />
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
       <x><n>40</n><length>1.2</length><progression>1.0</progression></x>
       <y><n>60</n><length>1.5</length><progression>1.0</progression></y>
     </generator>
   </mesh>
 </meshes>

 <material-heat-test>
   <nb-iteration>15</nb-iteration>
   <modification-flags>15</modification-flags>
   <check-numerical-result>false</check-numerical-result>
   <material>
     <name>MAT1</name>
   </material>
   <material>
     <name>MAT2</name>
   </material>
   <material>
     <name>MAT3</name>
   </material>

   <environment>
     <name>ENV1</name>
     <material>MAT1</material>
     <material>MAT2</material>
   </environment>
   <environment>
     <name>ENV2</name>
     <material>MAT2</material>
     <material>MAT3</material>
   </environment>

   <heat-object>
     <center>0.3 0.4 0.0</center>
     <velocity>0.02 0.04 0.0</velocity>
     <radius>0.18</radius>
     <material>ENV1_MAT1</material>
     <expected-final-temperature>3632937.10322508</expected-final-temperature>
   </heat-object>
   <heat-object>
     <center>0.8 0.4 0.0</center>
     <velocity>-0.02 0.04 0.0</velocity>
     <radius>0.25</radius>
     <material>ENV1_MAT2</material>
     <expected-final-temperature>7780818.83419631</expected-final-temperature>
   </heat-object>
   <heat-object>
     <center>0.2 1.2 0.0</center>
     <velocity>0.02 -0.05 0.0</velocity>
     <radius>0.2</radius>
     <material>ENV2_MAT2</material>
     <expected-final-temperature>4230364.18968662</expected-final-temperature>
   </heat-object>
   <heat-object>
     <center>0.9 0.9 0.0</center>
     <velocity>-0.02 -0.04 0.0</velocity>
     <radius>0.15</radius>
     <material>ENV2_MAT3</material>
     <expected-final-temperature>2259280.64283209</expected-final-temperature>
   </heat-object>

 </material-heat-test>

</case>
