<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
   <title>Test MaterialHeat</title>
   <description>Test des Materiaux</description>
   <timeloop>MaterialHeatTestLoop</timeloop>
 </arcane>

 <arcane-post-processing>
   <output-period>5</output-period>
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
       <x><n>20</n><length>1.2</length><progression>1.0</progression></x>
       <y><n>30</n><length>1.5</length><progression>1.0</progression></y>
     </generator>
   </mesh>
 </meshes>

 <material-heat-test>
   <nb-iteration>15</nb-iteration>
   <modification-flags>7</modification-flags>
   <check-numerical-result>true</check-numerical-result>
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
   </environment>
   <environment>
     <name>ENV2</name>
     <material>MAT1</material>
   </environment>
   <environment>
     <name>ENV3</name>
     <material>MAT2</material>
   </environment>
   <environment>
     <name>ENV4</name>
     <material>MAT2</material>
     <material>MAT3</material>
   </environment>

   <heat-object>
     <center>0.3 0.4 0.0</center>
     <velocity>0.02 0.04 0.0</velocity>
     <radius>0.18</radius>
     <material>ENV1_MAT1</material>
     <expected-final-temperature>228578.906835675</expected-final-temperature>
   </heat-object>
   <heat-object>
     <center>0.8 0.4 0.0</center>
     <velocity>-0.02 0.04 0.0</velocity>
     <radius>0.25</radius>
     <material>ENV2_MAT1</material>
     <expected-final-temperature>478199.099739938</expected-final-temperature>
   </heat-object>
   <heat-object>
     <center>0.2 1.2 0.0</center>
     <velocity>0.02 -0.05 0.0</velocity>
     <radius>0.2</radius>
     <material>ENV3_MAT2</material>
     <expected-final-temperature>274659.975324849</expected-final-temperature>
   </heat-object>
   <heat-object>
     <center>0.9 0.9 0.0</center>
     <velocity>-0.02 -0.04 0.0</velocity>
     <radius>0.15</radius>
     <material>ENV4_MAT2</material>
     <expected-final-temperature>134910.252007983</expected-final-temperature>
   </heat-object>
   <heat-object>
     <center>0.4 0.3 0.0</center>
     <velocity>0.02 0.04 0.0</velocity>
     <radius>0.1</radius>
     <material>ENV4_MAT3</material>
     <expected-final-temperature>47649.1358182313</expected-final-temperature>
   </heat-object>

 </material-heat-test>

</case>
