<?xml version="1.0" encoding="ISO-8859-1"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Hyoda Mixed Cells Test</title>
  <timeloop>BasicLoop</timeloop>
  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

 <mesh>
   <meshgenerator><sod>
       <x set='true' delta='1.0'>6</x>
       <y set='true' delta='1.0'>3</y>
   </sod></meshgenerator>
 </mesh>
 
 <master-module>
   <global-service name="HyodaMixedCellsUnitTest">
     <iterations>512</iterations> 
     <material><name>matLight</name></material>
     <material><name>matWater</name></material>
     <material><name>matHeavy</name></material>
     <environment><name>envLeft</name>
       <material>matLight</material>
       <material>matHeavy</material>
     </environment>
     <environment><name>envMiddle</name>
       <material>matWater</material>
     </environment>
     <environment><name>envRight</name>
       <material>matLight</material>
       <material>matWater</material>
       <material>matHeavy</material>
     </environment>
   </global-service>
 </master-module>
 
 <arcane-post-processing>
   <save-init>0</save-init>
	<output-period>0</output-period>
   <output-history-period>0</output-history-period>
   <end-execution-output>0</end-execution-output>
   <output>
     <variable>density</variable>
     <variable>qedge</variable>
   </output>
 </arcane-post-processing>
 
</case>
