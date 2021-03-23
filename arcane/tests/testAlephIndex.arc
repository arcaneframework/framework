<?xml version="1.0" encoding="ISO-8859-1"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Test ALEPH+index</title>
  <description>Test de resolutions asynchrones depuis l'indexeur</description>
  <timeloop>AlephIndexTestLoop</timeloop>
  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

 <arcane-post-processing>
   <save-init>0</save-init>
   <end-execution-output>1</end-execution-output>
   <output-period>0</output-period>
   <output>
     <variable>CellTemperature</variable>
   </output>
 </arcane-post-processing>
 
 <mesh>
   <meshgenerator>
     <sod>
       <x set='true' delta='0.02'>4</x><!-- 16 -->
       <y set='true' delta='0.02'>4</y><!-- 16 -->
       <!--z set='true' delta='0.02' total='true'>32</z--><!-- 48 -->
     </sod>
   </meshgenerator>
 </mesh>

 <aleph-index-test>
   <iterations>1</iterations>

   <aleph-number-of-solvers>4</aleph-number-of-solvers>

   <deltaT>0.001</deltaT>
   <init-temperature>300</init-temperature>

	<boundary-condition> <surface>YMIN</surface> <value>700</value> </boundary-condition>
	<boundary-condition> <surface>YMAX</surface> <value>300</value> </boundary-condition>
	<boundary-condition> <surface>XMIN</surface> <value>700</value> </boundary-condition>
	<boundary-condition> <surface>XMAX</surface> <value>300</value> </boundary-condition>
	<!--boundary-condition> <surface>ZMIN</surface> <value>700</value> </boundary-condition>
   <boundary-condition> <surface>ZMAX</surface> <value>300</value> </boundary-condition-->
 </aleph-index-test>
</case>
