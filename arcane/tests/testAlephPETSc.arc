<?xml version="1.0" encoding="ISO-8859-1"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Test ALEPH</title>
  <description>Test de resolutions asynchrones</description>
  <timeloop>AlephTestLoop</timeloop>
  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

 <arcane-post-processing>
   <save-init>1</save-init>
   <end-execution-output>1</end-execution-output>
   <output-period>1</output-period>
   <output>
     <variable>CellTemperature</variable>
     <variable>UniqueId</variable>
     <variable>SubDomainId</variable>
   </output>
 </arcane-post-processing>
  
 <mesh>
  <meshgenerator>
    <sod>
      <x set='true' delta='0.02'>8</x><!-- 16 -->
      <y set='true' delta='0.02'>8</y><!-- 16 -->
      <z set='true' delta='0.02' total='true'>32</z><!-- 48 -->
  </sod>
  </meshgenerator>
 </mesh>

 <aleph-test-module>
   <schema name="Faces"/>

   <iterations>2</iterations>

   <aleph-number-of-solvers>8</aleph-number-of-solvers>
   <aleph-number-of-cores>1</aleph-number-of-cores>
   <aleph-cell-ordering>false</aleph-cell-ordering>
   <aleph-underlying-solver>5</aleph-underlying-solver>

   <deltaT>0.1</deltaT>
   <init-temperature>300</init-temperature>
   <init-amr>0.0</init-amr>
   <trig-refine>0.01</trig-refine>
   <trig-coarse>0.0004</trig-coarse>
	<boundary-condition> <surface>YMIN</surface> <value>300</value> </boundary-condition>
	<boundary-condition> <surface>YMAX</surface> <value>700</value> </boundary-condition>
	<boundary-condition> <surface>XMIN</surface> <value>300</value> </boundary-condition>
	<boundary-condition> <surface>XMAX</surface> <value>700</value> </boundary-condition>
	<boundary-condition> <surface>ZMIN</surface> <value>300</value> </boundary-condition>
   <boundary-condition> <surface>ZMAX</surface> <value>700</value> </boundary-condition>
 </aleph-test-module>
</case>
