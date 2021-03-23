<?xml version="1.0" encoding="ISO-8859-1"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Test ALEPH+multi</title>
  <description>Test de multi resolutions asynchrones avec l'indexeur </description>
  <timeloop>AlephMultiTestLoop</timeloop>
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
       <x set='true' delta='0.02'>8</x>
       <y set='true' delta='0.02'>8</y>
       <z set='true' delta='0.02' total='false'>8</z>
     </sod>
   </meshgenerator>
 </mesh>

 <aleph-multi-test>
   <deltaT>0.01</deltaT>
   <iterations>4</iterations>
   <ini-temperature>300</ini-temperature>
   <hot-temperature>700</hot-temperature>
   <aleph-number-of-successive-solvers>8</aleph-number-of-successive-solvers>
   <aleph-number-of-resolutions-per-solvers>0x19548732</aleph-number-of-resolutions-per-solvers>
   <aleph-underlying-solver>0x01201212</aleph-underlying-solver>
   <aleph-number-of-cores>0x21021010</aleph-number-of-cores>
 </aleph-multi-test>
</case>
