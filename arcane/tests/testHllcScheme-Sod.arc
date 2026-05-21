<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>HLLC Scheme - Tube à choc de Sod</title>
  <timeloop>HllcSchemeLoop</timeloop>
 </arcane>

 <mesh>
  <meshgenerator><sod><x>100</x><y>5</y><z>5</z></sod></meshgenerator>
 </mesh>

 <arcane-post-processing>
   <output-period>10</output-period>
   <output>
    <variable>Density</variable>
    <variable>Pressure</variable>
    <variable>Momentum</variable>
    <variable>SoundSpeed</variable>
    <variable>CellVolume</variable>
   </output>
 </arcane-post-processing>
 <arcane-checkpoint>
  <do-dump-at-end>false</do-dump-at-end>
 </arcane-checkpoint>

 <hllc-scheme>
  <cfl>0.5</cfl>
  <final-time>0.2</final-time>
  <deltat-init>1.0e-4</deltat-init>
  <gamma>1.4</gamma>
  <spatial-order>2</spatial-order>
  <limiter>VanLeer</limiter>

  <boundary-condition>
   <type>Outflow</type>
   <surface>XMIN</surface>
  </boundary-condition>
  <boundary-condition>
   <type>Outflow</type>
   <surface>XMAX</surface>
  </boundary-condition>
  <boundary-condition>
   <type>Outflow</type>
   <surface>YMIN</surface>
  </boundary-condition>
  <boundary-condition>
   <type>Outflow</type>
   <surface>YMAX</surface>
  </boundary-condition>
  <boundary-condition>
   <type>Outflow</type>
   <surface>ZMIN</surface>
  </boundary-condition>
  <boundary-condition>
   <type>Outflow</type>
   <surface>ZMAX</surface>
  </boundary-condition>
 </hllc-scheme>
</case>
