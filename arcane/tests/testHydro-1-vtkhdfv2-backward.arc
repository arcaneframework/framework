<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Tube a choc de Sod</title>
  <timeloop>ArcaneHydroLoop</timeloop>
 </arcane>

 <mesh>

   <meshgenerator><sod><x>50</x><y>3</y><z>3</z></sod></meshgenerator>

   <initialisation>
     <variable nom="Density" valeur="1." groupe="ZG" />
     <variable nom="Pressure" valeur="1." groupe="ZG" />
     <variable nom="AdiabaticCst" valeur="1.4" groupe="ZG" />
     <variable nom="Density" valeur="0.125" groupe="ZD" />
     <variable nom="Pressure" valeur="0.1" groupe="ZD" />
     <variable nom="AdiabaticCst" valeur="1.4" groupe="ZD" />
   </initialisation>
 </mesh>

 <arcane-post-processing>
   <output-period>2</output-period>
   <format name="VtkHdfV2PostProcessor" >
     <max-write-size>15000</max-write-size>
   </format>
   <output>
    <variable>CellMass</variable>
    <variable>CellVolume</variable>
    <variable>Pressure</variable>
    <variable>Density</variable>
    <variable>Velocity</variable>
    <variable>NodeMass</variable>
    <variable>InternalEnergy</variable>
    <variable>SubDomainId</variable>
    <group>ZG</group>
    <group>ZD</group>
    <group>AllFaces</group>
    <group>XMIN</group>
    <group>XMAX</group>
    <group>YMIN</group>
    <group>YMAX</group>
    <group>ZMIN</group>
    <group>ZMAX</group>
   </output>
 </arcane-post-processing>
 <arcane-checkpoint>
  <do-dump-at-end>false</do-dump-at-end>
 </arcane-checkpoint>

 <!-- Configuration du module hydrodynamique -->
 <simple-hydro>

   <deltat-init>0.001</deltat-init>
   <deltat-min>0.0001</deltat-min>
   <deltat-max>0.01</deltat-max>
   <final-time>0.2</final-time>

  <backward-iteration>17</backward-iteration>

  <viscosity>cell</viscosity>
  <viscosity-linear-coef>0.5</viscosity-linear-coef>
  <viscosity-quadratic-coef>0.6</viscosity-quadratic-coef>

  <boundary-condition>
    <surface>XMIN</surface><type>Vx</type><value>0.</value>
  </boundary-condition>
  <boundary-condition>
    <surface>XMAX</surface><type>Vx</type><value>0.</value>
  </boundary-condition>
  <boundary-condition>
    <surface>YMIN</surface><type>Vy</type><value>0.</value>
  </boundary-condition>
  <boundary-condition>
    <surface>YMAX</surface><type>Vy</type><value>0.</value>
  </boundary-condition>
  <boundary-condition>
    <surface>ZMIN</surface><type>Vz</type><value>0.</value>
  </boundary-condition>
  <boundary-condition>
    <surface>ZMAX</surface><type>Vz</type><value>0.</value>
  </boundary-condition>
 </simple-hydro>
</case>
