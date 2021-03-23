<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Sod Choc Tube Hyoda</title>
  <timeloop>ArcaneHydroLoop</timeloop>
 </arcane>

 <mesh>
   <!-- 32768 coeurs + milliard de mailles, export ARCANE_PARALLEL_ARGS="-p batch -M 2000" -->
   <!--meshgenerator><sod><x>64</x><y>32</y><z>16</z></sod></meshgenerator-->
 
   <!-- 2048 coeurs => milliard de mailles OK -->
   <!--meshgenerator><sod><x>256</x><y>128</y><z>16</z></sod></meshgenerator-->
   
   <!-- 8192 mailles par défaut, pour les tests de non régression -->

   <!--meshgenerator><sod><x>256</x><y>128</y><z>16</z></sod></meshgenerator-->
   <!-- Attention au temps d'exécution -->
   <meshgenerator><sod><x>64</x><y>32</y><z>8</z></sod></meshgenerator>

 <initialisation>
  <variable nom="Density" valeur="1.0" groupe="ZD" />
  <variable nom="Pressure" valeur="8.0" groupe="ZD" />
  <variable nom="AdiabaticCst" valeur="1.4" groupe="ZD" />

  <variable nom="Density" valeur="0.1" groupe="ZG" />
  <variable nom="Pressure" valeur="0.1" groupe="ZG" />
  <variable nom="AdiabaticCst" valeur="1.4" groupe="ZG" />
 </initialisation>
 </mesh>

 <arcane-post-processing>
   <output-period>0</output-period>
   <output>
    <variable>Pressure</variable>
    <variable>Density</variable>
    <variable>Velocity</variable>
    <variable>InternalEnergy</variable>
    <variable>SubDomainId</variable>
    <variable>UniqueId</variable>
    <group>ZG</group>
    <group>ZD</group>
    <!--group>AllFaces</group>
    <group>XMIN</group>
    <group>XMAX</group>
    <group>YMIN</group>
    <group>YMAX</group>
    <group>ZMIN</group>
    <group>ZMAX</group-->
   </output>
   <output-history-period>0</output-history-period>
 </arcane-post-processing>


 <arcane-checkpoint>
  <do-dump-at-end>false</do-dump-at-end>
 </arcane-checkpoint>

 <!-- Configuration du module hydrodynamique -->
 <simple-hydro>
   <deltat-init>0.00001</deltat-init>
   <deltat-min>0.00001</deltat-min>
   <deltat-max>0.0001</deltat-max>
   <final-time>0.001</final-time>
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
