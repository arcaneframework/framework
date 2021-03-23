<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Tube à choc de Sod</title>
  <timeloop>ArcaneHydroLoop</timeloop>
 </arcane>

 <mesh>

  <!-- <file internal-partition="true">sod.vtk</file> -->
  <!-- <meshgenerator><sod><x>1200</x><y>25</y><z>50</z></sod></meshgenerator> --> <!-- MAX AVEC 3Go -->
  <!-- <meshgenerator><sod><x>800</x><y>25</y><z>50</z></sod></meshgenerator> --> <!-- STANDARD GROS MAILLES -->
  <!-- <meshgenerator><sod><x>500</x><y>15</y><z total="true">800</z></sod></meshgenerator> -->
  <meshgenerator><sod><x>100</x><y>15</y><z>15</z></sod></meshgenerator>
  <!-- <meshgenerator><sod><x>1000</x><y>20</y><z>20</z></sod></meshgenerator> -->
  <!-- <meshgenerator><sod><x>50</x><y>2</y><z>2</z></sod></meshgenerator> -->

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
   <output-period>0</output-period>
   <output-history-period>2</output-history-period>
   <!-- <format name="EnsightHdfPostProcessor" /> -->
   <output>
    <variable>CellMass</variable>
    <variable>CellVolume</variable>
    <variable>Pressure</variable>
    <variable>Density</variable>
    <variable>Velocity</variable>
    <variable>NodeMass</variable>
    <variable>InternalEnergy</variable>
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
   <!-- <ensight7gold>
    <binary-file>true</binary-file>
   </ensight7gold>-->
 </arcane-post-processing>
 <arcane-checkpoint>
  <!--<period>1</period>  -->
  <!-- <checkpoint-service name="ArcaneHdf5MultiCheckpointWriter" /> -->
  <do-dump-at-end>false</do-dump-at-end>
 </arcane-checkpoint>

 <!-- Configuration du module hydrodynamique -->
 <simple-hydro>

   <!-- <deltat-init>   0.0000001   </deltat-init>
   <deltat-min>    0.00000001   </deltat-min>
   <deltat-max>    0.000001   </deltat-max> -->
   <deltat-init>   0.001   </deltat-init>
   <deltat-min>    0.0001   </deltat-min>
   <deltat-max>    0.01   </deltat-max>
   <final-time>     0.2    </final-time>

  <viscosity>cell</viscosity>
  <viscosity-linear-coef>    .5    </viscosity-linear-coef>
  <viscosity-quadratic-coef> .6    </viscosity-quadratic-coef>

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
