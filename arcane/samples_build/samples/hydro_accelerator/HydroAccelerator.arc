<?xml version="1.0"?>
<case codename="HydroAccelerator" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Tube a choc de Sod avec accelerateur</title>
  <timeloop>ArcaneHydroGenericLoop</timeloop>
 </arcane>

 <mesh>
  <!-- La validite numerique suppose que le maillage est 100x15x15.
       Si ce n'est pas le cas, il faut desactiver l'option 'check-numerical-result' -->

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

 <!-- Configuration du module hydrodynamique -->
 <simple-hydro>
   <check-numerical-result>true</check-numerical-result>
   <generic-service-name>SimpleHydroAcceleratorService</generic-service-name>
   <deltat-init>0.001</deltat-init>
   <deltat-min>0.0001</deltat-min>
   <deltat-max>0.01</deltat-max>
   <final-time>0.2</final-time>

  <viscosity>cell</viscosity>
  <viscosity-linear-coef>.5</viscosity-linear-coef>
  <viscosity-quadratic-coef>.6</viscosity-quadratic-coef>

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
