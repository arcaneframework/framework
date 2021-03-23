<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
  <title>Tube à choc de Sod</title>
  <timeloop>HydroSimpleDepend</timeloop>
 </arcane>

 <mesh>

  <meshgenerator><sod><x>50</x><y>5</y><z>5</z></sod></meshgenerator>

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
