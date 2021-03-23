<?xml version='1.0'?>
<case codeversion="1.0" codename="MicroHydro" xml:lang="en">
  <arcane>
    <title>Exemple Arcane d'un module Hydro tres simplifie</title>
    <timeloop>MicroHydroLoop</timeloop>
  </arcane>

  <arcane-post-processing>
    <output-period>10</output-period>
    <output>
      <variable>CellMass</variable>
      <variable>Pressure</variable>
      <variable>Density</variable>
      <variable>Velocity</variable>
      <variable>NodeMass</variable>
      <variable>InternalEnergy</variable>
    </output>
    <format>
      <binary-file>false</binary-file>
    </format>
  </arcane-post-processing>

  <mesh>
    <file internal-partition='true'>sod.vtk</file>
    <initialisation>
      <variable nom="Density" valeur="1." groupe="ZG" />
      <variable nom="Pressure" valeur="1." groupe="ZG" />
      <variable nom="AdiabaticCst" valeur="1.4" groupe="ZG" />
      <variable nom="Density" valeur="0.125" groupe="ZD" />
      <variable nom="Pressure" valeur="0.1" groupe="ZD" />
      <variable nom="AdiabaticCst" valeur="1.4" groupe="ZD" />
    </initialisation>
  </mesh>

  <arcane-checkpoint>
    <period>0</period>
    <!-- Mettre '0' si on souhaite ne pas faire de protections a la fin du calcul -->
    <do-dump-at-end>0</do-dump-at-end>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter" />
  </arcane-checkpoint>

  <!-- Configuration du module hydrodynamique -->
  <micro-hydro>
    <deltat-init>0.001</deltat-init>
    <deltat-min>0.00001</deltat-min>
    <deltat-max>0.0001</deltat-max>
    <final-time>1.09e-2</final-time>
    <boundary-condition>
      <surface>XMIN</surface>
      <type>Vx</type>
      <value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>XMAX</surface>
      <type>Vx</type>
      <value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>YMIN</surface>
      <type>Vy</type>
      <value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>YMAX</surface>
      <type>Vy</type>
      <value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>ZMIN</surface>
      <type>Vz</type>
      <value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>ZMAX</surface>
      <type>Vz</type>
      <value>0.</value>
    </boundary-condition>
		
    <eos-model name="PerfectGas"/>

    <!-- <eos-model name="StiffenedGas">
         <limit-tension>0.01</limit-tension>
         </eos-model> -->
  </micro-hydro>
</case>
