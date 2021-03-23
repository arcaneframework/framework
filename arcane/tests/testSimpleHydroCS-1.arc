<?xml version='1.0'?>
<case codeversion="1.0" codename="ArcaneTest" xml:lang="en">
  <arcane>
    <title>Exemple Arcane d'un module Hydro très, très simplifié</title>
    <timeloop>SimpleHydroCSLoop</timeloop>
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
      <!-- <variable>CellVolume</variable> -->
    </output>
  </arcane-post-processing>

  <mesh>
    <meshgenerator>
      <sod>
        <x>100</x>
        <y>5</y>
        <z>5</z>
      </sod>
    </meshgenerator>
  </mesh>


  <!-- Configuration du module hydrodynamique -->
  <simple-hydro-c-s>
    <viscosity>cell</viscosity>
    <deltat-init>0.001</deltat-init>
    <deltat-min>0.00001</deltat-min>
    <deltat-max>0.0001</deltat-max>
    <final-time>0.05</final-time>
    <test-int32>32</test-int32>
    <test-int32>2</test-int32>
    <test-int32>42</test-int32>
    <test-int32>52</test-int32>
    <test-int32>632</test-int32>
    <!-- <boundary-condition>
	 <surface>XMIN</surface>
	 <type>Vx</type>
	 <value>0.</value>
	 </boundary-condition> -->

    <sub-type>Vx</sub-type>
    <sub-type>Vy</sub-type>
    <sub-type>Vz</sub-type>
    <sub-type>Vx</sub-type>
    <sub-type>Vy</sub-type>
    <sub-type>Vz</sub-type>
    <sub-type>Vx</sub-type>
    <sub-type>Vy</sub-type>
    <sub-type>Vz</sub-type>

    <volume>ZG</volume>
    <volume>ZD</volume>
    <volume>ZD</volume>
    <volume>XMIN</volume>
    <volume>XMAX</volume>

    <string-test>124321412vdscxsx</string-test>
    <string-test>12432132321412vdscxsx</string-test>
    <string-test>12432123f23f412vdscxsx</string-test>
    <string-test>124125346764y321412vdscxsx</string-test>

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
		
    <eos-model name="PerfectGas">
      <Test>1.0</Test>
    </eos-model>

    <!--
        <eos-model name="AirGas"/>
	<eos-model-array name="AirGas"/>
    -->
    <eos-model-array name="PerfectGas" >
      <Test>3.2</Test>
    </eos-model-array>

    <eos-model-array name="PerfectGas" >
    </eos-model-array>

    <eos-model-array name="PerfectGas" >
      <Test>6.3</Test>
    </eos-model-array>

    <!-- <eos-model-array name="StiffenedGas">
         <limit-tension>0.01</limit-tension>
         </eos-model-array> -->
    <!-- <eos-model name="PerfectGas"/> -->

    <!-- <eos-model name="StiffenedGas">
         <limit-tension>0.01</limit-tension>
         </eos-model> -->
  </simple-hydro-c-s>
</case>
