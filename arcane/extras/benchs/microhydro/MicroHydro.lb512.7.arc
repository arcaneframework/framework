<?xml version="1.0"?>
<case codename="MicroHydro" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Tube a choc de Sod avec accelerateur</title>
    <timeloop>MicroHydroLoop</timeloop>
    <modules>
      <module name="ArcaneLoadBalance" active="true" />
      <module name="AdditionalVariables" active="true" />
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <!-- La validite numerique suppose que le maillage est 100x15x15.
           Si ce n'est pas le cas, il faut desactiver l'option 'check-numerical-result' -->

      <ghost-layer-builder-version>4</ghost-layer-builder-version>
      <generator name="Cartesian3D" >
        <nb-part-x>8</nb-part-x>
        <nb-part-y>8</nb-part-y>
        <nb-part-z>8</nb-part-z>
        <origin>1.0 2.0 3.0</origin>
        <generate-sod-groups>true</generate-sod-groups>
        <x><n>480</n><length>1.0</length></x>
        <y><n>64</n><length>0.3</length></y>
        <z><n>64</n><length>0.3</length></z>
      </generator>

      <initialization>
        <variable><name>Density</name><value>1.0</value><group>ZG</group></variable>
        <variable><name>Density</name><value>0.125</value><group>ZD</group></variable>

        <variable><name>Pressure</name><value>1.0</value><group>ZG</group></variable>
        <variable><name>Pressure</name><value>0.1</value><group>ZD</group></variable>

        <variable><name>AdiabaticCst</name><value>1.4</value><group>ZG</group></variable>
        <variable><name>AdiabaticCst</name><value>1.4</value><group>ZD</group></variable>
      </initialization>
    </mesh>
  </meshes>

  <arcane-load-balance>
    <active>true</active>
    <!-- <partitioner name="Metis" /> -->
    <partitioner name="MeshPartitionerTester">
      <sub-rank-divider>8</sub-rank-divider>
    </partitioner>
    <period>5</period>
    <statistics>true</statistics>
    <max-imbalance>0.01</max-imbalance>
    <min-cpu-time>0</min-cpu-time>
  </arcane-load-balance>

  <additional-variables>
    <nb-additional-cell-variable>600</nb-additional-cell-variable>
    <cell-array-variable-size>50</cell-array-variable-size>
  </additional-variables>

  <!-- Configuration du module hydrodynamique -->
  <micro-hydro>
    <check-numerical-result>false</check-numerical-result>
    <deltat-init>0.001</deltat-init>
    <deltat-min>0.0001</deltat-min>
    <deltat-max>0.01</deltat-max>
    <!-- <final-time>2.76e-3</final-time> -->
    <final-time>7.52e-3</final-time>

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
  </micro-hydro>
</case>
