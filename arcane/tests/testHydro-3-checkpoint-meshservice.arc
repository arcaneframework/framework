<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Tube a choc de Sod</title>
    <timeloop>ArcaneHydroLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename internal-partition="true">sod.vtk</filename>
      <partitioner>MeshPartitionerTester</partitioner>
      <face-numbering-version>3</face-numbering-version>
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

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter" />
    <do-dump-at-end>true</do-dump-at-end>
  </arcane-checkpoint>

  <!-- Configuration du module hydrodynamique -->
  <simple-hydro>
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
