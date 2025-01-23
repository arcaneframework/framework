<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Tube a choc de Sod</title>
    <timeloop>ArcaneHydroLoop</timeloop>
    <!---<modules>
      <module name="ArcanePostProcessing" need="required" />
      <module name="ArcanePostProcessing" active="true" />
    </modules>-->
  </arcane>

  <meshes>
    <mesh>
    <!--<generator name="Cartesian3D" >
      <nb-part-x>2</nb-part-x>
      <nb-part-y>0</nb-part-y>
      <nb-part-z>0</nb-part-z>
      <origin>1.0 2.0 3.0</origin>
      <x><n>2</n><length>4.0</length></x>
      <y><n>1</n><length>4.0</length></y>
      <z><n>1</n><length>4.0</length></z>
      <face-numbering-version>1</face-numbering-version>
    </generator>
    <subdivider>
      <nb-subdivision>1</nb-subdivision>
    </subdivider>
    <partitioner>MeshPartitionerTester</partitioner>
    <initialization>
      <variable><name>Density</name><value>1.0</value><group>ZG</group></variable>
      <variable><name>Density</name><value>0.125</value><group>ZD</group></variable>

      <variable><name>Pressure</name><value>1.0</value><group>ZG</group></variable>
      <variable><name>Pressure</name><value>0.1</value><group>ZD</group></variable>

      <variable><name>AdiabaticCst</name><value>1.4</value><group>ZG</group></variable>
      <variable><name>AdiabaticCst</name><value>1.4</value><group>ZD</group></variable>
    </initialization>-->

    <filename internal-partition="true">sod.vtk</filename>
    <subdivider>
      <nb-subdivision>1</nb-subdivision>

    </subdivider>
    <partitioner>MeshPartitionerTester</partitioner>
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
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>
  <!--Postprocessing-->
  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>Pressure</variable>
      <variable>Density</variable>
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
