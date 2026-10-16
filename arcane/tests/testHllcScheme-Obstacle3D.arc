<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>HLLC Scheme - Uniform flow around a solid sphere</title>
    <timeloop>HllcSchemeLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian3D">
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <nb-part-z>1</nb-part-z>
        <origin>0.0 0.0 0.0</origin>
        <x><n>50</n><length>2.0</length></x>
        <y><n>25</n><length>1.0</length></y>
        <z><n>25</n><length>1.0</length></z>
      </generator>
    </mesh>
  </meshes>

  <arcane-post-processing>
    <output-period>10</output-period>
    <output>
      <variable>Density</variable>
      <variable>Pressure</variable>
      <variable>Momentum</variable>
      <variable>SoundSpeed</variable>
      <variable>CellVolume</variable>
      <variable>CellIsSolid</variable>
    </output>
  </arcane-post-processing>
  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <hllc-scheme>
    <cfl>0.5</cfl>
    <final-time>0.2</final-time>
    <deltat-init>1.0e-5</deltat-init>
    <gamma>1.4</gamma>
    <spatial-order>2</spatial-order>
    <limiter>VanLeer</limiter>

    <initial-condition>Uniform</initial-condition>
    <uniform-density>1.0</uniform-density>
    <uniform-pressure>1.0</uniform-pressure>
    <uniform-velocity-x>1.0</uniform-velocity-x>
    <uniform-velocity-y>0.0</uniform-velocity-y>
    <uniform-velocity-z>0.0</uniform-velocity-z>

    <!-- Solid sphere immersed in the fluid -->
    <obstacle-x>0.5</obstacle-x>
    <obstacle-y>0.5</obstacle-y>
    <obstacle-z>0.5</obstacle-z>
    <obstacle-radius>0.15</obstacle-radius>

    <boundary-condition>
      <type>Inflow</type>
      <surface>XMIN</surface>
      <density>1.0</density>
      <velocity-x>1.0</velocity-x>
      <velocity-y>0.0</velocity-y>
      <velocity-z>0.0</velocity-z>
      <pressure>1.0</pressure>
    </boundary-condition>
    <boundary-condition>
      <type>Outflow</type>
      <surface>XMAX</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Wall</type>
      <surface>YMIN</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Wall</type>
      <surface>YMAX</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Wall</type>
      <surface>ZMIN</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Wall</type>
      <surface>ZMAX</surface>
    </boundary-condition>
  </hllc-scheme>
</case>
