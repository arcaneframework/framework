<?xml version="1.0" ?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>HLLC Scheme</title>
    <timeloop>HllcSchemeLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian3D">
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <nb-part-z>1</nb-part-z>
        <origin>0.0 0.0 0.0</origin>
        <x><n>100</n><length>1.0</length></x>
        <y><n>4</n><length>0.2</length></y>
        <z><n>4</n><length>0.2</length></z>
        <face-numbering-version>1</face-numbering-version>
        <generate-sod-groups>true</generate-sod-groups>
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
    </output>
  </arcane-post-processing>
  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <hllc-scheme>
    <cfl>0.5</cfl>
    <final-time>10.0e-2</final-time>
    <deltat-init>1.0e-5</deltat-init>
    <gamma>1.4</gamma>
    <spatial-order>2</spatial-order>
    <limiter>VanLeer</limiter>

    <boundary-condition>
      <type>Outflow</type>
      <surface>XMIN</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Outflow</type>
      <surface>XMAX</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Outflow</type>
      <surface>YMIN</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Outflow</type>
      <surface>YMAX</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Outflow</type>
      <surface>ZMIN</surface>
    </boundary-condition>
    <boundary-condition>
      <type>Outflow</type>
      <surface>ZMAX</surface>
    </boundary-condition>
  </hllc-scheme>
</case>
