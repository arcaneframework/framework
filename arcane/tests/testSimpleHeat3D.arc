<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>SimpleHeatTestLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian3D">
        <nb-part-x>2</nb-part-x>
        <nb-part-y>2</nb-part-y>
        <nb-part-z>1</nb-part-z>
        <origin>0.0 0.0 -0.0</origin>
        <x><n>48</n><length>10.0</length><progression>1.05</progression></x>
        <y><n>12</n><length>3.0</length></y>
        <z><n>8</n><length>2.0</length></z>
      </generator>
    </mesh>
  </meshes>

  <arcane-post-processing>
    <output-period>10</output-period>
    <output>
      <variable>CellTemperature</variable>
      <variable>NodeTemperature</variable>
    </output>
  </arcane-post-processing>
  <simple-heat-test>
    <nb-iteration>50</nb-iteration>
  </simple-heat-test>
</case>
