<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>SimpleHeatTestLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <filename>circle_cut-poly.med</filename>
    </mesh>
  </meshes>

  <arcane-post-processing>
    <output-period>10</output-period>
    <output>
      <variable>CellTemperature</variable>
      <variable>NodeTemperature</variable>
    </output>
  </arcane-post-processing>
  <simple-heat>
    <nb-iteration>50</nb-iteration>
  </simple-heat>
</case>
