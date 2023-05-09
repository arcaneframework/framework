<?xml version="1.0"?>
<case codename="HoneyCombHeat" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>HoneyCombHeatLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="HoneyComb2D" >
        <origin>0.0 0.0</origin>
        <nb-layer>5</nb-layer>
        <pitch-size>1.0</pitch-size>
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
</case>
