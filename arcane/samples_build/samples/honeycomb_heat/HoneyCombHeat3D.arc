<?xml version="1.0"?>
<case codename="HoneyCombHeat" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Sample</title>
    <timeloop>HoneyCombHeatLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="HoneyComb3D" >
        <origin>0.0 0.0</origin>
        <nb-layer>15</nb-layer>
        <pitch-size>1.2</pitch-size>
       <heights>1.2 5.7 9.2 12.3 21.4</heights>
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
