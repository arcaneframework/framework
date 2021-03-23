<?xml version='1.0'?>
<case codeversion="1.0" codename="Poisson" xml:lang="en">
  <arcane>
    <title>Exemple Arcane de l'équation de Poisson</title>
    <timeloop>PoissonLoop</timeloop>
  </arcane>

  <arcane-post-processing>
    <output-period>10</output-period>
    <output>
      <variable>CellTemperature</variable>
      <variable>NodeTemperature</variable>
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

  <poisson>
    <init-temperature>300</init-temperature>
    <boundary-condition>
      <surface>ZMIN</surface>
      <type>Temperature</type>
      <value>900</value>
    </boundary-condition>
    <boundary-condition>
      <surface>XMIN</surface>
      <type>Temperature</type>
      <value>900</value>
    </boundary-condition>
  </poisson>
</case>
