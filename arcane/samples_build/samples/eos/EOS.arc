<?xml version='1.0'?>
<case codeversion="1.0" codename="EOS" xml:lang="en">
  <arcane>
    <title>Module de test des EOS</title>
    <timeloop>EOSLoop</timeloop>
  </arcane>

  <mesh>
    <meshgenerator>
      <sod>
        <x>100</x>
        <y>5</y>
        <z>5</z>
      </sod>
    </meshgenerator>
    <initialisation>
      <variable nom="Density" valeur="1." groupe="ZG" />
      <variable nom="Pressure" valeur="1." groupe="ZG" />
      <variable nom="AdiabaticCst" valeur="1.4" groupe="ZG" />
      <variable nom="Density" valeur="0.125" groupe="ZD" />
      <variable nom="Pressure" valeur="0.1" groupe="ZD" />
      <variable nom="AdiabaticCst" valeur="1.4" groupe="ZD" />
    </initialisation>
  </mesh>
  <e-o-s>
    <eos-model name="PerfectGas" />
  </e-o-s>
</case>
