<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test MeshMaterialSimd</titre>
  <description>Test accélérateur des materiaux</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>
 <maillage>
  <meshgenerator>
    <sod>
      <x set='false' delta='0.02'>100</x> <!-- Keep 50 to set 0.02 units -->
      <y set='false' delta='0.02'>15</y>
      <z set='false' delta='0.02'>15</z>
  </sod>
  </meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="MeshMaterialAcceleratorUnitTest" />
 </module-test-unitaire>
</cas>
