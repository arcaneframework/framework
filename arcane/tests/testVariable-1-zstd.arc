<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>15</y><z>5</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="VariableUnitTest">
    <nb-reference>10</nb-reference>
    <compressor name="zstdDataCompressor" />
  </test>
 </module-test-unitaire>

</cas>
