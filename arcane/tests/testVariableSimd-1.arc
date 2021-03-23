<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>TestVariableSimd 1</titre>
  <description>Teste la vectorisation sur les variables du maillage</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>15</y><z>15</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="VariableSimdUnitTest">
  </test>
 </module-test-unitaire>

</cas>
