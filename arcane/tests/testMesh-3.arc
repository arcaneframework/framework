<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 2D</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>10</y><random-coef>0.5</random-coef></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="MeshUnitTest">
   <ecrire-maillage>true</ecrire-maillage>
  </test>
 </module-test-unitaire>

</cas>
