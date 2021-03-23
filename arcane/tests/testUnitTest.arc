<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test unit test</titre>
  <description>Test de la gestion des tests unitaires dans Arcane</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>100</x><y>5</y><z>5</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <xml-test name="TestUnitTest">
   <my-int>5</my-int>
   <my-double>5.5</my-double>
   <my-boolean>true</my-boolean>
  </xml-test>
 </module-test-unitaire>
</cas>
