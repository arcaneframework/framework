<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1D</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>10</x><y>4</y><z>4</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="MeshModificationTester" />
 </module-test-unitaire>

</cas>
