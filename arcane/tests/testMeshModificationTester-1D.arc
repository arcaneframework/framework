<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1D</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <meshgenerator><sod><x>10</x></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="MeshModificationTester" />
 </module-test-unitaire>

</cas>
