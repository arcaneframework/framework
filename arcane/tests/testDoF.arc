<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test DoF</titre>
  <description>Test DoF</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
<!--    <fichier internal-partition="true">sphere.vtk</fichier> -->
   <meshgenerator><sod><x>2</x><y>2</y><z>2</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="DoFTester" />
 </module-test-unitaire>

</cas>
