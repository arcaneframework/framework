<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test DirectedGraph</titre>
  <description>Test DirectedGraph (based on GraphBaseT class)</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
<!--    <fichier internal-partition="true">sphere.vtk</fichier> -->
   <meshgenerator><sod><x>2</x><y>2</y><z>2</z></sod></meshgenerator>
 </maillage>

 <module-test-unitaire>
  <test name="DirectedGraphUnitTest" />
 </module-test-unitaire>

</cas>
