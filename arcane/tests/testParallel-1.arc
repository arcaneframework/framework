<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Parallel GhostItemsReduceOperation</titre>
  <description>Test Parallel GhostItemsReduceOperation</description>
  <boucle-en-temps>TestParallel</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition="true">sod.vtk</fichier>
  <!-- <meshgenerator><sod><x>200</x><y>40</y><z>40</z></sod></meshgenerator> -->
  <initialisation />
 </maillage>
 <parallel-tester>
  <test-id>TestAll</test-id>
 </parallel-tester>
</cas>
