<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Arcane 1</titre>
  <description>Test Arcane 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
  <modules>
   <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

 <maillage>
  <fichier internal-partition="true">tube2x2x4.vtk</fichier>
<!--   <fichier internal-partition="true">sod.vtk</fichier> -->
<!--   <meshgenerator><sod><x>2</x><y>3</y><z>2</z></sod></meshgenerator> -->
<!--   <meshgenerator><sod><x>10</x><y>4</y><z>5</z></sod></meshgenerator> -->
<!--   <meshgenerator><sod><x>10</x></sod></meshgenerator> A real 1D mesh is needed -->
 </maillage>

 <module-test-unitaire>
  <test name="ExchangeItemsUnitTest">
    <test-operation>repartition-cells</test-operation>
  </test>
 </module-test-unitaire>

 <arcane-post-traitement>
   <periode-sortie>1</periode-sortie>
   <!-- <format name="EnsightHdfPostProcessor" /> -->
   <sortie-fin-execution>true</sortie-fin-execution>
   <sauvegarde-initiale>true</sauvegarde-initiale>
  <depouillement>
    <variable>CellFamilyNewOwnerName</variable>
    <variable>ExchangeItemsTest_CellUids</variable>
    <variable>GhostPP</variable>
    <variable>NodeGhostPP</variable>
    <variable>FaceGhostPP</variable>
    <groupe>AllCells</groupe>
    <groupe>AllNodes</groupe>
    <groupe>AllFaces</groupe>
  </depouillement>
  <format-service name="Ensight7PostProcessor">
   <fichier-binaire>false</fichier-binaire>
  </format-service>

 </arcane-post-traitement>

</cas>
