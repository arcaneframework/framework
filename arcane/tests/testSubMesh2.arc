<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Parallel GhostItemsReduceOperation</titre>
  <description>Test Parallel GhostItemsReduceOperation</description>
  <!--boucle-en-temps>SubMeshTest</boucle-en-temps-->
  <boucle-en-temps>SubMeshTestLoop</boucle-en-temps>
  <modules>
  </modules>
 </arcane>

 <maillage>
   <fichier internal-partition="true">sod.vtk</fichier>
<!--    <fichier internal-partition="true">tube5x5x100.vtk</fichier> -->
<!-- <meshgenerator><sod><x>20</x><y>20</y><z total="true">20</z></sod></meshgenerator> -->
  <initialisation />
 </maillage>

 <arcane-protections-reprises>
  <en-fin-de-calcul>false</en-fin-de-calcul>
 </arcane-protections-reprises>

 <sub-mesh-test>
  <format-service name="Ensight7PostProcessor">
    <fichier-binaire>false</fichier-binaire>
  </format-service>
  <nb-iteration>10</nb-iteration>
  <genre-sous-maillage>face</genre-sous-maillage>
 </sub-mesh-test>

</cas>
