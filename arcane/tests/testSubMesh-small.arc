<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Parallel GhostItemsReduceOperation</titre>
  <description>Test Parallel GhostItemsReduceOperation</description>
  <!--boucle-en-temps>SubMeshTest</boucle-en-temps-->
  <boucle-en-temps>SubMeshTestLoop</boucle-en-temps>
  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

 <maillage>
<!--    <fichier internal-partition="true">sod.vtk</fichier> -->
<!--    <fichier internal-partition="true">tube5x5x100.vtk</fichier> -->
 <meshgenerator><sod><x>2</x><y>1</y><z total="true">3</z></sod></meshgenerator>
  <initialisation />
 </maillage>

 <arcane-protections-reprises>
  <service-protection name="ArcaneHdf5MultiCheckpoint"/>
  <!--service-protection name="ArcaneBasicCheckpointWriter"/-->
  <!--service-protection name="ArcaneHdf5Checkpoint2"/-->
  <!--service-protection name="ArcaneHdf5MpiCheckpoint2"/--> <!-- hdf5mpi: restart may fail -->
  <periode>0</periode>
  <en-fin-de-calcul>true</en-fin-de-calcul>
 </arcane-protections-reprises>

 <sub-mesh-test>
  <format-service name="Ensight7PostProcessor">
    <fichier-binaire>false</fichier-binaire>
  </format-service>
  <nb-iteration>10</nb-iteration>
  <genre-sous-maillage>cell</genre-sous-maillage>
 </sub-mesh-test>

 <arcane-post-traitement>
  <periode-sortie>1</periode-sortie>
  <format-service name="Ensight7PostProcessor">
    <fichier-binaire>false</fichier-binaire>
  </format-service>
  <depouillement>
   <variable>TestParallelCellRealValues</variable>
<!--   <variable>TestParallelFaceRealValues</variable> -->
   <variable>Data</variable>
     <groupe>AllCells</groupe>
<!--     <groupe>AllFaces</groupe> -->
  </depouillement>
 </arcane-post-traitement>


</cas>
