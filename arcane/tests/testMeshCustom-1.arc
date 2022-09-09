<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage Custom 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>CustomMeshTestLoop</boucle-en-temps>
  <modules>
    <module name="ArcanePostProcessing" active="true"/>
  </modules>
 </arcane>

 <meshes>
   <mesh name="PolyhedralMesh">
<!--     <fichier>example_polyhedral_cell.xmf</fichier>-->
     <fichier>faultx1_2x1x1.vtk</fichier>
   </mesh>
 </meshes>

<custom-mesh-test>
  <mesh-size>
    <nb-cells>2</nb-cells>
    <nb-faces>13</nb-faces>
    <nb-edges>26</nb-edges>
    <nb-nodes>16</nb-nodes>
  </mesh-size>
</custom-mesh-test>

  <arcane-post-traitement>
    <periode-sortie>1</periode-sortie>
    <depouillement>
      <variable>CellVariable</variable>
      <variable>FaceVariable</variable>
      <variable>NodeVariable</variable>
      <groupe>AllCells</groupe>
      <groupe>AllFaces</groupe>
    </depouillement>
    <sauvegarde-initiale>true</sauvegarde-initiale>
    <format-service name="Ensight7PostProcessor">
      <fichier-binaire>false</fichier-binaire>
    </format-service>
  </arcane-post-traitement>

</cas>
