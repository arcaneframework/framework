<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage Custom 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>CustomMeshTestLoop</boucle-en-temps>
 </arcane>

 <meshes>
   <mesh name="PolyhedralMesh">
<!--     <fichier>example_polyhedral_cell.xmf</fichier>-->
     <fichier>faultx1_2x1x1.vtk</fichier>
   </mesh>
 </meshes>

<!-- <module-test-unitaire>-->
<!--  <test name="MeshUnitTest">-->
<!--   <ecrire-maillage>true</ecrire-maillage>-->
<!--  </test>-->
<!-- </module-test-unitaire>-->

</cas>
