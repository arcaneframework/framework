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

<custom-mesh-test>
  <mesh-size>
    <nb-cells>2</nb-cells>
    <nb-faces>13</nb-faces>
    <nb-edges>26</nb-edges>
    <nb-nodes>16</nb-nodes>
  </mesh-size>
</custom-mesh-test>

</cas>
