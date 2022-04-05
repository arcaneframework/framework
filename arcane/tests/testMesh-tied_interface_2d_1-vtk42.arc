<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage VTK 4.2 tied_interface_2d_1</titre>
  <description>Test Maillage VTK 4.2 tied_interface_2d_1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition="true">tied_interface_2d_1_v4.2.vtk</fichier>
 </maillage>

 <module-test-unitaire>
   <test name="MeshUnitTest">
     <test-adjacence>0</test-adjacence>
   </test>
 </module-test-unitaire>

</cas>
