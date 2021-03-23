<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage VTK quadratique</titre>
  <description>Test Maillage VTK quadratique</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition="true">quadratic3d.vtk</fichier>
 </maillage>

 <module-test-unitaire>
   <test name="MeshUnitTest">
     <test-adjacence>0</test-adjacence>
   </test>
 </module-test-unitaire>

</cas>
