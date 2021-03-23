<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage utilise-unite="0">
  <fichier internal-partition="true">tied_interface_2d_1.vtk</fichier>
  <interfaces-liees>
   <semi-conforme esclave="Cb1_Zn2" />
  </interfaces-liees>
 </maillage>
 <module-test-unitaire>
  <test name="MeshUnitTest" />
 </module-test-unitaire>
</cas>
