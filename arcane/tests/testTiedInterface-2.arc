<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage utilise-unite="false">
  <fichier internal-partition="true">tied_interface_2.vtk</fichier>
  <interfaces-liees>
   <semi-conforme esclave="INTERFACE_1_M_PART1" />
   <semi-conforme esclave="INTERFACE_2_M_PART1" />
  </interfaces-liees>
 </maillage>
 <module-test-unitaire>
  <test name="MeshUnitTest" />
 </module-test-unitaire>
</cas>
