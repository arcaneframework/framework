<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Zoltan</titre>
  <description>Teste partitionnement avec Zoltan</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition="true" partitioner="Zoltan">sod.vtk</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="MeshUnitTest" />
 </module-test-unitaire>

</cas>
