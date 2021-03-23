<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test PTScotch</titre>
  <description>Teste partitionnement avec PTScotch</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition="true" partitioner="PTScotch">sod.vtk</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="MeshUnitTest" />
 </module-test-unitaire>

</cas>
