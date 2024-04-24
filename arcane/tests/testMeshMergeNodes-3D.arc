<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test fusion des noeuds 2D</titre>
  <description>Test fusion des noeuds 2D</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition='true'>merge_nodes_3d.vtk</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="MeshMergeNodesUnitTest">
  </test>
 </module-test-unitaire>

</cas>
