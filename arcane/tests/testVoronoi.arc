<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage Voronoï</titre>
  <description>Test Maillage Voronoï</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
  <modules>
   <module name="ArcanePostProcessing" active="true" />
  </modules>
 </arcane>

 <arcane-post-traitement>
  <periode-sortie>1</periode-sortie>
  <depouillement>
   <variable>DomainId</variable>
   <variable>CellFlags</variable>
   <groupe>AllCells</groupe>
<!-- 
   <groupe>AllFaces</groupe>
   <variable>FaceFlags</variable>
-->
  </depouillement>
 </arcane-post-traitement>

 <maillage>
   <fichier internal-partition="true">voronoi.vor</fichier>
 </maillage>

 <module-test-unitaire>
  <test name="VoronoiTest" />
 </module-test-unitaire>

</cas>
