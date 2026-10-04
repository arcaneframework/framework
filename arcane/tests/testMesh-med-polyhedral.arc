<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage MED polyedrique</titre>
  <description>Test Maillage MED2 polyedrique</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <maillage>
  <fichier internal-partition="true">cube_polyhedral.med</fichier>
 </maillage>

 <module-test-unitaire>
   <test name="MeshUnitTest">
     <test-adjacence>0</test-adjacence>
     <write-mesh-service-name>VtkLegacyMeshWriter</write-mesh-service-name>
   </test>
 </module-test-unitaire>

</cas>
