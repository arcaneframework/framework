<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <meshes>
  <mesh>
   <filename>elements.vtu</filename>
   <cell-dimension-kind>multi-dimension</cell-dimension-kind>
  </mesh>
 </meshes>


 <module-test-unitaire>
  <test name="MeshUnitTest">
   <ecrire-maillage>false</ecrire-maillage>
   <test-ecrivain-variable>0</test-ecrivain-variable>
   <test-adjacence>false</test-adjacence>
  </test>
 </module-test-unitaire>

</cas>
