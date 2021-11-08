<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test Maillage 1</titre>
  <description>Test Maillage 1</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <meshes>
   <mesh>
     <generator name="Sod3D">
       <x>10</x><y>5</y><z>5</z>
     </generator>
   </mesh>
 </meshes>

 <module-test-unitaire>
  <test name="MeshUnitTest">
   <ecrire-maillage>true</ecrire-maillage>
  </test>
 </module-test-unitaire>

</cas>
