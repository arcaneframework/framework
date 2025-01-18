<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test maillage avec entités libres</titre>
  <description>Test maillage avec entités libres</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>

 <meshes>
   <mesh>
     <filename>mesh_with_loose_items.msh</filename>
     <allow-loose-items>true</allow-loose-items>
     <face-numbering-version>0</face-numbering-version>
   </mesh>
 </meshes>

 <module-test-unitaire>
  <test name="MeshUnitTest">
    <create-edges>true</create-edges>
  </test>
 </module-test-unitaire>

</cas>
