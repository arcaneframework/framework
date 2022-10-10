<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test Execution directe avec CartesianMeshGenerator</titre>
    <description>Test de la generation de maillages cartesiens</description>
  </arcane>

 <meshes>
   <mesh>
     <generator name="Cartesian2D">
       <nb-part-x>2</nb-part-x> 
       <nb-part-y>2</nb-part-y>
       <origin>0.0 0.0</origin>
       <generate-sod-groups>true</generate-sod-groups>
       <x><n>2</n><length>2.0</length></x>
       <x><n>3</n><length>3.0</length></x>
       <x><n>3</n><length>3.0</length></x>
       <x><n>4</n><length>4.0</length></x>

       <y><n>2</n><length>2.0</length></y>
       <y><n>3</n><length>3.0</length></y>
       <y><n>3</n><length>3.0</length></y>
       <face-numbering-version>1</face-numbering-version>
     </generator>
   </mesh>
 </meshes>

</cas>
