<?xml version="1.0"?>
<cas codename="Arcane" xml:lang="fr" codeversion="1.0">
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
       <x><n>2</n><length>2.0</length><progression>1.0</progression></x>
       <x><n>3</n><length>3.0</length><progression>4.0</progression></x>
       <x><n>3</n><length>3.0</length><progression>8.0</progression></x>
       <x><n>4</n><length>4.0</length><progression>16.0</progression></x>

       <y><n>2</n><length>2.0</length><progression>1.0</progression></y>
       <y><n>3</n><length>3.0</length><progression>4.0</progression></y>
       <y><n>3</n><length>3.0</length><progression>8.0</progression></y>
     </generator>
   </mesh>
 </meshes>

</cas>
