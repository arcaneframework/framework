<?xml version="1.0" ?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test DoF with nodes</titre>
    <description>Test DoF</description>
    <boucle-en-temps>UnitTest</boucle-en-temps>
  </arcane>

 <meshes>
   <mesh>
     <generator name="Cartesian3D">
       <nb-part-x>2</nb-part-x>
       <nb-part-y>2</nb-part-y>
       <nb-part-z>1</nb-part-z>
       <origin>0.0 0.0 0.0</origin>
       <x><n>12</n><length>5.0</length></x>
       <y><n>6</n><length>3.0</length></y>
       <z><n>4</n><length>2.0</length></z>
     </generator>
   </mesh>
 </meshes>

  <module-test-unitaire>
    <test name="DoFNodeTestService" />
  </module-test-unitaire>

</cas>
