<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh</titre>

  <description>Test des maillages cartesiens</description>

  <boucle-en-temps>CartesianMeshTestLoop</boucle-en-temps>

  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>

 </arcane>

 <arcane-post-traitement>
   <periode-sortie>2</periode-sortie>
   <sauvegarde-initiale>true</sauvegarde-initiale>
   <depouillement>
    <variable>Density</variable>
    <groupe>AllCells</groupe>
   </depouillement>
 </arcane-post-traitement>
 
 <!--
     Le maillage de test 'planar_unstructured_quad1.msh' a pour intervalle
     X = [-1.25,1.25] et Y = [-0.5, 1.25 ].
     On prend un tout petit peu plus petit pour le maillage cartésien
 -->
 <meshes>
   <mesh>
     <generator name="Cartesian2D">
       <nb-part-x>2</nb-part-x> 
       <nb-part-y>3</nb-part-y>
       <origin>-1.20 -0.45</origin>
       <x><n>6</n><length>0.5</length><progression>1.1</progression></x>
       <x><n>6</n><length>1.0</length><progression>1.2</progression></x>
       <x><n>6</n><length>0.8</length><progression>1.3</progression></x>

       <y><n>6</n><length>0.5</length><progression>1.1</progression></y>
       <y><n>6</n><length>0.8</length><progression>1.2</progression></y>
       <y><n>6</n><length>0.4</length><progression>1.3</progression></y>
     </generator>
   </mesh>
 </meshes>

 <cartesian-mesh-tester>
   <unstructured-mesh-file>planar_unstructured_quad1.msh</unstructured-mesh-file>
 </cartesian-mesh-tester>
</cas>
