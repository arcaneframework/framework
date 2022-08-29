<?xml version="1.0" ?>
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
     <generator name="Cartesian3D">
       <nb-part-x>2</nb-part-x> 
       <nb-part-y>2</nb-part-y>
       <nb-part-z>3</nb-part-z>
       <origin>-3.0 -3.0 -3.0</origin>
       <x><n>30</n><length>6.0</length></x>
       <y><n>30</n><length>6.0</length></y>
       <z><n>30</n><length>6.0</length></z>
       <face-numbering-version>1</face-numbering-version>
     </generator>
   </mesh>
 </meshes>

 <cartesian-mesh-tester>
   <unstructured-mesh-file>sphere.vtk</unstructured-mesh-file>
   <expected-mesh-origin>-3.0 -3.0 -3.0</expected-mesh-origin>
   <expected-mesh-length>6.0 6.0 6.0</expected-mesh-length>
 </cartesian-mesh-tester>
</cas>
