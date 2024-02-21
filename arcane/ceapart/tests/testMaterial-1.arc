<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test MeshMaterial</titre>

  <description>Test de la gestion materiaux</description>

  <boucle-en-temps>MeshMaterialTestLoop</boucle-en-temps>

  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>

 </arcane>

 <arcane-post-traitement>
   <periode-sortie>2</periode-sortie>
   <depouillement>
    <variable>Density</variable>
    <variable>Density_ENV1_MAT1</variable>
    <variable>Pressure</variable>
    <groupe>StdMat_ENV1</groupe>
    <groupe>StdMat_ENV1_MAT1</groupe>
    <groupe>AllCells</groupe>
   </depouillement>
 </arcane-post-traitement>
 
 
 <maillage>
  <meshgenerator>
    <sod>
      <x set='false' delta='0.02'>20</x> <!-- Keep 50 to set 0.02 units -->
      <y set='false' delta='0.02'>5</y>
      <z set='false' delta='0.02'>5</z>
  </sod>
  </meshgenerator>
 </maillage>

 <mesh-material-tester>
  <material>
   <name>MAT1</name>
  </material>

  <environment>
   <name>ENV1</name>
   <material>MAT1</material>
  </environment>

 </mesh-material-tester>

</cas>
