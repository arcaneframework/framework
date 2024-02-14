<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test MeshMaterial</titre>

  <description>Test de la gestion materiaux: 1 seul milieu, 3 materiaux</description>

  <boucle-en-temps>MeshMaterialTestLoop</boucle-en-temps>

  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>

 </arcane>

 <arcane-post-traitement>
   <periode-sortie>1</periode-sortie>
   <depouillement>
    <variable>Density</variable>
    <variable>Density_ENV1_MAT1</variable>
    <variable>Density_ENV1_MAT2</variable>
    <variable>Density_ENV1_MAT3</variable>
    <variable>Pressure</variable>
    <variable>StdMat_CellNbMaterial_ENV1</variable>
    <groupe>StdMat_ENV1</groupe>
    <groupe>StdMat_ENV1_MAT1</groupe>
    <groupe>StdMat_ENV1_MAT2</groupe>
    <groupe>StdMat_ENV1_MAT3</groupe>
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
  <modification-flags>5</modification-flags>
  <material>
   <name>MAT1</name>
  </material>
  <material>
   <name>MAT2</name>
  </material>
  <material>
   <name>MAT3</name>
  </material>

  <environment>
   <name>ENV1</name>
   <material>MAT1</material>
   <material>MAT2</material>
   <material>MAT3</material>
  </environment>

 </mesh-material-tester>

</cas>
