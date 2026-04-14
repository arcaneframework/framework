<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test MeshMaterial</titre>

  <description>Test de la gestion materiaux</description>

  <boucle-en-temps>MeshMaterialTestLoop</boucle-en-temps>

  <modules>
    <module name="ArcanePostProcessing" active="true" />
    <module name="ArcaneCheckpoint" active="true" />
  </modules>

 </arcane>

 <arcane-post-traitement>
   <periode-sortie>1</periode-sortie>
   <depouillement>
    <variable>Density</variable>
    <variable>Density_ENV1_MAT1</variable>
    <variable>Density_ENV1_MAT2</variable>
    <variable>Density_ENV2_MAT2</variable>
    <variable>Pressure</variable>
    <groupe>StdMat_ENV1</groupe>
    <groupe>StdMat_ENV2</groupe>
    <groupe>StdMat_ENV3</groupe>
    <groupe>StdMat_ENV1_MAT1</groupe>
    <groupe>StdMat_ENV1_MAT2</groupe>
    <groupe>StdMat_ENV2_MAT2</groupe>
    <groupe>StdMat_ENV3_MAT1</groupe>
    <groupe>StdMat_ENV3_MAT3</groupe>
   <groupe>AllCells</groupe>
   </depouillement>
 </arcane-post-traitement>
 
 
 <maillage>
  <meshgenerator>
    <sod>
      <x set='false' delta='0.02'>20</x> <!-- Keep 50 to set 0.02 units -->
      <y set='false' delta='0.02'>5</y>
      <z set='false' delta='0.02' total='true'>16</z>
    </sod>
    <!-- <sod>
      <x set='false' delta='0.02'>10</x>
      <y set='false' delta='0.02'>2</y>
      <z set='false' delta='0.02' total='true'>8</z>
    </sod> -->
  </meshgenerator>
 </maillage>

 <mesh-material-tester>
  <recreate-from-dump>true</recreate-from-dump>
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
  </environment>

  <environment>
   <name>ENV2</name>
   <material>MAT2</material>
  </environment>

  <environment>
   <name>ENV3</name>
   <material>MAT3</material>
   <material>MAT1</material>
  </environment>

 </mesh-material-tester>

 <arcane-protections-reprises>
  <en-fin-de-calcul>true</en-fin-de-calcul>
  <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>

</cas>
