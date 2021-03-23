<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test MeshMaterial</titre>

  <description>Test de la gestion materiaux</description>

  <boucle-en-temps>MeshMaterialTestLoop</boucle-en-temps>

 </arcane>

 <maillage>
  <meshgenerator>
    <sod>
      <x set='false' delta='0.02'>20</x> <!-- Keep 50 to set 0.02 units -->
      <y set='false' delta='0.02'>15</y>
      <z set='false' delta='0.02'>32</z>
  </sod>
  </meshgenerator>
 </maillage>

 <mesh-material-tester>
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

</cas>
