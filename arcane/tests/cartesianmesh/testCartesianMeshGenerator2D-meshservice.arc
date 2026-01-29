<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
  <titre>Test CartesianMeshGenerator</titre>
  <description>Test de la generation de maillages cartesiens</description>
  <boucle-en-temps>AlephTestLoop</boucle-en-temps>
  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>
  </arcane>

  <arcane-post-traitement>
    <sauvegarde-initiale>1</sauvegarde-initiale>
    <periode-sortie>1</periode-sortie>
    <sortie-fin-execution>1</sortie-fin-execution>
    <depouillement>
      <variable>CellTemperature</variable>
      <variable>UniqueId</variable>
      <variable>SubDomainId</variable>
      <groupe>AllCells</groupe>
    </depouillement>
  </arcane-post-traitement>
 
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
       <face-numbering-version>1</face-numbering-version>
     </generator>
   </mesh>
 </meshes>


 <aleph-test-module>
   <schema name="Faces"/>
   <iterations>1</iterations>
   <aleph-number-of-solvers>1</aleph-number-of-solvers>
   <aleph-number-of-cores>1</aleph-number-of-cores>
   <aleph-cell-ordering>false</aleph-cell-ordering>
   <aleph-underlying-solver>2</aleph-underlying-solver>
   <deltaT>0.1</deltaT>
   <init-temperature>300</init-temperature>
   <init-amr>0.0</init-amr>
   <trig-refine>0.01</trig-refine>
   <trig-coarse>0.0004</trig-coarse>
 </aleph-test-module>

</cas>
