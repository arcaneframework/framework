<?xml version="1.0" encoding="ISO-8859-1"?>
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
    <periode-sortie>0</periode-sortie>
    <sortie-fin-execution>1</sortie-fin-execution>
    <depouillement>
      <variable>UniqueId</variable>
      <variable>SubDomainId</variable>
      <groupe>AllCells</groupe>
    </depouillement>
  </arcane-post-traitement>
 
  <maillage>
    <meshgenerator>
      <cartesian>
        <nsd>3 1</nsd>
        <origine>0.0 0.0</origine>
        <lx nx="100" prx="1.0">1.0</lx>
        <ly ny="5" pry="1.0">0.1</ly>
      </cartesian>
    </meshgenerator> 
  </maillage>

 <aleph-test-module>
   <schema name="Faces"/>
   <iterations>4</iterations>
   <aleph-number-of-solvers>8</aleph-number-of-solvers>
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
