<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh</titre>

  <description>Test des deraffinement maillage cartesien 3D</description>

  <boucle-en-temps>CartesianMeshTestLoop</boucle-en-temps>

  <modules>
    <module name="ArcanePostProcessing" active="true" />
  </modules>

 </arcane>

 <arcane-post-traitement>
   <periode-sortie>1</periode-sortie>
   <depouillement>
    <variable>Density</variable>
    <variable>NodeDensity</variable>
    <groupe>AllCells</groupe>
    <groupe>AllNodes</groupe>
    <groupe>AllFacesDirection0</groupe>
    <groupe>AllFacesDirection1</groupe>
   </depouillement>
 </arcane-post-traitement>
 
 <maillage amr="true">
   <meshgenerator>
     <cartesian>
       <nsd>2 2 1</nsd>
       <origine>0.0 0.0 0.0</origine>
       <lx nx='3' prx='1.1'>3.0</lx>
       <lx nx='5' prx='1.0'>5.0</lx>

       <ly ny='3' pry='1.0'>2.0</ly>
       <ly ny='5' pry='1.1'>5.0</ly>

       <lz nz='4' prz='1.0'>2.0</lz>
     </cartesian>
   </meshgenerator>
 </maillage>

 <cartesian-mesh-tester>
   <coarse-cartesian-mesh>2</coarse-cartesian-mesh>
 </cartesian-mesh-tester>
</cas>
