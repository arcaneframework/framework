<?xml version="1.0" encoding="ISO-8859-1"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh</titre>

  <description>Test des maillages cartesiens 2D</description>

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
 
 
 <maillage>
   <meshgenerator>
     <cartesian>
       <nsd>3 2</nsd>
       <origine>0.0 0.0</origine>
       <face-numbering-version>4</face-numbering-version>
       <lx nx='2' prx='1.0'>2.0</lx>
       <lx nx='3' prx='4.0'>3.0</lx>
       <lx nx='3' prx='8.0'>3.0</lx>
       <lx nx='4' prx='16.0'>4.0</lx>

       <ly ny='2' pry='1.0'>2.0</ly>
       <ly ny='3' pry='4.0'>3.0</ly>
       <ly ny='3' pry='8.0'>3.0</ly>
     </cartesian>
   </meshgenerator>
 </maillage>

 <cartesian-mesh-tester>
 </cartesian-mesh-tester>

</cas>
