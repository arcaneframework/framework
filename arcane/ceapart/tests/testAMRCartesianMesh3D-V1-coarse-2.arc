<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh</titre>

  <description>Test des maillages cartesiens AMR 2D</description>

  <boucle-en-temps>AMRCartesianMeshTestLoop</boucle-en-temps>

  <modules>
    <module name="ArcanePostProcessing" active="true" />
    <module name="ArcaneCheckpoint" active="true" />
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
       <nsd>2 2 2</nsd>
       <origine>0.0 0.0 0.0</origine>
       <lx nx='20' prx='1.0'>4.0</lx>
       <ly ny='40' pry='1.0'>12.0</ly>
       <lz nz='12' prz='1.0'>2.0</lz>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <verbosity-level>0</verbosity-level>
   <renumber-patch-method>3</renumber-patch-method>
   <coarse-at-init>true</coarse-at-init>
   <refinement-3d>
     <position>1.0 2.0 0.5</position>
     <length>1.0 2.0 1.0</length>
   </refinement-3d>
   <refinement-3d>
     <position>1.4 3.0 1.0</position>
     <length>0.5 1.0 0.5</length>
   </refinement-3d> 
   <expected-number-of-cells-in-patchs>1200 9600 1440 1440</expected-number-of-cells-in-patchs>
   <expected-number-of-ghost-cells-in-patchs>912 7296 1920 2400</expected-number-of-ghost-cells-in-patchs>
   <nodes-uid-hash>f31f0f9a96f12cb401ebeedf3489b863</nodes-uid-hash>
   <faces-uid-hash>dff1af950378dd219e556868e4c21f0e</faces-uid-hash>
   <cells-uid-hash>8691598c0bccee7c1030d0e30541ed04</cells-uid-hash>
   <nodes-direction-hash>bc756d56298def13cb83acb56738652e</nodes-direction-hash>
   <faces-direction-hash>2a7795b689b0cedcaa1f0bd2ee5f5619</faces-direction-hash>
   <cells-direction-hash>775f249f3fb3892d8cf5212a0cb06edc</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
