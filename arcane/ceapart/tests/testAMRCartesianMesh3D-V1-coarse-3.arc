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
       <lx nx='40' prx='1.0'>4.0</lx>
       <ly ny='60' pry='1.0'>12.0</ly>
       <lz nz='20' prz='1.0'>2.0</lz>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <verbosity-level>0</verbosity-level>
   <renumber-patch-method>1</renumber-patch-method>
   <coarse-at-init>true</coarse-at-init>
   <refinement-3d>
     <position>1.0 2.0 0.5</position>
     <length>1.0 2.0 1.0</length>
   </refinement-3d>
   <refinement-3d>
     <position>1.4 3.0 1.0</position>
     <length>0.5 1.0 0.5</length>
   </refinement-3d> 
   <refinement-3d>
     <position>1.4 3.0 1.0</position>
     <length>0.5 1.0 0.5</length>
   </refinement-3d> 
   <refinement-3d>
     <position>2.4 4.0 0.5</position>
     <length>0.2 0.4 0.4</length>
   </refinement-3d> 
   <expected-number-of-cells-in-patchs>6000 48000 8000 8000 64000 128</expected-number-of-cells-in-patchs>
   <expected-number-of-ghost-cells-in-patchs>2448 19584 5440 5440 43520 32</expected-number-of-ghost-cells-in-patchs>
   <nodes-uid-hash>a64c23c9390d624cb33caf8b24faec1b</nodes-uid-hash>
   <faces-uid-hash>6c7bfcefd4cfb8022db87a31ce5a3b62</faces-uid-hash>
   <cells-uid-hash>12c3d384e39ef3a038e73491d63e6745</cells-uid-hash>
   <nodes-direction-hash>731d24216cdc5b10690672004c811f38</nodes-direction-hash>
   <faces-direction-hash>8f31f320fceb0e4be70dc48fcfb2cf4e</faces-direction-hash>
   <cells-direction-hash>3524f3ccc6702fc0ae6edc014418d44e</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
