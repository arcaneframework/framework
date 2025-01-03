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
 
 
 <maillage amr="true" nb-ghostlayer="3" ghostlayer-builder-version="3">
   <meshgenerator>
     <cartesian>
       <nsd>2 2</nsd>
       <origine>0.0 0.0</origine>
       <lx nx='2' prx='1.0'>2.0</lx>
       <lx nx='4' prx='1.1'>3.0</lx>
       <lx nx='4' prx='1.2'>3.0</lx>
       <lx nx='6' prx='1.3'>4.0</lx>

       <ly ny='2' pry='1.0'>2.0</ly>
       <ly ny='6' pry='1.1'>4.0</ly>
       <ly ny='4' pry='1.2'>5.0</ly>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <renumber-patch-method>1</renumber-patch-method>
   <coarse-at-init>true</coarse-at-init>
   <refinement-2d>
     <position>1.0 2.0</position>
     <length>1.0 2.0</length>
   </refinement-2d>
   <refinement-2d>
     <position>1.4 3.0</position>
     <length>0.5 1.0</length>
   </refinement-2d>
   <refinement-2d>
     <position>4.0 5.0</position>
     <length>3.0 4.0</length>
   </refinement-2d>
   <refinement-2d>
     <position>5.0 7.0</position>
     <length>2.0 2.0</length>
   </refinement-2d>
   <expected-number-of-cells-in-patchs>48 192 12 8 64 72</expected-number-of-cells-in-patchs>
   <expected-number-of-ghost-cells-in-patchs>120 480 12 8 192 216</expected-number-of-ghost-cells-in-patchs>
   <nodes-uid-hash>abe468e571c521f6e1d0537d07c82d9f</nodes-uid-hash>
   <faces-uid-hash>671e325a4fcd2785a83f38c5c227fbff</faces-uid-hash>
   <cells-uid-hash>d21fc971bb27a0e0ed872cb755912f64</cells-uid-hash>
   <nodes-direction-hash>137df0471a6b16dfaa4ce0986eaf98ab</nodes-direction-hash>
   <faces-direction-hash>58caef0c2449b9091f81e6ed756ae482</faces-direction-hash>
   <cells-direction-hash>74c1b4e69029f1f132a94cd56e67e1c1</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
