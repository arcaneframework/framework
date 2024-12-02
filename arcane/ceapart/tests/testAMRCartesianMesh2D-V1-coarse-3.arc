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
       <lx nx='80' prx='1.0'>8.0</lx>
       <ly ny='80' pry='1.0'>8.0</ly>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <verbosity-level>0</verbosity-level>
   <dump-svg>false</dump-svg>
   <renumber-patch-method>1</renumber-patch-method>
   <coarse-at-init>true</coarse-at-init>
   <refinement-2d>
     <position>0.0 0.0</position>
     <length>1.1 1.1</length>
   </refinement-2d>

   <expected-number-of-cells-in-patchs>1600 6400 484</expected-number-of-cells-in-patchs>
   <expected-number-of-ghost-cells-in-patchs>516 2064 0</expected-number-of-ghost-cells-in-patchs>
   <nodes-uid-hash>123ea5235d1fbd0f946407da35010199</nodes-uid-hash>
   <faces-uid-hash>406c3302e3f2fedf56129dfbdfd27617</faces-uid-hash>
   <cells-uid-hash>bd641ece2308db5e65f6c967c0e08478</cells-uid-hash>
   <nodes-direction-hash>bc756d56298def13cb83acb56738652e</nodes-direction-hash>
   <faces-direction-hash>2a7795b689b0cedcaa1f0bd2ee5f5619</faces-direction-hash>
   <cells-direction-hash>775f249f3fb3892d8cf5212a0cb06edc</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
