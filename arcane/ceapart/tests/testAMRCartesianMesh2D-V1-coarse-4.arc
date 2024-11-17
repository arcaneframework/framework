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

   <refinement-2d>
     <position>5.0 0.5</position>
     <length>2.2 1.3</length>
   </refinement-2d>
   <refinement-2d>
     <position>1.0 4.0</position>
     <length>2.2 2.2</length>
   </refinement-2d>

   <refinement-2d>
     <position>4.0 5.0</position>
     <length>3.0 4.0</length>
   </refinement-2d>
   <refinement-2d>
     <position>5.0 3.0</position>
     <length>2.0 2.0</length>
   </refinement-2d>
   <expected-number-of-cells-in-patchs>1600 6400 484 1144 1936 3600 1600</expected-number-of-cells-in-patchs>
   <expected-number-of-ghost-cells-in-patchs>516 2064 0 0 528 720 960</expected-number-of-ghost-cells-in-patchs>
   <nodes-uid-hash>390f7e52dece2f3c6e83c63c1bba7adf</nodes-uid-hash>
   <faces-uid-hash>873c8d5a520f07b7c63eabd973a0d657</faces-uid-hash>
   <cells-uid-hash>922ed4e573d0b7905e2a276199b88dab</cells-uid-hash>
   <nodes-direction-hash>ce2bc68ab338c9bac278585de96d464a</nodes-direction-hash>
   <faces-direction-hash>341b0eff332739d7f8e77a2898c3ac87</faces-direction-hash>
   <cells-direction-hash>b66d6fe87ccb745773bb14d810b9b871</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
