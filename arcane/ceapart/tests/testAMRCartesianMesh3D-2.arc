<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh</titre>

  <description>Test des maillages cartesiens AMR 3D</description>

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
    <groupe>AMRPatchCells0</groupe>
    <groupe>AMRPatchCells1</groupe>
    <groupe>AMRPatchCells2</groupe>
   </depouillement>
 </arcane-post-traitement>

 <maillage amr="true">
   <meshgenerator>
     <cartesian>
       <nsd>2 2 2</nsd>
       <origine>0.0 0.0 0.0</origine>
       <lx nx='2'>4.0</lx>
       <ly ny='2'>4.0</ly>
       <lz nz='2'>4.0</lz>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <refinement-3d>
     <position>0.0 0.0 0.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>0.0 0.0 2.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>0.0 2.0 0.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>0.0 2.0 2.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>2.0 0.0 0.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>2.0 0.0 2.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>2.0 2.0 0.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
   <expected-number-of-cells-in-patchs>8 8 8 8 8 8 8 8</expected-number-of-cells-in-patchs>
   <nodes-uid-hash>4924c41673a98fa2b7ca257f84c75dd2</nodes-uid-hash>
   <faces-uid-hash>3098df318d2f128195062aed3f8e3118</faces-uid-hash>
   <cells-uid-hash>e8b1b15951e58e18300ba9395f316418</cells-uid-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
