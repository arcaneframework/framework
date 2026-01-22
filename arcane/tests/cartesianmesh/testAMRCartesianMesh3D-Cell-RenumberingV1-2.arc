<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh 3D Cell Renumbering V1 (Variant 2)</titre>

  <description>Test du raffinement d'un maillage cartesian 3D avec le type d'AMR Cell et la renumerotation V1</description>

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
   <renumber-patch-method>1</renumber-patch-method>
   <refinement-3d>
     <position>0.0 0.0 0.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>0.0 2.0 0.0</position>
     <length>2.0 2.0 2.0</length>
   </refinement-3d>
    <refinement-3d>
     <position>0.0 0.0 2.0</position>
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
   <nodes-uid-hash>3cbd376d768e895a0b8e33d091bb3ff5</nodes-uid-hash>
   <faces-uid-hash>2582173a840acc76dabb2abf8528410b</faces-uid-hash>
   <cells-uid-hash>a39d1e99166f13d23ffb74d4965b06a2</cells-uid-hash>
   <nodes-direction-hash>a7b637176f0ef9d990b6c516494aa9ac</nodes-direction-hash>
   <faces-direction-hash>213c4b9e07dbe47ed61920b06a8bd0bf</faces-direction-hash>
   <cells-direction-hash>3a7d81687858cd6957d92f6f08ee3cf5</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
