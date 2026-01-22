<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh 3D Cell Renumbering V2 (Variant 1)</titre>

  <description>Test du raffinement d'un maillage cartesian 3D avec le type d'AMR Cell et la renumérotation V2</description>

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
       <nsd>2 2 1</nsd>
       <origine>0.0 0.0 0.0</origine>
       <lx nx='2' prx='1.0'>2.0</lx>
       <lx nx='4' prx='1.2'>3.0</lx>
       <lx nx='5' prx='1.3'>3.0</lx>
       <lx nx='6' prx='1.4'>4.0</lx>

       <ly ny='2' pry='1.0'>2.0</ly>
       <ly ny='3' pry='1.1'>4.0</ly>
       <ly ny='4' pry='1.3'>5.0</ly>

       <lz nz='8' prz='1.0'>2.0</lz>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <renumber-patch-method>2</renumber-patch-method>
   <refinement-3d>
     <position>1.0 2.0 -0.1</position>
     <length>2.0 4.0 0.5</length>
   </refinement-3d>
   <refinement-3d>
     <position>1.4 3.0 0.3</position>
     <length>0.5 1.0 1.0</length>
   </refinement-3d>
   <!-- <refinement-3d>
     <position>4.0 5.0 0.2</position>
     <length>3.0 4.0 0.8</length>
   </refinement-3d> -->
   <expected-number-of-cells-in-patchs>1224 144 40</expected-number-of-cells-in-patchs>
   <nodes-uid-hash>3ef51546e83d8303b770300b808f617e</nodes-uid-hash>
   <faces-uid-hash>460c3b5083dce8f2c88d1e2b68d28c7f</faces-uid-hash>
   <cells-uid-hash>91b7c816009cca00c223d088d86f9a1e</cells-uid-hash>
   <nodes-direction-hash>e41422b49ead384b0e355b3b18551136</nodes-direction-hash>
   <faces-direction-hash>41314c62bfdd308f0ee049cd0a8e2e55</faces-direction-hash>
   <cells-direction-hash>af74f7091afe3e27a1c816ea258da85b</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
