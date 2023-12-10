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
   <nodes-uid-hash>108ab65a162f420b2b49beee10b1f85b</nodes-uid-hash>
   <faces-uid-hash>819e4e765d55fcb976d8b00e4c8d7718</faces-uid-hash>
   <cells-uid-hash>04538c5721f934d717ee13d181b81d76</cells-uid-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
