<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh 2D Cell Renumbering V2 (Variant 2)</titre>

  <description>Test du raffinement d'un maillage cartesian 2D avec le type d'AMR Cell et la renumérotation V2</description>

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
       <nsd>2 2</nsd>
       <origine>0.0 0.0</origine>
       <lx nx='2'>4.0</lx>
       <ly ny='2'>4.0</ly>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <renumber-patch-method>2</renumber-patch-method>
   <refinement-2d>
     <position>2.0 0.0</position>
     <length>2.0 2.0</length>
   </refinement-2d>
    <refinement-2d>
     <position>0.0 0.0</position>
     <length>2.0 2.0</length>
   </refinement-2d>
   <expected-number-of-cells-in-patchs>4 4 4</expected-number-of-cells-in-patchs>
   <nodes-uid-hash>3a7a75dd3d510a4d523c182a3b76c448</nodes-uid-hash>
   <faces-uid-hash>7cb9c4b04f06f119bd1f0390e275d396</faces-uid-hash>
   <cells-uid-hash>fab8abc5aa12ff95bc06b4e3553f967a</cells-uid-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
