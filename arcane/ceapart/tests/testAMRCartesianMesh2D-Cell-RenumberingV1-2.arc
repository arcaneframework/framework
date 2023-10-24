<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh 2D Cell Renumbering V1 (Variant 2)</titre>

  <description>Test du raffinement d'un maillage cartesian 2D avec le type d'AMR Cell et la renumérotation V1</description>

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
   <renumber-patch-method>1</renumber-patch-method>
   <refinement-2d>
     <position>2.0 0.0</position>
     <length>2.0 2.0</length>
   </refinement-2d>
    <refinement-2d>
     <position>0.0 0.0</position>
     <length>2.0 2.0</length>
   </refinement-2d>
   <expected-number-of-cells-in-patchs>4 4 4</expected-number-of-cells-in-patchs>
   <nodes-uid-hash>5fe67ed5bc2ee2112947e7db8fad5ed4</nodes-uid-hash>
   <faces-uid-hash>e27ee8eef5bb304b1f09ca3286a5980e</faces-uid-hash>
   <cells-uid-hash>cee8747ee64863eca941b41345f23f9d</cells-uid-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
