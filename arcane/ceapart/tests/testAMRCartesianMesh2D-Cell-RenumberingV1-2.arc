<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh 2D Cell Renumbering V1 (Variant 2)</titre>

  <description>Test du raffinement d'un maillage cartesian 2D avec le type d'AMR Cell et la renumerotation V1</description>

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
   <nodes-direction-hash>8bd289ee9a585ecd8bcad81b8c925a6c</nodes-direction-hash>
   <faces-direction-hash>017f27669f0e2764a067f9389b8dee80</faces-direction-hash>
   <cells-direction-hash>d8d050b8ab575659333f73943ca2c699</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
