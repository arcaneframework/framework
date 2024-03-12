<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh 2D Cell Renumbering V1 (Variant 1)</titre>

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
       <lx nx='2' prx='1.0'>2.0</lx>
       <lx nx='4' prx='1.2'>3.0</lx>
       <lx nx='5' prx='1.3'>3.0</lx>
       <lx nx='6' prx='1.4'>4.0</lx>

       <ly ny='2' pry='1.0'>2.0</ly>
       <ly ny='3' pry='1.1'>4.0</ly>
       <ly ny='4' pry='1.3'>5.0</ly>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <renumber-patch-method>1</renumber-patch-method>
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
   <expected-number-of-cells-in-patchs>153 8 4 80 128</expected-number-of-cells-in-patchs>
   <nodes-uid-hash>eef03075d1b1d32b63c9014bc567704e</nodes-uid-hash>
   <faces-uid-hash>6d1b4ec0963f467f704c08f3f50dbd73</faces-uid-hash>
   <cells-uid-hash>a577cb5e34dcbac51cec397ff7cb684e</cells-uid-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
