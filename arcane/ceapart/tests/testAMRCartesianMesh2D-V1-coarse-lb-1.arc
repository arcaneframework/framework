<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
 <arcane>
  <titre>Test CartesianMesh</titre>

  <description>Test des maillages cartesiens AMR 2D</description>

  <boucle-en-temps>AMRCartesianMeshTestLoop</boucle-en-temps>

  <modules>
    <module name="ArcanePostProcessing" active="true" />
    <module name="ArcaneCheckpoint" active="true" />
    <module name="ArcaneLoadBalance" active="true" />
  </modules>

 </arcane>

 <arcane-equilibrage-charge>
   <actif>true</actif>
   <desequilibre-maximal>0.0</desequilibre-maximal>
   <temps-cpu-minimal>0</temps-cpu-minimal>
 </arcane-equilibrage-charge>

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
       <lx nx='4' prx='1.1'>3.0</lx>
       <lx nx='4' prx='1.2'>3.0</lx>
       <lx nx='6' prx='1.3'>4.0</lx>

       <ly ny='2' pry='1.0'>2.0</ly>
       <ly ny='6' pry='1.1'>4.0</ly>
       <ly ny='4' pry='1.2'>5.0</ly>
     </cartesian>
   </meshgenerator>
 </maillage>

 <a-m-r-cartesian-mesh-tester>
   <renumber-patch-method>1</renumber-patch-method>
   <coarse-at-init>true</coarse-at-init>
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
   <expected-number-of-cells-in-patchs>48 192 12 8 64 72</expected-number-of-cells-in-patchs>
   <expected-number-of-ghost-cells-in-patchs>32 128 4 8 76 72</expected-number-of-ghost-cells-in-patchs>
   <nodes-uid-hash></nodes-uid-hash>
   <faces-uid-hash></faces-uid-hash>
   <cells-uid-hash></cells-uid-hash>
   <nodes-uid-hash>553b7ae4b2f540a15650369dad6e857c</nodes-uid-hash>
   <faces-uid-hash>ea76dd5830a9d62b4e1b190dfdcaa9aa</faces-uid-hash>
   <cells-uid-hash>807d68b16f89fe2b78af26b3d74661b7</cells-uid-hash>
   <nodes-direction-hash>bc756d56298def13cb83acb56738652e</nodes-direction-hash>
   <faces-direction-hash>2a7795b689b0cedcaa1f0bd2ee5f5619</faces-direction-hash>
   <cells-direction-hash>775f249f3fb3892d8cf5212a0cb06edc</cells-direction-hash>
 </a-m-r-cartesian-mesh-tester>

 <arcane-protections-reprises>
   <service-protection name="ArcaneBasic2CheckpointWriter" />
 </arcane-protections-reprises>
</cas>
