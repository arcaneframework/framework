<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 2D PatchCartesianMeshOnly Coarse Zone (Variant 1)</title>

    <description>
      Test du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly
      puis du dé-raffinement de certaines zones (sans renumérotation)
    </description>

    <timeloop>AMRCartesianMeshTestLoop</timeloop>

    <modules>
      <module name="ArcanePostProcessing" active="true"/>
      <module name="ArcaneCheckpoint" active="true"/>
    </modules>

  </arcane>

  <arcane-post-processing>
    <output-period>1</output-period>
    <output>
      <variable>Density</variable>
      <variable>NodeDensity</variable>
      <group>AllCells</group>
      <group>AllNodes</group>
      <group>AllFacesDirection0</group>
      <group>AllFacesDirection1</group>
    </output>
  </arcane-post-processing>


  <mesh amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='5' prx='1.0'>5.0</lx>
        <ly ny='5' pry='1.0'>5.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>

    <refinement-2d>
      <position>1.0 1.0</position>
      <length>3.0 3.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </refinement-2d>

    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>1.0 1.0</length>
    </coarse-zone-2d>

    <!--    <expected-number-of-cells-in-patchs>25 32</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>25 8 8 8 8</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>228ccabec148d8994007ec68e78ff7d4</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>2a641fe98a56f0938992201e96d1cee8</faces-uid-hash>-->
    <faces-uid-hash>72db90136e254db309faaa5351174a38</faces-uid-hash>
    <cells-uid-hash>f3b6adc61a780f25ff6580c7c9f39142</cells-uid-hash>
    
    <nodes-direction-hash>08ab40806c58a2025313aa45658d9a13</nodes-direction-hash>
    <faces-direction-hash>42c28bc7f547a42be25cce44e8778a91</faces-direction-hash>
    <cells-direction-hash>80b801ca138491629baa42ef67d1a7fc</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
