<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 2D PatchCartesianMeshOnly Coarse Zone (Variant 3)</title>

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

    <expected-number-of-cells-in-patchs>25 32</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>228ccabec148d8994007ec68e78ff7d4</nodes-uid-hash>
    <faces-uid-hash>2a641fe98a56f0938992201e96d1cee8</faces-uid-hash>
    <cells-uid-hash>f3b6adc61a780f25ff6580c7c9f39142</cells-uid-hash>
    <nodes-direction-hash>c9a52b9949dc01d2f96eef356ffdd8c6</nodes-direction-hash>
    <faces-direction-hash>66f244a79b230df70dec93aa16ea622f</faces-direction-hash>
    <cells-direction-hash>9729d1bc9dd344ca1632b0d4ead1418c</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
