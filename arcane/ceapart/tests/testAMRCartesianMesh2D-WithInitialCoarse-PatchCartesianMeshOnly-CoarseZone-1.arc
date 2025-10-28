<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 2D Coaren PatchCartesianMeshOnly Coarse Zone (Variant 1)</title>

    <description>
      Test du dé-raffinement initial,
      puis du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly,
      puis du dé-raffinement de certaines zones
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
        <lx nx='8' prx='1.0'>8.0</lx>
        <ly ny='8' pry='1.0'>8.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>

    <refinement-2d>
      <position>1.0 1.0</position>
      <length>6.0 6.0</length>
    </refinement-2d>

    <coarse-zone-2d>
      <position>1.0 1.0</position>
      <length>6.0 4.0</length>
    </coarse-zone-2d>
    <coarse-zone-2d>
      <position>2.0 2.0</position>
      <length>4.0 2.0</length>
    </coarse-zone-2d>

    <!--    <expected-number-of-cells-in-patchs>16 56 48</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>16 48 8 16 16 16</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>bc99948909b1c4499d164c6da40dcad9</nodes-uid-hash>
    <faces-uid-hash>f4914df7549e82882fc28849b89c8b5f</faces-uid-hash>
    <cells-uid-hash>6e75535e29f9cb664058906a6e54c0a6</cells-uid-hash>
    <nodes-direction-hash>aadacb309fd2f436af0c8fd4959af8e7</nodes-direction-hash>
    <faces-direction-hash>9897ea6879fa40978a487931568562cf</faces-direction-hash>
    <cells-direction-hash>4d240366d0b76e05b254e0da4ead19ef</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
