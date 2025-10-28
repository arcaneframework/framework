<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 3D Coaren PatchCartesianMeshOnly Coarse Zone (Variant 1)</title>

    <description>
      Test du dé-raffinement initial,
      puis du raffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly,
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
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='8' prx='1.0'>8.0</lx>
        <ly ny='8' pry='1.0'>8.0</ly>
        <lz nz='8' prz='1.0'>8.0</lz>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>

    <refinement-3d>
      <position>1.0 1.0 1.0</position>
      <length>6.0 6.0 6.0</length>
    </refinement-3d>

    <coarse-zone-3d>
      <position>1.0 1.0 1.0</position>
      <length>6.0 4.0 4.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>2.0 2.0 2.0</position>
      <length>4.0 2.0 2.0</length>
    </coarse-zone-3d>

    <!--    <expected-number-of-cells-in-patchs>64 496 960</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>64 384 576 32 48 64 96 128 128</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>f1eedebec73b1051ef2b7a77ed815702</nodes-uid-hash>
    <faces-uid-hash>0b1bba1f8971df16968f9648112dcbb1</faces-uid-hash>
    <cells-uid-hash>85f1a6bff5d48797f358c0145fb9127c</cells-uid-hash>
    <nodes-direction-hash>a362e7707f0455ec1fdc7e787488bc36</nodes-direction-hash>
    <faces-direction-hash>acf1ed66f073090ef1254410deb54fdb</faces-direction-hash>
    <cells-direction-hash>0a299b23e95a42809e3e7b1156b78039</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
