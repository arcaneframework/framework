<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Test CartesianMesh 3D PatchCartesianMeshOnly Coarse Zone (Variant 1)</title>

    <description>
      Test du raffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly
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
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='5' prx='1.0'>5.0</lx>
        <ly ny='5' pry='1.0'>5.0</ly>
        <lz nz='5' prz='1.0'>5.0</lz>
      </cartesian>
    </meshgenerator>
  </mesh>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>

    <refinement-3d>
      <position>1.0 1.0 1.0</position>
      <length>3.0 3.0 3.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>2.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </refinement-3d>

    <coarse-zone-3d>
      <position>2.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </coarse-zone-3d>
    <coarse-zone-3d>
      <position>2.0 2.0 2.0</position>
      <length>1.0 1.0 1.0</length>
    </coarse-zone-3d>


    <!--    <expected-number-of-cells-in-patchs>125 208</expected-number-of-cells-in-patchs>-->
    <expected-number-of-cells-in-patchs>125 8 16 16 16 32 48 72</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>df3786d49d37ad67e721c927500d6fd1</nodes-uid-hash>
    <faces-uid-hash>fde2439e36a81ef94dc2fbcf88fd750d</faces-uid-hash>
    <cells-uid-hash>0e51a40d9d1a79d115f4bb5ddeefd827</cells-uid-hash>
    <nodes-direction-hash>7b972b16de85388414d158e1d0d0f934</nodes-direction-hash>
    <faces-direction-hash>e530a9f3b04d4794e58e53b8e0351750</faces-direction-hash>
    <cells-direction-hash>f5756692579d9efafeade38c72812304</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-checkpoint>
    <checkpoint-service name="ArcaneBasic2CheckpointWriter"/>
  </arcane-checkpoint>
</case>
