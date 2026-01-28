<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Coarse PatchCartesianMeshOnly (Variant 1)</titre>

    <description>Test du déraffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly</description>

    <boucle-en-temps>AMRCartesianMeshTestLoop</boucle-en-temps>

    <modules>
      <module name="ArcanePostProcessing" active="true"/>
      <module name="ArcaneCheckpoint" active="true"/>
    </modules>
  </arcane>

  <arcane-post-traitement>
    <periode-sortie>1</periode-sortie>
    <depouillement>
      <variable>Density</variable>
      <variable>NodeDensity</variable>
      <groupe>AllCells</groupe>
      <groupe>AllNodes</groupe>
      <groupe>AMRPatchCells0</groupe>
      <groupe>AMRPatchCells1</groupe>
      <groupe>AMRPatchCells2</groupe>
    </depouillement>
  </arcane-post-traitement>

  <maillage amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='4'>8.0</lx>
        <ly ny='4'>8.0</ly>
        <lz nz='4'>8.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-3d>
      <position>0.0 0.0 0.0</position>
      <length>8.0 8.0 8.0</length>
    </refinement-3d>
    <expected-number-of-cells-in-patchs>8 64 512</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>56 448 3584</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>edb045323bdca6fddabb5715cfad0689</nodes-uid-hash>
    <faces-uid-hash>be24e6e593b925b836e14e77e71b2fa8</faces-uid-hash>
    <cells-uid-hash>810038612b615339d58e9df030ecac6f</cells-uid-hash>
    <nodes-direction-hash>89d8e402a49d2762e21e552fbcb7b179</nodes-direction-hash>
    <faces-direction-hash>740e42da191ee7a74e0fbe55f250de00</faces-direction-hash>
    <cells-direction-hash>f50094a413133995e237857d606ff73c</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
