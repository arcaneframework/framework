<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Coarse PatchCartesianMeshOnly (Variant 4)</titre>

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

  <maillage amr-type="3" nb-ghostlayer="3" ghostlayer-builder-version="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='8'>8.0</lx>
        <ly ny='8'>8.0</ly>
        <lz nz='8'>8.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-3d>
      <position>2.0 2.0 2.0</position>
      <length>4.0 2.0 6.0</length>
    </refinement-3d>
    <expected-number-of-cells-in-patchs>64 512 384</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>448 3584 2688</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>9a6ecb9085b0f90823a1b98e57abb659</nodes-uid-hash>
    <faces-uid-hash>e6e329dfd3688eb2fc8d7c9721e73d86</faces-uid-hash>
    <cells-uid-hash>5ed44fd6b3d490cef95cfdc4ec3b82eb</cells-uid-hash>
    <nodes-direction-hash>242f80e5b3b8fcc4786a5e25b0559cce</nodes-direction-hash>
    <faces-direction-hash>5e0c8bb5737256b7c58616cf4346d6a8</faces-direction-hash>
    <cells-direction-hash>ed54a31be6004e06c81b98a4c21439bb</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
