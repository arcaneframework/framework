<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Coarse PatchCartesianMeshOnly (Variant 3)</titre>

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
      <position>0.0 0.0 0.0</position>
      <length>8.0 8.0 8.0</length>
    </refinement-3d>
    <expected-number-of-cells-in-patchs>64 512 4096</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>448 3584 28672</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>2f858eadfbabd4f78ed4dd9e0f14c548</nodes-uid-hash>
    <faces-uid-hash>e35ce2254ab2fa4c4213656e022939a9</faces-uid-hash>
    <cells-uid-hash>d6c5dea0caeb38b6be110739aaca7721</cells-uid-hash>
    <nodes-direction-hash>211d4e251942ab55a6b66c321eccc9f7</nodes-direction-hash>
    <faces-direction-hash>0a48d41ce4dcb712dcaf331154efae9a</faces-direction-hash>
    <cells-direction-hash>8a8b8e862f8b945cc4e86cad3a097f20</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
