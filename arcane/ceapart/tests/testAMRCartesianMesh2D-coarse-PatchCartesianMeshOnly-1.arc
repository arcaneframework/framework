<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 1)</titre>

    <description>Test du déraffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly
    </description>

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


  <maillage amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='4'>8.0</lx>
        <ly ny='4'>8.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-2d>
      <position>0.0 0.0</position>
      <length>8.0 8.0</length>
    </refinement-2d>
    <expected-number-of-cells-in-patchs>4 16 64</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>4a6217f9352c7168e50c4936d24dcfd4</nodes-uid-hash>
    <faces-uid-hash>2b1d775d942edb86c900dcfd5ec34963</faces-uid-hash>
    <cells-uid-hash>b4393f81ee32b8d0d1f58c2199307cb9</cells-uid-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
