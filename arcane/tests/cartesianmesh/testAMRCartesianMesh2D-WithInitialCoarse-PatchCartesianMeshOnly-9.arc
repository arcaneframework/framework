<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 2 du variant 4)</titre>

    <description>
      Test du déraffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly et avec deux couches
      de mailles de recouvrement.
    </description>

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
      <groupe>AllFacesDirection0</groupe>
      <groupe>AllFacesDirection1</groupe>
    </depouillement>
  </arcane-post-traitement>

  <maillage amr-type="3" nb-ghostlayer="3" ghostlayer-builder-version="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='8'>8.0</lx>
        <ly ny='8'>8.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-2d>
      <position>2.0 2.0</position>
      <length>4.0 2.0</length>
    </refinement-2d>
    <overlap-layer-size-top-level>2</overlap-layer-size-top-level>

    <expected-number-of-cells-in-patchs>16 64 96</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>48 192 288</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>3d4d0d43592225c40cd482c5ba56d737</nodes-uid-hash>
    <faces-uid-hash>17e53282e91abd21138f00308a8b3bdd</faces-uid-hash>
    <cells-uid-hash>62cdf368ef95d43e8e3e0606e9b32f11</cells-uid-hash>

    <nodes-direction-hash>2c31e5d69a79f2dbfcfcd350a76d1794</nodes-direction-hash>
    <faces-direction-hash>b26717bc3079a953e9696d18b7d840ad</faces-direction-hash>
    <cells-direction-hash>352e5bfb089a051374ce457246ea2c55</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
