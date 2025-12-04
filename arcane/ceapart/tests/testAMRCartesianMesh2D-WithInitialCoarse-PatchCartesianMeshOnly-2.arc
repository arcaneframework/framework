<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 2)</titre>

    <description>Test du déraffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly
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
    <expected-number-of-ghost-cells-in-patchs>12 48 192</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>4a6217f9352c7168e50c4936d24dcfd4</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>2b1d775d942edb86c900dcfd5ec34963</faces-uid-hash>-->
    <faces-uid-hash>fd2882117f9c2e0be1ca628ada7e01b7</faces-uid-hash>
    <cells-uid-hash>b4393f81ee32b8d0d1f58c2199307cb9</cells-uid-hash>

    <nodes-direction-hash>1b3b79c88b906c7ea30055461ab40dc3</nodes-direction-hash>
    <!--    <faces-direction-hash>24d088e8a475328d0d8bbc6a80568282</faces-direction-hash>-->
    <faces-direction-hash>f80dfd53f8f74771489acf74ad5e416a</faces-direction-hash>
    <cells-direction-hash>d060d0a4629df55c0e31deb1a9a9fbc3</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
