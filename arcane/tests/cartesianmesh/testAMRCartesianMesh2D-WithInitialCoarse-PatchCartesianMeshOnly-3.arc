<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 3)</titre>

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
        <lx nx='8'>8.0</lx>
        <ly ny='8'>8.0</ly>
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
    <expected-number-of-cells-in-patchs>16 64 256</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>48 192 768</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>022636abec5864f5eba9bfede0d353d8</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>e336302a8acf3ceff59d983362b3f430</faces-uid-hash>-->
    <faces-uid-hash>97729f00685b8fe898ea64bd0652e182</faces-uid-hash>
    <cells-uid-hash>2508f369f438bec453bde7439dfab6c3</cells-uid-hash>

    <nodes-direction-hash>23f32dabc9a6199b0f77523630cc2a7c</nodes-direction-hash>
    <!--    <faces-direction-hash>05aa80941c48106bc0bfb14a4713a540</faces-direction-hash>-->
    <faces-direction-hash>2be9af18f378a737d3bf81039bba9e29</faces-direction-hash>
    <cells-direction-hash>a497a5f8d77c6dab70ccb414f7d9ae20</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
