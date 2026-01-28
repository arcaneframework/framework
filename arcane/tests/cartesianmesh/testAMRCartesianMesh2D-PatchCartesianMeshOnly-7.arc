<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D PatchCartesianMeshOnly (Variant 7)</titre>

    <description>
      Test du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly et avec deux couches
      de mailles de recouvrement.
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
        <lx nx='12'>12.0</lx>
        <ly ny='12'>12.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <refinement-2d>
      <position>1.0 3.0</position>
      <length>4.0 4.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>5.0 5.0</position>
      <length>3.0 5.0</length>
    </refinement-2d>

    <refinement-2d>
      <position>1.0 3.0</position>
      <length>4.0 4.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>5.0 5.0</position>
      <length>3.0 5.0</length>
    </refinement-2d>

    <refinement-2d>
      <position>1.0 3.0</position>
      <length>4.0 4.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>5.0 5.0</position>
      <length>3.0 5.0</length>
    </refinement-2d>

    <overlap-layer-size-top-level>4</overlap-layer-size-top-level>

    <expected-number-of-cells-in-patchs>144 224 252 576 560 1600 1536</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>52 136 144 320 448 864 1536</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>668bf67ecece0ba53b3eb9d39fb6d048</nodes-uid-hash>
    <faces-uid-hash>04530763e30494215f7da14aafa1a8d8</faces-uid-hash>
    <cells-uid-hash>7be5b5bdcb0d69b688df2de6b1f09fca</cells-uid-hash>

    <nodes-direction-hash>a9f892ef624bed2e716e1d660cad5905</nodes-direction-hash>
    <faces-direction-hash>510c70db1a68d44c952031277a77f660</faces-direction-hash>
    <cells-direction-hash>9a97c69423c410f116b0c669d3fde77c</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter" />
  </arcane-protections-reprises>
</cas>
