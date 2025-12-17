<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D PatchCartesianMeshOnly (Variant 4)</titre>

    <description>Test du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly</description>

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
        <lx nx='2'>4.0</lx>
        <ly ny='2'>4.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <refinement-2d>
      <position>0.0 0.0</position>
      <length>4.0 2.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>1.0 1.0</position>
      <length>2.0 1.0</length>
    </refinement-2d>
    <expected-number-of-cells-in-patchs>4 8 8</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>12 24 24</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>4f6e3623644ee01fd476f5f385a86d63</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>3810620acfbfdb8356e0ae1321d234b2</faces-uid-hash>-->
    <faces-uid-hash>89e6cbcf6f5ae7f6ffda7ffb54a9dda5</faces-uid-hash>
    <cells-uid-hash>c308bbf6b06bb6f83ffcaf9669286a77</cells-uid-hash>

    <nodes-direction-hash>11b2632fc81c72880c6bf06f82a37d6e</nodes-direction-hash>
    <!--    <faces-direction-hash>24222edf08d6cac277814f3daa4501a4</faces-direction-hash>-->
    <faces-direction-hash>fbf098b120e158237e383636b52affe1</faces-direction-hash>
    <cells-direction-hash>3273f83c9794fc6bdd1a4382f84e44d0</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter" />
  </arcane-protections-reprises>
</cas>
