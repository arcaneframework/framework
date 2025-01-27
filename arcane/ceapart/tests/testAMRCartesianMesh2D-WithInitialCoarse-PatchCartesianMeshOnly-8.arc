<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 8)</titre>

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
        <lx nx='80' prx='1.0'>8.0</lx>
        <ly ny='80' pry='1.0'>8.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <verbosity-level>0</verbosity-level>
    <dump-svg>false</dump-svg>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-2d>
      <position>0.0 0.0</position>
      <length>1.1 1.1</length>
    </refinement-2d>

    <refinement-2d>
      <position>5.0 0.5</position>
      <length>2.2 1.3</length>
    </refinement-2d>
    <refinement-2d>
      <position>1.0 4.0</position>
      <length>2.2 2.2</length>
    </refinement-2d>

    <refinement-2d>
      <position>4.0 5.0</position>
      <length>3.0 4.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>5.0 3.0</position>
      <length>2.0 2.0</length>
    </refinement-2d>
    <expected-number-of-cells-in-patchs>1600 6400 484 1144 1936 3600 1600</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>516 2064 0 0 528 720 960</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>2ec1e04de3e4927fcb7578afedba1446</nodes-uid-hash>
    <faces-uid-hash>52335db38b14b6c2655b35770ea92dd2</faces-uid-hash>
    <cells-uid-hash>66dae987c5a023af83c87cd6131d86f4</cells-uid-hash>
    <nodes-direction-hash>7e24e79d5b9aff75821178382e8f28d1</nodes-direction-hash>
    <faces-direction-hash>930ac5e4d435ae95a6b483f906383b0b</faces-direction-hash>
    <cells-direction-hash>1d9186fdcce9a53553218d893b476c89</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
