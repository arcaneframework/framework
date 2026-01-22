<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 7)</titre>

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
    <dump-svg>true</dump-svg>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-2d>
      <position>0.0 0.0</position>
      <length>1.1 1.1</length>
    </refinement-2d>

    <expected-number-of-cells-in-patchs>1600 6400 484</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>516 2064 0</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>d908f0b0fb7d7b8f5191444cee921518</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>7a55f9bbcd8ffe328455b774c619bb34</faces-uid-hash>-->
    <faces-uid-hash>f2b6f769b25c76e345530e4526f17fec</faces-uid-hash>
    <cells-uid-hash>cfa32d74b8f1ba19ab55eae6e84ce87a</cells-uid-hash>

    <nodes-direction-hash>04f730c075fc1e636790583d4e8b8646</nodes-direction-hash>
    <!--    <faces-direction-hash>ff5f7573b0107d492962605201ce6499</faces-direction-hash>-->
    <faces-direction-hash>23558cab23146443187e7ea9da0b3cb8</faces-direction-hash>
    <cells-direction-hash>bcd8f765aa2599913658588408bd995b</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
