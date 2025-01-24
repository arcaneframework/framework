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
    <renumber-patch-method>1</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-2d>
      <position>0.0 0.0</position>
      <length>1.1 1.1</length>
    </refinement-2d>

    <expected-number-of-cells-in-patchs>1600 6400 484</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>516 2064 0</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>be945c17467f6ba7bbf41ee1baf19c8c</nodes-uid-hash>
    <faces-uid-hash>c92d0e52cea7b406b449d57521f42124</faces-uid-hash>
    <cells-uid-hash>19dd0b9269462bd111611a5f0adac601</cells-uid-hash>
    <nodes-direction-hash>e5c557100202662f5b0f896186988899</nodes-direction-hash>
    <faces-direction-hash>8199a4db6ae7b59a3cd7e8dd5dddc1df</faces-direction-hash>
    <cells-direction-hash>0c31d9305b33eda7aa50f222f7395693</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
