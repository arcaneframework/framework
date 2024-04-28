<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 6)</titre>

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
        <lx nx='2' prx='1.0'>2.0</lx>
        <lx nx='4' prx='1.1'>3.0</lx>
        <lx nx='4' prx='1.2'>3.0</lx>
        <lx nx='6' prx='1.3'>4.0</lx>

        <ly ny='2' pry='1.0'>2.0</ly>
        <ly ny='6' pry='1.1'>4.0</ly>
        <ly ny='4' pry='1.2'>5.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-2d>
      <position>1.0 2.0</position>
      <length>1.0 2.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>1.4 3.0</position>
      <length>0.5 1.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>4.0 5.0</position>
      <length>3.0 4.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>5.0 7.0</position>
      <length>2.0 2.0</length>
    </refinement-2d>
    <expected-number-of-cells-in-patchs>48 192 12 8 64 72</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>72 288 12 8 160 168</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>133eed4a37931e6cd103f2fb12c354eb</nodes-uid-hash>
    <faces-uid-hash>0ba793345bd95bd20e83fc7cd6de9f02</faces-uid-hash>
    <cells-uid-hash>7eec15be89bd293981c7b92df927cb78</cells-uid-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
