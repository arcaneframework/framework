<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Coarse PatchCartesianMeshOnly (Variant 5)</titre>

    <description>Test du déraffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly</description>

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
      <groupe>AMRPatchCells0</groupe>
      <groupe>AMRPatchCells1</groupe>
      <groupe>AMRPatchCells2</groupe>
    </depouillement>
  </arcane-post-traitement>

  <maillage amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2 1</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='3' prx='1.1'>3.0</lx>
        <lx nx='5' prx='1.0'>5.0</lx>

        <ly ny='3' pry='1.0'>3.0</ly>
        <ly ny='5' pry='1.1'>10.0</ly>

        <lz nz='4' prz='1.0'>2.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-3d>
      <position>1.0 2.0 0.5</position>
      <length>1.0 2.0 1.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>1.4 3.0 1.0</position>
      <length>0.5 1.0 0.5</length>
    </refinement-3d>
    <expected-number-of-cells-in-patchs>32 256 32 16</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>40 320 32 16</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>292074b8de182d037fed5361ec74bdce</nodes-uid-hash>
    <faces-uid-hash>ca50bf895683774ee8821dc7b7a18ac5</faces-uid-hash>
    <cells-uid-hash>0ed0ccf01939cd86c83e0af0ac830d21</cells-uid-hash>
    <nodes-direction-hash>79ac1c393e6237b247e695f2b7b3aeff</nodes-direction-hash>
    <faces-direction-hash>8fe664c847fe17d1bc46cd0593ce3033</faces-direction-hash>
    <cells-direction-hash>0fcae39769e6ab2dc095c72770b1ab4e</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter" />
  </arcane-protections-reprises>
</cas>
