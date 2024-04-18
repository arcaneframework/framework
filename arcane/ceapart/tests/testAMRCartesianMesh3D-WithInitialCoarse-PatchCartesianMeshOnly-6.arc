<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Coarse PatchCartesianMeshOnly (Variant 6)</titre>

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
        <nsd>2 2 2</nsd>
        <origine>0.0 0.0 0.0</origine>
        <lx nx='20' prx='1.0'>4.0</lx>
        <ly ny='40' pry='1.0'>12.0</ly>
        <lz nz='12' prz='1.0'>2.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <verbosity-level>0</verbosity-level>
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
    <expected-number-of-cells-in-patchs>1200 9600 1440 1440</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>912 7296 1920 2400</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>116ad9966194965be13328f287dac0e5</nodes-uid-hash>
    <faces-uid-hash>f61273cea361175b897d72924b1b55d1</faces-uid-hash>
    <cells-uid-hash>e5defe21854a4b80e6c36efaa946b0d7</cells-uid-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter" />
  </arcane-protections-reprises>
</cas>