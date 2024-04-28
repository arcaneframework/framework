<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D Coarse PatchCartesianMeshOnly (Variant 7)</titre>

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
        <lx nx='40' prx='1.0'>4.0</lx>
        <ly ny='60' pry='1.0'>12.0</ly>
        <lz nz='20' prz='1.0'>2.0</lz>
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
    <refinement-3d>
      <position>1.4 3.0 1.0</position>
      <length>0.5 1.0 0.5</length>
    </refinement-3d>
    <refinement-3d>
      <position>2.4 4.0 0.5</position>
      <length>0.2 0.4 0.4</length>
    </refinement-3d>
    <expected-number-of-cells-in-patchs>6000 48000 8000 8000 64000 128</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>2448 19584 5440 5440 43520 32</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>644e07670c466bf30a4f3fcd664e0a16</nodes-uid-hash>
    <faces-uid-hash>734b0d14dcbd202dc7a0b20941da18ea</faces-uid-hash>
    <cells-uid-hash>b69cbc48c5ab88a146526f6e49497886</cells-uid-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter" />
  </arcane-protections-reprises>
</cas>
