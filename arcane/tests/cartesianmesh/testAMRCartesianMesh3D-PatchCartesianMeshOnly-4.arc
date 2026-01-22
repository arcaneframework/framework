<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D PatchCartesianMeshOnly (Variant 4)</titre>

    <description>Test du raffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly</description>

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
        <lx nx='2'>4.0</lx>
        <ly ny='2'>4.0</ly>
        <lz nz='2'>4.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <refinement-3d>
      <position>0.0 0.0 0.0</position>
      <length>4.0 4.0 4.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>1.0 1.0 1.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <expected-number-of-cells-in-patchs>8 64 64</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>2a94c530c6fc1e564f43fcdc26673d45</nodes-uid-hash>
    <faces-uid-hash>aca4c7354239d631906cbbcaddfce4b2</faces-uid-hash>
    <cells-uid-hash>a02bbff14ec87be39f4672a8cd936501</cells-uid-hash>
    <nodes-direction-hash>a0560ca2bb8199808562e8cdd7e91549</nodes-direction-hash>
    <faces-direction-hash>7bade7add00635817cdb7094eb042a9a</faces-direction-hash>
    <cells-direction-hash>03b9105a6a18fa65bab7f57384453a01</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
