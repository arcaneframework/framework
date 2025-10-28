<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D PatchCartesianMeshOnly (Variant 2)</titre>

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
      <position>2.0 2.0 2.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>2.0 0.0 0.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>0.0 2.0 0.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>0.0 0.0 2.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>0.0 0.0 0.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>0.0 2.0 2.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>2.0 0.0 2.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>2.0 2.0 0.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <merge-patches>false</merge-patches>
    <expected-number-of-cells-in-patchs>8 8 8 8 8 8 8 8 8</expected-number-of-cells-in-patchs>
    <nodes-uid-hash>bfa069f213eef90d389efa5c3ca0745d</nodes-uid-hash>
    <faces-uid-hash>5b12ac3a6d9ed116b024074cdef808c6</faces-uid-hash>
    <cells-uid-hash>1ee6fc646290a97f10cef6795ac106f0</cells-uid-hash>
    <nodes-direction-hash>9b08918c51db2517dd59e9c8194708a8</nodes-direction-hash>
    <faces-direction-hash>436913077cf6cda018aef54ca7dd456e</faces-direction-hash>
    <cells-direction-hash>e974c5f17047b2eb7253bff0c937e99a</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
