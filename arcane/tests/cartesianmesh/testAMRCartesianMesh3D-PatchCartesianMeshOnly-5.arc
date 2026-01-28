<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 3D PatchCartesianMeshOnly (Variant 3)</titre>

    <description>
      Test du raffinement d'un maillage cartesian 3D avec le type d'AMR PatchCartesianMeshOnly et une couche
      de mailles de recouvrement.
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
        <lx nx='6'>6.0</lx>
        <ly ny='6'>6.0</ly>
        <lz nz='6'>6.0</lz>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <refinement-3d>
      <position>0.0 0.0 0.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>2.0 1.0 1.0</position>
      <length>2.0 2.0 2.0</length>
    </refinement-3d>
    <refinement-3d>
      <position>3.0 3.0 0.0</position>
      <length>1.0 1.0 1.0</length>
    </refinement-3d>
    <overlap-layer-size-top-level>2</overlap-layer-size-top-level>

    <expected-number-of-cells-in-patchs>216 216 512 144</expected-number-of-cells-in-patchs>
<!--    <expected-number-of-cells-in-patchs>216 64 64 8</expected-number-of-cells-in-patchs>-->
    <nodes-uid-hash>4b16c11b780f8806b4af41259472b207</nodes-uid-hash>
    <faces-uid-hash>c90207c07725934656ffedb5ca27d59b</faces-uid-hash>
    <cells-uid-hash>48f51f5e4f400513931718de45b2f314</cells-uid-hash>
    <nodes-direction-hash>24e2c1553bfc4abd9ee7106d0aa90421</nodes-direction-hash>
    <faces-direction-hash>0d0b26f7117d592b33d90f26c61ff69c</faces-direction-hash>
    <cells-direction-hash>7c671a75f54a27c6f893bbb050d8a875</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
