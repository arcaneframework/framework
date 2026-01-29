<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D PatchCartesianMeshOnly (Variant 5)</titre>

    <description>
      Test du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly et avec mailles
      de recouvrement.
    </description>

    <boucle-en-temps>AMRCartesianMeshTestLoop</boucle-en-temps>

    <modules>
      <module name="ArcanePostProcessing" active="true" />
      <module name="ArcaneCheckpoint" active="true" />
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


  <maillage amr-type="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2</nsd>
        <origine>0.0 0.0</origine>
        <lx nx='2'>4.0</lx>
        <ly ny='2'>4.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <refinement-2d>
      <position>0.0 0.0</position>
      <length>4.0 2.0</length>
    </refinement-2d>
    <refinement-2d>
      <position>1.0 1.0</position>
      <length>2.0 1.0</length>
    </refinement-2d>
    <overlap-layer-size-top-level>0</overlap-layer-size-top-level>
    <expected-number-of-cells-in-patchs>4 16 8</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>12 48 24</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>d47ab391c4a2f43ce4db0fb8315c524e</nodes-uid-hash>
    <faces-uid-hash>9dce494427c85b629522b94d86f7a65f</faces-uid-hash>
    <cells-uid-hash>3e4181d288a5a191e33374f91cd42892</cells-uid-hash>

    <nodes-direction-hash>22ce00575b01657a87341ca084677dfd</nodes-direction-hash>
    <faces-direction-hash>8fafa79105aed5e20c394b8b722f7737</faces-direction-hash>
    <cells-direction-hash>30d50c687d1c9f9aedc7f487161beec7</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter" />
  </arcane-protections-reprises>
</cas>
