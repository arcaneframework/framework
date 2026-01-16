<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D PatchCartesianMeshOnly (Variant 5)</titre>

    <description>Test du raffinement d'un maillage cartesian 2D avec le type d'AMR PatchCartesianMeshOnly</description>

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
        <lx nx='4'>4.0</lx>
        <ly ny='4'>4.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <refinement-2d>
      <position>0.0 0.0</position>
      <length>2.0 2.0</length>
    </refinement-2d>
    <expected-number-of-cells-in-patchs>16 16</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>20 20</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>d9bda47b8232ef43ac7a8d86d193090a</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>542d7f574af6aa7c8352ab30fd6e1756</faces-uid-hash>-->
    <faces-uid-hash>0dc25efa3d0c49eeeb48fd01a21645db</faces-uid-hash>
    <cells-uid-hash>d3d68d4ddecd3bde5738ac942e17f3b9</cells-uid-hash>
    <nodes-direction-hash>56d01ab8ec9cdaba5fc092ae1680afe8</nodes-direction-hash>
    <!--    <faces-direction-hash>aabfb9b5e481a88ae2b485d6f20fd3cf</faces-direction-hash>-->
    <faces-direction-hash>5618d0b768afe7f57b328506e213a62a</faces-direction-hash>
    <cells-direction-hash>953548f021995149882b9b7c19849ff1</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter" />
  </arcane-protections-reprises>
</cas>
