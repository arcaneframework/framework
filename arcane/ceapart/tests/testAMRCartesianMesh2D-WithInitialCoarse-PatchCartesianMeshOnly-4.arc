<?xml version="1.0"?>
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <titre>Test CartesianMesh 2D Coarse PatchCartesianMeshOnly (Variant 4)</titre>

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
        <lx nx='8'>8.0</lx>
        <ly ny='8'>8.0</ly>
      </cartesian>
    </meshgenerator>
  </maillage>

  <a-m-r-cartesian-mesh-tester>
    <renumber-patch-method>0</renumber-patch-method>
    <coarse-at-init>true</coarse-at-init>
    <refinement-2d>
      <position>2.0 2.0</position>
      <length>4.0 2.0</length>
    </refinement-2d>
    <expected-number-of-cells-in-patchs>16 64 32</expected-number-of-cells-in-patchs>
    <expected-number-of-ghost-cells-in-patchs>48 192 96</expected-number-of-ghost-cells-in-patchs>
    <nodes-uid-hash>bfc28548e16d92620894553886d14e10</nodes-uid-hash>
    <!-- Hash avant renumérotation niveau 0. -->
    <!--    <faces-uid-hash>259a096743a77ac3161c5d5caa2fa195</faces-uid-hash>-->
    <faces-uid-hash>70a8d404324d63a73ec54eea9a66983c</faces-uid-hash>
    <cells-uid-hash>62a70cd12a7cea4ffd3e978d131daa58</cells-uid-hash>

    <nodes-direction-hash>01dfe69b9523f60aa5b2272d9c0f5956</nodes-direction-hash>
    <!--    <faces-direction-hash>41622c019ba3355990b889919f04e828</faces-direction-hash>-->
    <faces-direction-hash>e3f9052522b7300f2c510a19f73ebb9d</faces-direction-hash>
    <cells-direction-hash>15285920a71247986bf03915c10cdf69</cells-direction-hash>
  </a-m-r-cartesian-mesh-tester>

  <arcane-protections-reprises>
    <service-protection name="ArcaneBasic2CheckpointWriter"/>
  </arcane-protections-reprises>
</cas>
